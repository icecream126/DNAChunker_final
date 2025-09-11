import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.modules.mamba_simple import Block  # Legacy mambav1 file structure
except ImportError:
    from mamba_ssm.modules.block import Block  # mambav2 file structure
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn  # Legacy mambav1 file structure
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn  # mambav2 file structure
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .configuration_hnet import HNetConfig


def rotate_half(x):
    """Rotates half the hidden dimensions of the input tensor."""
    # Split the last dimension into two halves
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    # Concatenate with the second half negated
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Precompute theta values for the given dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k):
        """
        Args:
            q (torch.Tensor): Query tensor (batch, n_heads, seq_len, head_dim)
            k (torch.Tensor): Key tensor (batch, n_heads, seq_len, head_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Query and Key with applied RoPE.
        """
        # q, k shape: (batch, n_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        # Create positional indices
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        # Calculate frequency components
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Concatenate for both halves of the dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Unsqueeze for broadcasting across batch and heads
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)

        # Apply rotation
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


# --- Utility: Straight-Through Estimator ---
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # In the forward pass, round to the nearest integer (0 or 1)
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, treat the operation as the identity function
        return grad_output


# --- H-Net Dynamic Chunking (DC) Components ---
class RoutingModule(nn.Module):
    """ Predicts chunk boundaries based on cosine similarity of adjacent vectors. """
    def __init__(self, hid_size):
        super().__init__()
        # Projections for query (q) and key (k)
        self.w_q = nn.Linear(hid_size, hid_size, bias=False)
        self.w_k = nn.Linear(hid_size, hid_size, bias=False)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hid_size)
        Returns:
            p (torch.Tensor): Boundary probabilities of shape (batch, seq_len)
            b (torch.Tensor): Discretized boundary indicators of shape (batch, seq_len)
        """
        batch_size = x.shape[0]

        # --- FIX: Manual and more stable cosine similarity calculation ---
        
        # To compare q_t and k_{t-1}, we align their sequences.
        q_aligned = self.w_q(x[:, 1:, :])      # Shape: (batch, seq_len - 1, dim)
        k_aligned = self.w_k(x[:, :-1, :])     # Shape: (batch, seq_len - 1, dim)

        # Manually compute cosine similarity for maximum numerical stability
        dot_product = torch.sum(q_aligned * k_aligned, dim=-1)
        
        q_norm = torch.linalg.vector_norm(q_aligned, dim=-1)
        k_norm = torch.linalg.vector_norm(k_aligned, dim=-1)
        
        # Clamp the denominator to a small positive value to prevent division by zero
        eps = torch.finfo(q_norm.dtype).eps
        norm_product = (q_norm * k_norm).clamp(min=eps)
        
        similarity = dot_product / norm_product
        
        # --- END OF FIX ---
        
        # Boundary probability p_t = 0.5 * (1 - similarity)
        p_values = 0.5 * (1 - similarity)
        
        # The first token is always a boundary by definition 
        first_p = torch.ones(batch_size, 1, device=x.device, dtype=p_values.dtype)
        p = torch.cat([first_p, p_values], dim=1) # Final shape: (batch, seq_len)

        # Get discrete boundaries for downsampling
        b = StraightThroughEstimator.apply(p)
        return p, b


class Downsampler(nn.Module):
    """ Compresses the sequence by selecting vectors at boundary locations. """
    def forward(self, x, boundaries):
        """
        Args:
            x (torch.Tensor): Sequence to be downsampled (batch, seq_len, dim)
            boundaries (torch.Tensor): Boundary indicators (batch, seq_len), 1s and 0s.
        Returns:
            torch.Tensor: Compressed and padded sequence (batch, max_chunks, dim).
            torch.Tensor: Length of each compressed sequence in the batch (batch,).
        """
        mask = boundaries.bool()
        B, L, D = x.shape
        
        # Get chunk lengths
        chunk_lengths = torch.sum(mask, dim=1)
        max_chunks = torch.max(chunk_lengths) if chunk_lengths.numel() > 0 else 0
        
        # Create output tensor
        padded_chunks = torch.zeros(B, max_chunks, D, device=x.device, dtype=x.dtype)
        
        # Create a mask for padding
        padding_mask = torch.arange(max_chunks, device=x.device)[None, :] < chunk_lengths[:, None]
        
        # Select the elements from x that correspond to boundaries
        # and place them into padded_chunks
        if torch.any(mask):
            padded_chunks[padding_mask] = x[mask]
        
        return padded_chunks, chunk_lengths


class CrossAttentionUpsampler(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_norm = RMSNorm(hidden_size=self.d_head)
        self.k_norm = RMSNorm(hidden_size=self.d_head)\

    def forward(self, query, key_value, key_padding_mask=None):
        B, L_q, D = query.shape
        B, L_kv, D_kv = key_value.shape

        q = self.q_proj(query)
        k, v = self.kv_proj(key_value).chunk(2, dim=-1)

        q = q.view(B, L_q, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, L_kv, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, L_kv, self.n_head, self.d_head).transpose(1, 2)


        q = self.q_norm(q)
        k = self.k_norm(k)

        # Convert padding mask to attention mask
        attn_mask = None
        if key_padding_mask is not None:
            # Expand mask to match attention dimensions
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(q.dtype)  # (B, 1, 1, L_kv)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,  # ← Now using the padding mask!
            dropout_p=0.0, 
            is_causal=False, 
            scale=None
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_proj(attn_output)


@dataclass
class HNetTransformerModelOutput(BaseModelOutputWithNoAttention):
    ratio_loss: Optional[torch.FloatTensor] = None
    motif_loss: Optional[torch.FloatTensor] = None


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create HNet block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = Block
    # mambav2 compatibility
    if "mlp_cls" in inspect.signature(block_cls.__init__).parameters:
        block = block_cls(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        block = block_cls(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out


class HNetEmbeddings(nn.Module):
    def __init__(
            self,
            config: HNetConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)

class HNetEmbeddingsSTFT(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config

        d_emb = int(config.d_model * 0.7)
        d_stft = config.d_model - d_emb
        # 1. Standard learnable embeddings for nucleotides
        self.word_embeddings = nn.Embedding(
            config.vocab_size, d_emb, **factory_kwargs
        )

        # 2. STFT parameters
        self.n_fft = 256
        self.hop_length = 64
        # Register the window as a buffer so it moves to the correct device with the model
        self.register_buffer('window', torch.hann_window(self.n_fft, **factory_kwargs))

        # 3. A linear layer to project the high-dimensional STFT output
        #    to a manageable size for the positional encoding.
        #    Input dim = vocab_size * (n_fft // 2 + 1 frequency bins)
        stft_input_dim = config.vocab_size * (self.n_fft // 2 + 1)
        self.stft_projection = nn.Linear(
            stft_input_dim, d_stft, **factory_kwargs
        )

    def forward(self, input_ids):
        # import pdb; pdb.set_trace()
        """
        input_ids: (batch_size, seq_len)
        """
        # --- Step 1: Get standard learnable embeddings ---
        # (batch, seq_len) -> (batch, seq_len, d_model)
        word_embeds = self.word_embeddings(input_ids)

        # --- Step 2: Calculate STFT on the raw token IDs ---
        # a) Convert integer token IDs to a one-hot signal
        # (batch, seq_len) -> (batch, seq_len, vocab_size)
        signal = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        
        # b) Permute for torch.stft, which expects (..., time)
        # (batch, seq_len, vocab) -> (batch, vocab, seq_len)
        signal = signal.permute(0, 2, 1)

        # c) Compute STFT on each channel (A,C,G,T) for the whole batch
        # torch.stft takes (batch, time), so we reshape
        batch_size, vocab_size, seq_len = signal.shape
        signal_flat = signal.reshape(batch_size * vocab_size, seq_len)
        
        stft_complex = torch.stft(
            signal_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        ) # Shape: (batch * vocab, freq_bins, time_frames)

        # d) Use the magnitude of the complex output (power spectrum)
        stft_mag = stft_complex.abs()

        # e) Reshape back to separate batch and vocab dimensions
        num_freq_bins = stft_mag.shape[1]
        num_frames = stft_mag.shape[2]
        stft_mag = stft_mag.reshape(batch_size, vocab_size, num_freq_bins, num_frames)

        # --- Step 3: Upsample STFT to match original sequence length ---
        # The STFT downsamples the time dimension. We bring it back.
        # Permute to (batch, channels, time) for interpolation
        # Channels here are vocab_size * freq_bins
        stft_to_upsample = stft_mag.permute(0, 2, 1, 3).flatten(1, 2)
        stft_upsampled = F.interpolate(
            stft_to_upsample, size=seq_len, mode='linear', align_corners=False
        )

        # Reshape to (batch, seq_len, vocab_size * freq_bins) for projection
        stft_flat_features = stft_upsampled.permute(0, 2, 1)

        # --- Step 4: Project STFT features to the desired dimension ---
        # (batch, seq_len, vocab*bins) -> (batch, seq_len, stft_d_model)
        stft_encoding = self.stft_projection(stft_flat_features)

        # --- Step 5: Concatenate to create the final, augmented embedding ---
        # (batch, seq_len, d_model + stft_d_model)
        final_embeddings = torch.cat([word_embeds, stft_encoding], dim=-1)
        return final_embeddings

class RotarySelfAttention(nn.Module):
    def __init__(self, config: HNetConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.transformer_n_head
        self.d_head = self.d_model // self.n_head
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"

        self.rotary_emb = rotary_emb
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # ADD THESE "SAFETY VALVE" NORMALIZATION LAYERS
        self.q_norm = RMSNorm(hidden_size=self.d_head)
        self.k_norm = RMSNorm(hidden_size=self.d_head)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = q.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.d_head).transpose(1, 2)

        if q.isnan().any() or k.isnan().any():
            print("Q or K is nan")
            import pdb; pdb.set_trace()
        q, k = self.rotary_emb(q, k)
        if q.isnan().any() or k.isnan().any():
            print("Q or K is nan after rotary")
            import pdb; pdb.set_trace()

        if attention_mask is not None and attention_mask.all(dim=-1).any():
            print("!!! All-masked scenario detected. One or more batch items have no keys to attend to. !!!")
            import pdb; pdb.set_trace()  
        
        # NORMALIZE Q AND K RIGHT BEFORE THE ATTENTION FUNCTION
        # This prevents their dot product from exploding.
        q = self.q_norm(q)
        k = self.k_norm(k)

        if q.isnan().any() or k.isnan().any():
            print("Q or K is nan after norm")
            import pdb; pdb.set_trace()

        # This function will now receive stable inputs.
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.to(q.dtype), is_causal=False, scale=None)

        # Zero out the padded positions
        # 1. Transpose the attention output to (B, L, n_heads, D_head)
        padding_mask = attention_mask.transpose(2, 3)

        # Apply the mask directly. It will broadcast over the n_heads and d_head dimensions.
        attn_output = attn_output.masked_fill(padding_mask, 0.0)

        # --- END: DEFINITIVE "ZEROING OUT" FIX ---

        # Reshape and project as before.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: HNetConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = RotarySelfAttention(config, rotary_emb)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        mlp_hidden_size = config.d_model * config.transformer_mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, config.d_model),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class HNetMixerModel(nn.Module):
    def __init__(
            self,
            config: HNetConfig,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.config = config
        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32

        self.tokenizer_type = config.tokenizer_type
        if config.tokenizer_type == "default":
            self.embeddings = HNetEmbeddings(config, **factory_kwargs)
        elif config.tokenizer_type == "stft":
            self.embeddings = HNetEmbeddingsSTFT(config, **factory_kwargs)
        else:
            raise ValueError(f"Invalid tokenizer type: {config.tokenizer_type}")

        # HNet components
        self.routing_module = RoutingModule(config.d_model)
        self.downsampler = Downsampler()
        self.upsampler = CrossAttentionUpsampler(config.d_model, config.transformer_n_head)
        self.residual_proj = nn.Linear(config.d_model, config.d_model)
        self.target_ratio = getattr(config, "target_ratio", 0.3)
        self.motif_ratio = getattr(config, "motif_ratio", 0.0)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            create_block(
                config.d_model,
                ssm_cfg=config.ssm_cfg,
                norm_epsilon=config.norm_epsilon,
                rms_norm=config.rms_norm,
                residual_in_fp32=config.residual_in_fp32,
                fused_add_norm=config.fused_add_norm,
                layer_idx=i,
                bidirectional=config.bidirectional,
                bidirectional_strategy=config.bidirectional_strategy,
                bidirectional_weight_tie=config.bidirectional_weight_tie,
                **factory_kwargs,
            )
            for i in range(config.n_enc_layer)
        ])

        # Main model - Transformer
        d_head = config.d_model // config.transformer_n_head
        rotary_emb = RotaryEmbedding(dim=d_head)
        self.main_model = nn.ModuleList(
            [
                TransformerBlock(config, rotary_emb)
                for _ in range(config.n_main_layer)
            ]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([
            create_block(
                config.d_model,
                ssm_cfg=config.ssm_cfg,
                norm_epsilon=config.norm_epsilon,
                rms_norm=config.rms_norm,
                residual_in_fp32=config.residual_in_fp32,
                fused_add_norm=config.fused_add_norm,
                layer_idx=config.n_enc_layer + config.n_main_layer + i,
                bidirectional=config.bidirectional,
                bidirectional_strategy=config.bidirectional_strategy,
                bidirectional_weight_tie=config.bidirectional_weight_tie,
                **factory_kwargs,
            )
            for i in range(config.n_dec_layer)
        ])

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f

    def calculate_special_token_boundaries(self, input_ids):
        """Calculate boundaries for special tokens."""
        # boundaries should be length n+1 where n is input_ids length
        boundaries = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long, device=input_ids.device)
        
        special_tokens = (input_ids <= 6)
        
        # For each special token at position i, set boundaries[i] and boundaries[i+1]
        boundaries[:, 1:][special_tokens] = 1  # Boundary before each special token
        boundaries[:, :-1][special_tokens] = 1  # Boundary after each special token
        
        return boundaries[:, :-1]
        
    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False, boundaries=None):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        
        # HNet-style forward
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # 1. Encoder
        residual_after_encoder = residual
        for layer in self.encoder_layers:
            hidden_states, residual_after_encoder = layer(hidden_states, residual_after_encoder, inference_params=None)
        x_hat = hidden_states

        # 2. Chunking
        p_original, b_original = self.routing_module(x_hat)
        special_token_boundaries = self.calculate_special_token_boundaries(input_ids)

        b_original = (b_original.bool() | special_token_boundaries.bool()).float()
        
        x_s, chunk_lengths = self.downsampler(x_hat, b_original)

        # 3. Main Network (Transformer)
        main_hidden_states = x_s
        max_chunks = x_s.shape[1]
        attention_mask = torch.arange(max_chunks, device=x_s.device)[None, :] >= chunk_lengths[:, None]
        attention_mask = attention_mask[:, None, None, :]  # Shape: (batch, 1, 1, seq_len) for broadcasting
        for layer in self.main_model:
            main_hidden_states = layer(main_hidden_states, attention_mask=attention_mask)
        if main_hidden_states.isnan().any():
            print("Main hidden states is nan")
            import pdb; pdb.set_trace()
        z_hat_s = main_hidden_states
        all_hidden_states.append((z_hat_s, attention_mask))

        # 4. Dechunking with Cross-Attention
        max_chunks = z_hat_s.shape[1]
        key_padding_mask = torch.arange(max_chunks, device=z_hat_s.device)[None, :] >= chunk_lengths[:, None]
        z_dechunked = self.upsampler(query=x_hat, key_value=z_hat_s, key_padding_mask=key_padding_mask)

        # 5. Decoder
        hidden_states = z_dechunked + self.residual_proj(x_hat)
        residual = None # Residual is handled inside the Mamba blocks
        for layer in self.decoder_layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)

        # Ratio Loss
        f = b_original.sum(dim=1) / b_original.shape[1]
        G = p_original.mean(dim=1)
        N = 1.0 / self.target_ratio
        ratio_loss = (N / (N - 1)) * ((N - 1) * f * G + (1 - f) * (1 - G)) if N > 1 else torch.zeros_like(f)
        ratio_loss = ratio_loss.mean()

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # if ratio_loss.isnan().any():
        #     print("Ratio loss is nan")
        #     import pdb; pdb.set_trace()
        # if hidden_states.isnan().any():
        #     print("Hidden states is nan")
        #     import pdb; pdb.set_trace(1)
        return hidden_states, all_hidden_states, ratio_loss


def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    if ignore_index is None:
        ignore_index = -100
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class HNetPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for HNet backbone."""
    config_class = HNetConfig
    base_model_prefix = "caduceus_hnet_transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper", "TransformerBlock"]

    def _init_weights(
            self,
            module,
            initializer_range=0.02,  # Now only used for embedding layer.
            **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight", "mlp.2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class HNetTransformer(HNetPreTrainedModel):
    """HNet model that can be instantiated using HF patterns."""
    def __init__(self, config: HNetConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = HNetMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            boundaries: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple, HNetTransformerModelOutput]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states, ratio_loss = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            boundaries=boundaries
        )
        if return_dict:
            return HNetTransformerModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
                ratio_loss=ratio_loss,
            )
        
        output = (hidden_states,)
        if output_hidden_states:
            output += (all_hidden_states,)
        output += (ratio_loss,)
        return output


class HNetTransformerForMaskedLM(HNetPreTrainedModel):
    """HF-compatible HNet model for masked language modeling."""

    def __init__(self, config: HNetConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = HNetTransformer(config, **factory_kwargs, **kwargs)

        self.lm_head = nn.Linear(
            config.d_model,
            self.config.vocab_size,
            bias=False,
            **factory_kwargs
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """
        Override the default tie_weights behavior. Only tie weights if
        we are NOT using the composite STFT embedding.
        """
        # Check the tokenizer type from the backbone's config
        if self.caduceus.backbone.tokenizer_type == "stft":
            print("Weight tying is disabled for STFT embeddings.")
            return # Do nothing
        else:
            # If not using STFT, proceed with normal weight tying
            super().tie_weights()

    def get_decoder(self):
        return self.caduceus

    def set_decoder(self, decoder):
        self.caduceus = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boundaries: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            boundaries=boundaries
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights, ignore_index=self.config.pad_token_id)
            else:
                loss = cross_entropy(logits, labels, ignore_index=self.config.pad_token_id)
            
            ratio_loss = outputs.ratio_loss if return_dict else outputs[-1]
            if ratio_loss is not None:
                loss += ratio_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class HNetTransformerForSequenceClassification(HNetPreTrainedModel):
    def __init__(
            self,
            config: HNetConfig,
            pooling_strategy: str = "mean",
            conjoin_train: bool = False,
            conjoin_eval: bool = False,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = HNetTransformer(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        self.post_init()
        self.init_scorer()

    def init_scorer(self, initializer_range=0.02):
        initializer_range = self.config.initializer_cfg.get("initializer_range", initializer_range) \
            if self.config.initializer_cfg is not None else initializer_range
        self.score.weight.data.normal_(std=initializer_range)

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.caduceus.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.caduceus(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs.last_hidden_state

        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
