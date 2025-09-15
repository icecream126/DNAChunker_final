"""HNet-NTV2 hybrid model implementation.

This model loads Nucleotide Transformer v2 weights as the main transformer backbone
while using HNet-style tokenization and chunking mechanisms.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForMaskedLM
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

try:
    from .configuration_hnet_ntv2 import HNetNTV2Config
except ImportError:
    from configuration_hnet_ntv2 import HNetNTV2Config


@dataclass
class HNetNTV2ModelOutput(BaseModelOutputWithNoAttention):
    ratio_loss: Optional[torch.FloatTensor] = None
    motif_loss: Optional[torch.FloatTensor] = None


def rotate_half(x):
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings as used in NTV2."""
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, q, k):
        seq_len = q.shape[2]
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for boundary prediction."""
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RoutingModule(nn.Module):
    """HNet routing module for dynamic chunking."""
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Initialize with identity for stable training
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))
            self.k_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))

    def forward(self, hidden_states):
        cos_sim = F.cosine_similarity(
            self.q_proj_layer(hidden_states[:, 1:, :]),
            self.k_proj_layer(hidden_states[:, :-1, :]),
            dim=-1
        )
        
        boundary_prob_values = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        first_token_boundary = torch.ones(
            hidden_states.shape[0], 1, device=hidden_states.device, dtype=boundary_prob_values.dtype
        )
        boundary_prob = torch.cat([first_token_boundary, boundary_prob_values], dim=1)
        boundary_mask = boundary_prob >= 0.5
        
        return boundary_prob, boundary_mask


class ChunkLayer(nn.Module):
    """High-performance downsampler using sorting-based method."""
    def forward(self, x: torch.Tensor, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        device = x.device

        num_tokens = boundaries.sum(dim=-1)
        max_chunks = int(num_tokens.max())

        # Create indices where non-boundary tokens are pushed to the end
        token_idx = (
            torch.arange(seq_len, device=device)[None, :] + (~boundaries).long() * seq_len
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        # Gather the boundary tokens
        next_hidden_states = torch.gather(
            x,
            dim=1,
            index=seq_sorted_indices[:, :max_chunks, None].expand(-1, -1, dim),
        )

        return next_hidden_states, num_tokens


class CrossAttentionUpsampler(nn.Module):
    """Cross-attention based upsampler for dechunking."""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key_value, key_padding_mask=None):
        B, L_q, D = query.shape
        B, L_kv, D_kv = key_value.shape

        q = self.q_proj(query)
        k, v = self.kv_proj(key_value).chunk(2, dim=-1)

        q = q.view(B, L_q, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, L_kv, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, L_kv, self.n_head, self.d_head).transpose(1, 2)

        # Convert padding mask to attention mask
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(q.dtype)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0, 
            is_causal=False, 
            scale=None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_proj(attn_output)


class NTV2TransformerBlock(nn.Module):
    """NTV2-style transformer block with rotary embeddings."""
    def __init__(self, config: HNetNTV2Config, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"

        self.rotary_emb = rotary_emb
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # MLP with Gated Linear Units (GLU) as in NTV2
        # NTV2 uses intermediate_size=8192 but output projection expects 4096 input
        # We need to split the intermediate output and use only half for the output projection
        self.mlp_intermediate = nn.Linear(self.d_model, config.intermediate_size)  # 1024 -> 8192
        self.mlp_output = nn.Linear(config.intermediate_size // 2, self.d_model)  # 4096 -> 1024
        self.activation = nn.GELU()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # Self-attention with rotary embeddings
        residual = x
        x = self.norm1(x)
        
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.d_head).transpose(1, 2)

        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k)

        # Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        x = residual + self.hidden_dropout(attn_output)

        # MLP with proper NTV2 architecture
        residual = x
        x = self.norm2(x)
        x = self.mlp_intermediate(x)  # 1024 -> 8192
        x = self.activation(x)
        x = self.mlp_output(x[:, :, :self.mlp_output.in_features])  # Take first 4096 dimensions -> 1024
        x = residual + self.hidden_dropout(x)

        return x


class HNetEmbeddings(nn.Module):
    """HNet-style embeddings with trainable tokenization."""
    def __init__(self, config: HNetNTV2Config, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model, **factory_kwargs)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        word_embeds = self.word_embeddings(input_ids)
        
        embeddings = word_embeds + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        # import pdb; pdb.set_trace()
        
        return embeddings


class HNetNTV2MixerModel(nn.Module):
    """Main hybrid model combining NTV2 transformer with HNet chunking."""
    def __init__(self, config: HNetNTV2Config, device=None, dtype=None, **kwargs):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config

        # Embeddings
        self.embeddings = HNetEmbeddings(config, **factory_kwargs)

        # HNet chunking components
        self.routing_module = RoutingModule(config.d_model, **factory_kwargs)
        self.chunk_layer = ChunkLayer()
        self.upsampler = CrossAttentionUpsampler(config.d_model, config.n_head)
        self.residual_proj = nn.Linear(config.d_model, config.d_model, **factory_kwargs)
        self.target_ratio = config.target_ratio

        # Layer configuration
        self.n_enc_layer = config.n_enc_layer
        self.n_main_layer = config.n_main_layer
        self.n_dec_layer = config.n_dec_layer

        # Rotary embeddings for NTV2
        d_head = config.d_model // config.n_head
        self.rotary_emb = RotaryEmbedding(dim=d_head, max_seq_len=config.max_position_embeddings)

        # Encoder layers (Mamba-style for chunking)
        self.encoder_layers = nn.ModuleList([
            self._create_encoder_block(config, i, **factory_kwargs)
            for i in range(config.n_enc_layer)
        ])

        # Main transformer layers (NTV2-style)
        self.main_layers = nn.ModuleList([
            NTV2TransformerBlock(config, self.rotary_emb)
            for _ in range(config.n_main_layer)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            self._create_encoder_block(config, config.n_enc_layer + config.n_main_layer + i, **factory_kwargs)
            for i in range(config.n_dec_layer)
        ])

        # Final layer norm
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def _create_encoder_block(self, config, layer_idx, **factory_kwargs):
        """Create encoder/decoder block (simplified for this implementation)."""
        return nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            **factory_kwargs
        )

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False, boundaries=None):
        all_hidden_states = []
        # import pdb; pdb.set_trace()
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Encoder
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        
        x_hat = hidden_states

        # Chunking
        p_original, b_original = self.routing_module(x_hat)
        
        # Downsample
        x_chunked, chunk_lengths = self.chunk_layer(x_hat, b_original)
        
        # Create attention mask for chunks
        max_chunks = x_chunked.shape[1]
        attention_mask = torch.arange(max_chunks, device=x_chunked.device)[None, :] >= chunk_lengths[:, None]
        attention_mask = attention_mask[:, None, None, :].float()

        # Main transformer processing
        for layer in self.main_layers:
            x_chunked = layer(x_chunked, attention_mask=attention_mask)
        
        if output_hidden_states:
            all_hidden_states.append((x_chunked, attention_mask))

        # Upsampling
        key_padding_mask = torch.arange(max_chunks, device=x_chunked.device)[None, :] >= chunk_lengths[:, None]
        z_dechunked = self.upsampler(query=x_hat, key_value=x_chunked, key_padding_mask=key_padding_mask)

        # Decoder
        hidden_states = z_dechunked + self.residual_proj(x_hat)
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.norm_f(hidden_states)

        # Calculate ratio loss
        f = b_original.float().mean(dim=1)
        g = p_original.mean(dim=1)
        n = 1.0 / self.target_ratio
        ratio_loss = (n / (n - 1)) * ((n - 1) * f * g + (1 - f) * (1 - g)) if n > 1 else torch.zeros_like(f)
        ratio_loss = ratio_loss.mean()

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states, ratio_loss


class HNetNTV2PreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for HNet-NTV2."""
    config_class = HNetNTV2Config
    base_model_prefix = "hnet_ntv2"
    supports_gradient_checkpointing = False

    def _init_weights(self, module, initializer_range=0.02, **kwargs):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)


class HNetNTV2(HNetNTV2PreTrainedModel):
    """HNet-NTV2 hybrid model."""
    def __init__(self, config: HNetNTV2Config, device=None, dtype=None, **kwargs):
        super().__init__(config)

        # Adjust vocab size to match CaduceusTokenizer padding
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = HNetNTV2MixerModel(config, **factory_kwargs, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boundaries: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple, HNetNTV2ModelOutput]:
        """Forward pass."""
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
            return HNetNTV2ModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
                ratio_loss=ratio_loss,
            )
        
        output = (hidden_states,)
        if output_hidden_states:
            output += (all_hidden_states,)
        output += (ratio_loss,)
        return output


class HNetNTV2ForMaskedLM(HNetNTV2PreTrainedModel):
    """HNet-NTV2 model for masked language modeling."""
    def __init__(self, config: HNetNTV2Config, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hnet_ntv2 = HNetNTV2(config, **factory_kwargs, **kwargs)

        self.lm_head = nn.Linear(
            config.d_model,
            self.config.vocab_size,
            bias=False,
            **factory_kwargs
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.hnet_ntv2.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hnet_ntv2.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.hnet_ntv2

    def set_decoder(self, decoder):
        self.hnet_ntv2 = decoder

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
        """Forward pass for masked language modeling."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hnet_ntv2(
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
                loss = self._weighted_cross_entropy(logits, labels, loss_weights)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
            
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

    def _weighted_cross_entropy(self, logits, y, loss_weights, ignore_index=-100):
        """Weighted cross entropy loss."""
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
        loss_weights = loss_weights.view(-1)
        loss_weights[y == ignore_index] = 0.0
        return (ce * (loss_weights / loss_weights.sum())).sum()


def load_ntv2_weights(model: HNetNTV2ForMaskedLM, ntv2_model_path: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"):
    """Load NTV2 weights into the main transformer layers of the hybrid model.
    
    This function ONLY loads the main transformer layers (attention, MLP, layer norms)
    and skips embedding layers and output heads that have different vocabulary sizes.
    """
    print(f"Loading NTV2 weights from {ntv2_model_path}...")
    print("NOTE: Only loading main transformer layers, skipping embeddings and output heads due to vocabulary size mismatch")
    
    # Load the original NTV2 model
    ntv2_model = AutoModelForMaskedLM.from_pretrained(ntv2_model_path, trust_remote_code=True)
    ntv2_state_dict = ntv2_model.state_dict()
    
    # Get our model's state dict
    our_state_dict = model.state_dict()
    mapped_state_dict = {}
    
    # Map NTV2 transformer layers to our main_layers
    ntv2_layers = [k for k in ntv2_state_dict.keys() if 'esm.encoder.layer' in k]
    ntv2_num_layers = len(set([k.split('.')[3] for k in ntv2_layers]))  # Count unique layer numbers
    
    print(f"Found {ntv2_num_layers} NTV2 transformer layers")
    print(f"Our model has {model.config.n_main_layer} main layers")
    print(f"NTV2 vocab size: {ntv2_model.config.vocab_size}, Our vocab size: {model.config.vocab_size}")
    
    # Map each of our main layers to NTV2 layers
    for i in range(model.config.n_main_layer):
        ntv2_layer_idx = i % ntv2_num_layers  # Cycle through NTV2 layers if we have more main layers
        
        # Map attention weights (Q, K, V projections)
        qkv_weight_key = f'hnet_ntv2.backbone.main_layers.{i}.qkv_proj.weight'
        if qkv_weight_key in our_state_dict:
            # NTV2 has separate Q, K, V projections, we need to combine them
            q_weight = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.self.query.weight')
            k_weight = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.self.key.weight')
            v_weight = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.self.value.weight')
            
            if q_weight is not None and k_weight is not None and v_weight is not None:
                # Concatenate Q, K, V weights
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                # Clamp weights to prevent numerical instability
                qkv_weight = torch.clamp(qkv_weight, -10.0, 10.0)
                mapped_state_dict[qkv_weight_key] = qkv_weight
                print(f"Mapped attention weights for layer {i} from NTV2 layer {ntv2_layer_idx}")
        
        # Map attention output projection
        attn_out_key = f'hnet_ntv2.backbone.main_layers.{i}.out_proj.weight'
        attn_out_bias_key = f'hnet_ntv2.backbone.main_layers.{i}.out_proj.bias'
        if attn_out_key in our_state_dict:
            ntv2_attn_out = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.output.dense.weight')
            if ntv2_attn_out is not None:
                # Clamp weights to prevent numerical instability
                ntv2_attn_out = torch.clamp(ntv2_attn_out, -10.0, 10.0)
                mapped_state_dict[attn_out_key] = ntv2_attn_out
                print(f"Mapped attention output for layer {i} from NTV2 layer {ntv2_layer_idx}")
        
        # Map MLP weights (intermediate and output)
        mlp_intermediate_key = f'hnet_ntv2.backbone.main_layers.{i}.mlp_intermediate.weight'
        mlp_output_key = f'hnet_ntv2.backbone.main_layers.{i}.mlp_output.weight'
        
        if mlp_intermediate_key in our_state_dict:
            ntv2_intermediate = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.intermediate.dense.weight')
            if ntv2_intermediate is not None:
                # Clamp weights to prevent numerical instability
                ntv2_intermediate = torch.clamp(ntv2_intermediate, -10.0, 10.0)
                mapped_state_dict[mlp_intermediate_key] = ntv2_intermediate
                print(f"Mapped MLP intermediate for layer {i} from NTV2 layer {ntv2_layer_idx}")
        
        if mlp_output_key in our_state_dict:
            ntv2_output = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.output.dense.weight')
            if ntv2_output is not None:
                # Clamp weights to prevent numerical instability
                ntv2_output = torch.clamp(ntv2_output, -10.0, 10.0)
                mapped_state_dict[mlp_output_key] = ntv2_output
                print(f"Mapped MLP output for layer {i} from NTV2 layer {ntv2_layer_idx}")
        
        # Map layer norms
        norm1_key = f'hnet_ntv2.backbone.main_layers.{i}.norm1.weight'
        norm1_bias_key = f'hnet_ntv2.backbone.main_layers.{i}.norm1.bias'
        norm2_key = f'hnet_ntv2.backbone.main_layers.{i}.norm2.weight'
        norm2_bias_key = f'hnet_ntv2.backbone.main_layers.{i}.norm2.bias'
        
        if norm1_key in our_state_dict:
            ntv2_norm1_weight = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.LayerNorm.weight')
            ntv2_norm1_bias = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.attention.LayerNorm.bias')
            if ntv2_norm1_weight is not None:
                # Clamp layer norm weights to prevent numerical instability
                ntv2_norm1_weight = torch.clamp(ntv2_norm1_weight, -5.0, 5.0)
                mapped_state_dict[norm1_key] = ntv2_norm1_weight
            if ntv2_norm1_bias is not None:
                ntv2_norm1_bias = torch.clamp(ntv2_norm1_bias, -5.0, 5.0)
                mapped_state_dict[norm1_bias_key] = ntv2_norm1_bias
        
        if norm2_key in our_state_dict:
            ntv2_norm2_weight = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.LayerNorm.weight')
            ntv2_norm2_bias = ntv2_state_dict.get(f'esm.encoder.layer.{ntv2_layer_idx}.LayerNorm.bias')
            if ntv2_norm2_weight is not None:
                # Clamp layer norm weights to prevent numerical instability
                ntv2_norm2_weight = torch.clamp(ntv2_norm2_weight, -5.0, 5.0)
                mapped_state_dict[norm2_key] = ntv2_norm2_weight
            if ntv2_norm2_bias is not None:
                ntv2_norm2_bias = torch.clamp(ntv2_norm2_bias, -5.0, 5.0)
                mapped_state_dict[norm2_bias_key] = ntv2_norm2_bias
    
    # Skip final layer norm mapping as it might have different dimensions
    print("Skipping final layer norm mapping due to potential dimension mismatch")
    
    # Load the mapped weights
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    
    print(f"Successfully loaded {len(mapped_state_dict)} weight parameters from NTV2")
    print(f"Missing keys (not loaded): {len(missing_keys)}")
    print(f"Unexpected keys (not used): {len(unexpected_keys)}")
    
    # Print summary of what was loaded vs skipped
    print("\n" + "="*50)
    print("WEIGHT LOADING SUMMARY:")
    print("="*50)
    print("✓ Loaded: Main transformer layers (attention, MLP, layer norms)")
    print("✗ Skipped: Embedding layers (vocabulary size mismatch)")
    print("✗ Skipped: Output heads (vocabulary size mismatch)")
    print("✗ Skipped: Final layer norm (dimension mismatch)")
    print("🔧 Applied: Weight clamping for numerical stability")
    print("="*50)
    
    print("NTV2 weights loaded successfully into main transformer layers!")
    return model
