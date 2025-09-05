"""Caduceus model for Hugging Face.

"""

import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

# --- Mamba Imports ---
try:
    from mamba_ssm.modules.mamba_simple import Mamba, Block
except ImportError:
    # Handle mambav2 file structure
    from mamba_ssm.modules.block import Block
    from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

from .configuration_caduceus_hnet import CaduceusHNetConfig


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
    """ Compresses the sequence by selecting vectors at boundary locations. (Vectorized) """
    def forward(self, x: torch.Tensor, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Sequence to be downsampled (batch, seq_len, dim)
            boundaries (torch.Tensor): Boundary indicators (batch, seq_len), 1s and 0s.
        Returns:
            torch.Tensor: Compressed and padded sequence (batch, max_chunks, dim).
            torch.Tensor: Length of each compressed sequence in the batch (batch,).
        """
        # --- FIX: Detach boundaries from the graph for this non-differentiable op ---
        # chunk_lengths is only used for shape calculation and doesn't need a gradient.
        chunk_lengths = torch.sum(boundaries.detach(), dim=1, dtype=torch.long)
        
        # The rest of the function remains the same
        max_chunks = torch.max(chunk_lengths).item() if chunk_lengths.numel() > 0 else 0
        batch_size, _, dim = x.shape

        padded_chunks = torch.zeros(
            batch_size, max_chunks, dim, device=x.device, dtype=x.dtype
        )

        # The original 'boundaries' tensor (with its grad_fn) is used here, which is correct
        mask = boundaries.bool()
        batch_indices, seq_indices = torch.nonzero(mask, as_tuple=True)

        dest_indices = (torch.cumsum(boundaries, dim=1)[mask] - 1).long()

        padded_chunks[batch_indices, dest_indices] = x[batch_indices, seq_indices]

        return padded_chunks, chunk_lengths


"""Caduceus model for Hugging Face.

This script contains a memory-efficient, block-wise vectorized version of the
SmoothingModule to prevent OOM errors while maintaining high performance.
"""

import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union
import torchaudio.transforms as T


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

from .configuration_caduceus_hnet import CaduceusHNetConfig


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
    """ High-performance routing module based on cosine similarity. """
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # --- FIX: Add identity initialization for stable training start ---
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))
            self.k_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))
        # --- END FIX ---

    def forward(self, hidden_states):
        # We expect hidden_states to be (B, L, D)
        cos_sim = F.cosine_similarity(
            self.q_proj_layer(hidden_states[:, 1:, :]),
            self.k_proj_layer(hidden_states[:, :-1, :]),
            dim=-1
        )
        
        # Clamp for numerical stability
        boundary_prob_values = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force the first token of every sequence to be a boundary
        first_token_boundary = torch.ones(
            hidden_states.shape[0], 1, device=hidden_states.device, dtype=boundary_prob_values.dtype
        )
        boundary_prob = torch.cat([first_token_boundary, boundary_prob_values], dim=1)
        
        # Determine boundaries based on probability > 0.5
        boundary_mask = boundary_prob >= 0.5
        
        return boundary_prob, boundary_mask


class ChunkLayer(nn.Module):
    """ High-performance downsampler using a sorting-based method. """
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

        # Gather the boundary tokens, which are now at the beginning
        next_hidden_states = torch.gather(
            x,
            dim=1,
            index=seq_sorted_indices[:, :max_chunks, None].expand(-1, -1, dim),
        )

        return next_hidden_states, num_tokens

# --- FIX: Merged DeChunkLayer and Upsampler into a single module ---
class DeChunkAndUpsampleLayer(nn.Module):
    """ High-performance dechunking and upsampling using Mamba's scan kernel. """
    def __init__(self, d_model, dtype=torch.bfloat16, block_size=256, headdim=64):
        super().__init__()
        if mamba_chunk_scan_combined is None:
            raise ImportError("Mamba Triton kernels not found. Please install mamba_ssm with `pip install mamba-ssm --upgrade`.")
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0, "d_model must be divisible by headdim"
        self.nheads = d_model // self.headdim

    def forward(self, hidden_states_chunked, p_chunked, original_boundary_mask):
        original_dtype = hidden_states_chunked.dtype

        # Clamp probabilities for numerical stability
        p = torch.clamp(p_chunked.float(), min=1e-4, max=1.0 - 1e-4)

        # Re-formulate the EMA as an SSM to use the Mamba kernel
        dt = torch.log(1.0 / (1.0 - p)).to(self.dtype)
        x = (hidden_states_chunked / dt.unsqueeze(-1)).to(self.dtype)
        A = -torch.ones((self.nheads,), device=hidden_states_chunked.device, dtype=torch.float32)
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        # Apply Mamba's parallel scan
        out_chunked = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
        )
        out_chunked = rearrange(out_chunked, "b l h p -> b l (h p)")
        
        # Upsampling logic: broadcast chunked representations back to original sequence length
        chunk_indices = torch.cumsum(original_boundary_mask, dim=1).long() - 1
        chunk_indices = chunk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        
        z_tilde = torch.gather(out_chunked, 1, chunk_indices)
        
        return z_tilde.to(original_dtype)


class Downsampler(nn.Module):
    """ Compresses the sequence by selecting vectors at boundary locations. (Vectorized) """
    def forward(self, x: torch.Tensor, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Sequence to be downsampled (batch, seq_len, dim)
            boundaries (torch.Tensor): Boundary indicators (batch, seq_len), 1s and 0s.
        Returns:
            torch.Tensor: Compressed and padded sequence (batch, max_chunks, dim).
            torch.Tensor: Length of each compressed sequence in the batch (batch,).
        """
        chunk_lengths = torch.sum(boundaries.detach(), dim=1, dtype=torch.long)
        
        max_chunks = torch.max(chunk_lengths).item() if chunk_lengths.numel() > 0 else 0
        batch_size, _, dim = x.shape

        padded_chunks = torch.zeros(
            batch_size, max_chunks, dim, device=x.device, dtype=x.dtype
        )

        mask = boundaries.bool()
        batch_indices, seq_indices = torch.nonzero(mask, as_tuple=True)

        dest_indices = (torch.cumsum(boundaries, dim=1)[mask] - 1).long()

        padded_chunks[batch_indices, dest_indices] = x[batch_indices, seq_indices]

        return padded_chunks, chunk_lengths


class SmoothingModule(nn.Module):
    """ Applies a memory-efficient, block-wise vectorized EMA. """
    def __init__(self, block_size=1024):
        super().__init__()
        self.block_size = block_size

    def forward(self, z_hat, P, lengths):
        """
        Args:
            z_hat (torch.Tensor): Padded compressed representations (batch, max_chunks, dim).
            P (torch.Tensor): Padded boundary probabilities for the compressed sequence (batch, max_chunks).
            lengths (torch.Tensor): Length of each compressed sequence in the batch (batch,).
        Returns:
            torch.Tensor: Smoothed representations (batch, max_chunks, dim).
        """
        batch_size, max_chunks, dim = z_hat.shape
        
        # Initialize output tensor
        z_bar = torch.zeros_like(z_hat)

        # Pre-compute log cumulative products for stability and efficiency
        P_clamped = P.clamp(0.0, 1.0 - 1e-6)
        q = 1.0 - P_clamped
        log_q_padded = torch.log(torch.cat([torch.ones(batch_size, 1, device=P.device), q], dim=1))
        log_cumprod_q = torch.cumsum(log_q_padded, dim=1)
        log_cumprod_q_vals = log_cumprod_q[:, 1:]

        # Process in blocks to avoid creating the giant (T, T) matrix
        for t_start in range(0, max_chunks, self.block_size):
            t_end = min(t_start + self.block_size, max_chunks)
            current_block_size = t_end - t_start

            # Create indices for the current block
            t_indices = torch.arange(t_start, t_end, device=P.device).view(1, current_block_size, 1)
            k_indices = torch.arange(max_chunks, device=P.device).view(1, 1, max_chunks)

            # --- Build the matrix slice using broadcasting ---
            gathered_t = log_cumprod_q_vals[:, t_start:t_end].unsqueeze(2).expand(-1, -1, max_chunks)
            gathered_k = log_cumprod_q_vals.unsqueeze(1).expand(-1, current_block_size, -1)
            log_prod_terms = gathered_t - gathered_k
            
            log_M_slice = torch.log(P_clamped.clamp(min=1e-6)).unsqueeze(1) + log_prod_terms
            
            # Mask out the upper triangle of the slice
            mask = (t_indices < k_indices).to(z_hat.device)
            log_M_slice.masked_fill_(mask, -torch.finfo(log_M_slice.dtype).max)
            
            M_slice = torch.exp(log_M_slice)

            # Apply the transformation for the current block
            z_bar_slice = torch.bmm(M_slice, z_hat)
            z_bar[:, t_start:t_end, :] = z_bar_slice

        # Mask out padding based on original chunk lengths
        padding_mask = torch.arange(max_chunks, device=z_hat.device)[None, :] < lengths[:, None]
        z_bar = z_bar * padding_mask.unsqueeze(-1)

        return z_bar

class Upsampler(nn.Module):
    """ Decompresses the sequence back to its original resolution. """
    def forward(self, z_bar, p_original, b_original):
        """
        Args:
            z_bar (torch.Tensor): Smoothed, compressed representations (batch, max_chunks, dim).
            p_original (torch.Tensor): Original boundary probabilities (batch, seq_len).
            b_original (torch.Tensor): Original boundary indicators (batch, seq_len).
        Returns:
            torch.Tensor: Upsampled sequence of shape (batch, seq_len, dim).
        """
        c = torch.where(b_original.bool(), p_original, 1 - p_original)
        c_ste = StraightThroughEstimator.apply(c)
        
        chunk_indices = torch.cumsum(b_original, dim=1).long() - 1
        # We need to gather from z_bar using chunk_indices.
        # z_bar is (batch, max_chunks, dim), chunk_indices is (batch, seq_len)
        # We want output of (batch, seq_len, dim)
        
        # Add a dimension to chunk_indices for gather
        chunk_indices = chunk_indices.unsqueeze(-1).expand(-1, -1, z_bar.shape[-1])
        z_tilde = torch.gather(z_bar, 1, chunk_indices)
        
        return c_ste.unsqueeze(-1) * z_tilde


@dataclass
class CaduceusHNetModelOutput(BaseModelOutputWithNoAttention):
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
    """Create Caduceus block.

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


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusHNetConfig,
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

class CaduceusEmbeddingsSTFT(nn.Module):
    """
    Augments standard token embeddings with frequency-domain features derived
    from a Short-Time Fourier Transform (STFT) of the one-hot encoded sequence.
    
    This allows the model to learn positional and motif-based patterns from
    the local frequency content of the input sequence.
    """
    def __init__(
        self,
        config: CaduceusHNetConfig,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config

        # Split the model dimension for standard and STFT embeddings
        d_emb = int(config.d_model * 0.7)
        d_stft = config.d_model - d_emb

        # 1. Standard learnable embeddings for each token
        self.word_embeddings = nn.Embedding(
            config.vocab_size, d_emb, **factory_kwargs
        )

        # 2. STFT parameters and the transform layer from torchaudio
        self.n_fft = 256
        self.hop_length = 64
        
        self.stft_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=None,  # Crucial: Returns the complex spectrogram
            center=True,
        )

        # 3. A linear layer to project the high-dimensional STFT output
        #    to the desired embedding dimension.
        #    Input dim = vocab_size * (n_fft // 2 + 1 frequency bins)
        stft_input_dim = config.vocab_size * (self.n_fft // 2 + 1)
        self.stft_projection = nn.Linear(
            stft_input_dim, d_stft, **factory_kwargs
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.LongTensor): Tensor of token IDs of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: The final augmented embeddings of shape (batch_size, seq_len, d_model)
        """
        # --- Step 1: Get standard learnable embeddings ---
        # (batch, seq_len) -> (batch, seq_len, d_emb)
        word_embeds = self.word_embeddings(input_ids)

        # --- Step 2: Calculate STFT on the raw token IDs ---
        # a) Convert integer token IDs to a one-hot signal
        # (batch, seq_len) -> (batch, seq_len, vocab_size)
        signal = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        
        # b) Permute for torchaudio, which expects (..., time)
        # (batch, seq_len, vocab) -> (batch, vocab, seq_len)
        signal = signal.permute(0, 2, 1)

        # c) Compute STFT on each channel (A,C,G,T, etc.) for the whole batch
        # We flatten the batch and vocab dims to treat each channel as a separate signal
        batch_size, vocab_size, seq_len = signal.shape
        signal_flat = signal.reshape(batch_size * vocab_size, seq_len)
        
        # (batch * vocab, seq_len) -> (batch * vocab, freq_bins, time_frames)
        stft_complex = self.stft_transform(signal_flat)

        # d) Use the magnitude of the complex output (power spectrum)
        stft_mag = stft_complex.abs()

        # e) Reshape back to separate batch and vocab dimensions
        num_freq_bins = stft_mag.shape[1]
        num_frames = stft_mag.shape[2]
        stft_mag = stft_mag.reshape(batch_size, vocab_size, num_freq_bins, num_frames)

        # --- Step 3: Upsample STFT to match original sequence length ---
        # The STFT downsamples the time dimension. We bring it back via interpolation.
        # Permute to (batch, channels, time) where channels = vocab * freq_bins
        stft_to_upsample = stft_mag.permute(0, 2, 1, 3).flatten(1, 2)
        
        # Upsample the time dimension back to the original sequence length
        stft_upsampled = F.interpolate(
            stft_to_upsample,
            size=seq_len,
            mode='nearest', # 'nearest' is faster than 'linear'
            align_corners=None,
        )

        # Reshape to (batch, seq_len, features) for the projection layer
        stft_flat_features = stft_upsampled.permute(0, 2, 1)

        # --- Step 4: Project STFT features to the desired dimension ---
        # (batch, seq_len, vocab*bins) -> (batch, seq_len, d_stft)
        stft_encoding = self.stft_projection(stft_flat_features)

        # --- Step 5: Concatenate to create the final, augmented embedding ---
        # (batch, seq_len, d_emb + d_stft) -> (batch, seq_len, d_model)
        final_embeddings = torch.cat([word_embeds, stft_encoding], dim=-1)
        
        return final_embeddings

class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusHNetConfig,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.config = config
        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32

        self.n_enc_layer = getattr(config, "n_enc_layer", 1)
        self.n_main_layer = getattr(config, "n_main_layer", config.n_layer - 2)
        self.n_dec_layer = getattr(config, "n_dec_layer", 1)

        if self.n_enc_layer + self.n_main_layer + self.n_dec_layer != config.n_layer:
            raise ValueError("The sum of n_enc_layer, n_main_layer, and n_dec_layer must be equal to n_layer.")

        self.tokenizer_type = config.tokenizer_type
        if config.tokenizer_type == "default":
            self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)
        elif config.tokenizer_type == "stft":
            # Assuming CaduceusEmbeddingsSTFT is defined elsewhere as in your code
            self.embeddings = CaduceusEmbeddingsSTFT(config, **factory_kwargs)
        else:
            raise ValueError(f"Invalid tokenizer type: {config.tokenizer_type}")

        # High-performance H-Net components
        self.routing_module = RoutingModule(config.d_model, **factory_kwargs)
        self.chunk_layer = ChunkLayer()
        # --- FIX: Use the new combined de-chunk and upsample layer ---
        self.dechunk_and_upsample = DeChunkAndUpsampleLayer(config.d_model, dtype=dtype)
        # --- END FIX ---
        self.residual_proj = nn.Linear(config.d_model, config.d_model, **factory_kwargs)
        self.target_ratio = getattr(config, "target_ratio", 0.25)

        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
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
                    rcps=False,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
    
    def calculate_boundary_loss(self, p_original, boundaries):
        boundary_logit = torch.stack([1-p_original, p_original], dim=-1).clamp(min=1e-7, max=1-1e-7)
        boundary_target = boundaries.long()
        mask = boundaries == 1

        if mask.any():
            boundary_logits_flat = boundary_logit.view(-1, 2)[mask.view(-1)]
            boundary_target_flat = boundary_target.view(-1)[mask.view(-1)]
            boundary_loss = F.cross_entropy(boundary_logits_flat, boundary_target_flat, reduction="mean")
        else:
            boundary_loss = 0.0
        
        return boundary_loss
    
    def _pad_to_global_max(self, tensor: torch.Tensor, max_len: int, pad_value=0):
        """Pads the sequence dimension of a tensor to a global max length."""
        pad_len = max_len - tensor.shape[1]
        if pad_len > 0:
            return F.pad(tensor, (0, 0, 0, pad_len), value=pad_value)
        return tensor

    def get_mask_boundaries(self, input_ids):
        mask_index = 3
        mask = input_ids == mask_index

        # retrieve boundaries, where the boundary should be set surrounding the mask index
        boundaries = torch.zeros_like(input_ids)
        
        # Vectorized approach: find first and last mask positions for each batch
        batch_size, seq_len = input_ids.shape
        
        # Find first mask position for each batch (or seq_len if no mask found)
        first_mask_pos = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
        last_mask_pos = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
        
        # Get indices where mask tokens exist
        batch_indices, seq_indices = torch.where(mask)
        
        if len(batch_indices) > 0:
            # Group by batch and find min/max positions
            for batch_idx in range(batch_size):
                batch_mask_positions = seq_indices[batch_indices == batch_idx]
                if len(batch_mask_positions) > 0:
                    first_mask_pos[batch_idx] = batch_mask_positions.min()
                    last_mask_pos[batch_idx] = batch_mask_positions.max()
        
        # Set boundaries at start of mask regions
        valid_start_mask = first_mask_pos < seq_len
        boundaries[valid_start_mask, first_mask_pos[valid_start_mask]] = 1
        
        # Set boundaries after end of mask regions (if not at sequence end)
        valid_end_mask = (last_mask_pos >= 0) & (last_mask_pos < seq_len - 1)
        boundaries[valid_end_mask, last_mask_pos[valid_end_mask] + 1] = 1
        import pdb; pdb.set_trace()
        return boundaries
        
    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False, boundaries=None):
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        
        if self.n_main_layer < 1:
            raise ValueError("CaduceusHNet requires at least 1 main layer.")

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # 1. Encoder
        encoder_hidden_states = hidden_states
        encoder_residual = residual
        for layer in self.layers[:self.n_enc_layer]:
            encoder_hidden_states, encoder_residual = layer(encoder_hidden_states, encoder_residual, inference_params=None)
        x_hat = encoder_hidden_states
        
        # 2. Chunking
        p_original, b_original_dynamic = self.routing_module(x_hat)
        
        if boundaries is not None:
            boundary_loss = self.calculate_boundary_loss(p_original, boundaries)
        else:
            boundary_loss = 0.0
        
        mask_boundaries = self.get_mask_boundaries(input_ids)

        x_s_unpadded, num_tokens = self.chunk_layer(x_hat, b_original_dynamic)
        p_s_unpadded, _ = self.chunk_layer(p_original.unsqueeze(-1), b_original_dynamic)
        p_s_unpadded = p_s_unpadded.squeeze(-1)
        
        # DDP FIX for synchronizing lengths
        global_max_chunks = x_s_unpadded.shape[1]
        if dist.is_initialized() and self.training:
            local_max_chunks_tensor = torch.tensor([global_max_chunks], device=x_s_unpadded.device, dtype=torch.long)
            dist.all_reduce(local_max_chunks_tensor, op=dist.ReduceOp.MAX)
            global_max_chunks = local_max_chunks_tensor.item()

        x_s = self._pad_to_global_max(x_s_unpadded, global_max_chunks, pad_value=0)
        p_s = self._pad_to_global_max(p_s_unpadded, global_max_chunks, pad_value=0)
        
        pad_mask = torch.arange(global_max_chunks, device=x_s.device)[None, :] < num_tokens[:, None]
        
        # 3. Main Network
        main_hidden_states = x_s
        main_residual = None
        start_main = self.n_enc_layer
        end_main = self.n_enc_layer + self.n_main_layer
        for layer in self.layers[start_main:end_main]:
            main_hidden_states, main_residual = layer(main_hidden_states, main_residual, inference_params=None)
        
        main_hidden_states = main_hidden_states * pad_mask.unsqueeze(-1)
        z_hat_s = main_hidden_states

        # --- FIX: Call the single de-chunking and upsampling module ---
        # 4. Dechunking and Upsampling
        z_dechunked = self.dechunk_and_upsample(z_hat_s, p_s, b_original_dynamic)
        
        # 5. Decoder with Gated Residual Connection
        # The original probabilities 'p_original' act as a gate.
        gated_dechunked = z_dechunked * p_original.unsqueeze(-1)
        z_s = gated_dechunked + self.residual_proj(x_hat)
        # --- END FIX ---
        
        decoder_hidden_states = z_s
        decoder_residual = None
        start_decoder = self.n_enc_layer + self.n_main_layer
        for layer in self.layers[start_decoder:]:
            decoder_hidden_states, decoder_residual = layer(decoder_hidden_states, decoder_residual, inference_params=None)
        hidden_states = decoder_hidden_states
        residual = decoder_residual

        # Ratio Loss
        F = b_original_dynamic.float().mean(dim=1)
        G = p_original.mean(dim=1)
        N = 1.0 / self.target_ratio
        ratio_loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G)) if N > 1 else torch.zeros_like(F)
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
        return hidden_states, all_hidden_states, ratio_loss + boundary_loss

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
    if ignore_index is None:
        ignore_index = -100
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    config_class = CaduceusHNetConfig
    base_model_prefix = "caduceus_hnet"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

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
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class CaduceusHNet(CaduceusPreTrainedModel):
    """Caduceus model that can be instantiated using HF patterns."""
    def __init__(self, config: CaduceusHNetConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = CaduceusMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            boundaries: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple, CaduceusHNetModelOutput]:
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
            return CaduceusHNetModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
                ratio_loss=ratio_loss,
            )
        
        output = (hidden_states,)
        if output_hidden_states:
            output += (all_hidden_states,)
        output += (ratio_loss,)
        return output


class CaduceusHNetForMaskedLM(CaduceusPreTrainedModel):
    """HF-compatible Caduceus model for masked language modeling."""

    def __init__(self, config: CaduceusHNetConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = CaduceusHNet(config, **factory_kwargs, **kwargs)

        self.lm_head = nn.Linear(
            config.d_model,
            self.config.vocab_size,  # Use caduceus config as it might have been updated
            bias=False,
            **factory_kwargs
        )

        # Initialize weights and apply final processing
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
        if self.config.tokenizer_type == "stft":
            pass
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.caduceus

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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


class CaduceusHNetForSequenceClassification(CaduceusPreTrainedModel):
    def __init__(
            self,
            config: CaduceusHNetConfig,
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
        self.caduceus = CaduceusHNet(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
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
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.caduceus(
            input_ids,
            inputs_embeds=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs.last_hidden_state

        # Pool and get logits
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
