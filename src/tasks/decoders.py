"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)

class AttentionPooling(nn.Module):
    """
    A simple Attention Pooling module for aggregating sequence representations.
    
    This module uses a single learnable query vector to perform attention over
    the input sequence and produce a pooled representation. This is commonly
    used for tasks like text classification where a single vector is needed
    to represent the entire sequence.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, d_output=1):
        """
        Initializes the Attention Pooling module.

        Args:
            d_model (int): The dimension of the input and output representations.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout probability for the attention weights.
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        # The core of attention pooling: a single, learnable query token.
        # This token is a nn.Parameter that will be optimized during training
        # to learn how to best query the input sequence.
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Multi-Head Attention layer. The query comes from our learnable token,
        # while the key and value are the input sequence itself.
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=min(num_heads, d_model // 64),  # Prevents num_heads from exceeding d_model
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization for stabilizing training
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_output)

    def forward(self, x, attn_mask=None):
        """
        Performs attention pooling on the input sequence.

        Args:
            x (torch.Tensor): The input sequence tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): A boolean mask of shape (batch_size, seq_len)
                                                where `True` indicates padding locations.
                                                Used to prevent attention to padded tokens.

        Returns:
            torch.Tensor: The pooled representation of shape (batch_size, d_model).
        """
        batch_size, seq_len, d_model = x.shape
        
        # Expand the single learnable query token to match the batch size.
        # The query's shape becomes (batch_size, 1, d_model).
        query = self.query_token.expand(batch_size, -1, -1)

        if attn_mask is None:
            attn_mask = torch.all(x == 0, dim=-1).squeeze()
        
        # Apply the attention mechanism.
        # query: The learnable query for aggregation.
        # key: The input sequence x.
        # value: The input sequence x.
        # key_padding_mask: Ensures we don't attend to padded tokens.
        # print(f"ATTN shape: {attn_mask.shape}")
        # print(f"QUERY shape: {query.shape}")
        # print(f"X shape: {x.shape}")
        # print(f"VALUE shape: {x.shape}")
        # import pdb; pdb.set_trace()
        attn_output, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=attn_mask
        )
        
        # The attention output has shape (batch_size, 1, d_model).
        # We squeeze the sequence dimension and apply layer normalization.
        output = self.norm(attn_output.squeeze(1))
        output = self.linear(output)
        
        return output


class AttnPool(nn.Module):
    """Attention pooling layer for sequence aggregation that respects valid sequence lengths."""
    
    def __init__(self, d_model=None, num_heads=4, dropout=0.1, d_output=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        if d_model is not None:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
        else:
            self.attention = None
            self.norm = None

        self.linear = nn.Linear(d_model, d_output)
    
    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq_len, d_model)
        attn_mask: (batch, seq_len) - True for padding locations, False for valid tokens
        Returns: (batch, d_model) - pooled representation
        """
        if self.attention is None:
            # Initialize attention layer with input dimensions
            d_model = x.size(-1)
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=min(self.num_heads, d_model // 64),  # Ensure num_heads doesn't exceed d_model
                dropout=self.dropout,
                batch_first=True
            ).to(x.device)
            self.norm = nn.LayerNorm(d_model).to(x.device)
        
        batch_size, seq_len, d_model = x.shape
        
        # Create a learnable query for attention pooling
        if attn_mask is not None:
            # Use mean of valid tokens only (where attn_mask is False)
            valid_mask = attn_mask  # (batch, seq_len) - True for valid tokens
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
            
            # Mask out padding tokens and sum
            masked_x = x * valid_mask.unsqueeze(-1)  # (batch, seq_len, d_model)
            sum_valid = masked_x.sum(dim=1, keepdim=True)  # (batch, 1, d_model)
            
            # Avoid division by zero - use at least 1 for count
            valid_counts = torch.clamp(valid_counts, min=1.0)
            query = sum_valid / valid_counts.unsqueeze(-1)  # (batch, 1, d_model)
        else:
            # Use mean of all tokens
            query = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=(~attn_mask).float()
        )
        
        # Apply layer norm and return pooled representation
        output = self.norm(attn_output.squeeze(1))  # (batch, d_model)
        output = self.linear(output)
        return output

class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False
    ):
        super().__init__()
        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False


        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        if mode == 'attn_pool':
            self.output_transform = AttentionPooling(d_model=d_model, d_output=d_output)
        else:
            self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, x, state=None, lengths=None, l_output=None, attention_mask=None):
        """
        x: (n_batch, l_seq, d_model) or potentially (n_batch, l_seq, d_model, 2) if using rc_conjoin
        Returns: (n_batch, l_output, d_output)
        """
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                if attention_mask is not None:
                    # Find the last valid position for each sequence
                    valid_lengths = attention_mask.sum(dim=1)  # (B,)
                    # Ensure we don't go beyond valid length
                    actual_l_output = torch.minimum(valid_lengths, torch.tensor(l_output, device=x_seq.device))
                    # For each sequence, take the last actual_l_output elements
                    result = []
                    for i in range(x_seq.size(0)):
                        seq_len = actual_l_output[i].item()
                        if seq_len > 0:
                            result.append(x_seq[i, -seq_len:, :])
                        else:
                            # If no valid tokens, return zeros
                            result.append(torch.zeros(l_output, x_seq.size(-1), device=x_seq.device, dtype=x_seq.dtype))
                    return torch.stack(result)
                else:
                    return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                if attention_mask is not None:
                    # Find the first valid position for each sequence
                    valid_lengths = attention_mask.sum(dim=1)  # (B,)
                    # Ensure we don't go beyond valid length
                    actual_l_output = torch.minimum(valid_lengths, torch.tensor(l_output, device=x_seq.device))
                    # For each sequence, take the first actual_l_output elements
                    result = []
                    for i in range(x_seq.size(0)):
                        seq_len = actual_l_output[i].item()
                        if seq_len > 0:
                            result.append(x_seq[i, :seq_len, :])
                        else:
                            # If no valid tokens, return zeros
                            result.append(torch.zeros(l_output, x_seq.size(-1), device=x_seq.device, dtype=x_seq.dtype))
                    return torch.stack(result)
                else:
                    return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            # print(f"Attention mask is : {attention_mask}")
            def restrict(x_seq):
                """Pool entire sequence into single output"""
                # x_seq is (batch, seq_len, d_model)
                if attention_mask is not None:
                    # Apply attention mask: set masked positions to 0
                    masked_x = x_seq * attention_mask.unsqueeze(-1)
                    # Sum over valid (non-masked) positions
                    s = masked_x.sum(dim=1, keepdim=True)
                    # Count valid positions for each sample
                    valid_lengths = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
                    # Avoid division by zero
                    valid_lengths = torch.clamp(valid_lengths, min=1)
                    return s / valid_lengths
                else:
                    s = x_seq.sum(dim=1, keepdim=True)
                    L = x_seq.size(1)
                    return s / L
        elif self.mode == "sum":
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                if attention_mask is not None:
                    # Apply attention mask: set masked positions to 0
                    x_seq = x_seq * attention_mask.unsqueeze(-1)
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'attn_pool':
            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq
        elif self.mode == "len_pool":
            restrict = None
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'ragged']"
            )


        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        elif self.mode == 'len_pool':
            attn_mask = ~torch.all(x == 0, dim=-1)  # (b, n) - True for valid tokens
            val_len = attn_mask.sum(dim=-1, keepdim=True).float()  # (b, 1)

            # Mask out zero locations and sum
            masked_x = x * attn_mask.unsqueeze(-1)  # (b, n, d)
            tot_sum = masked_x.sum(dim=1, keepdim=True)  # (b, 1, d)

            # Avoid division by zero
            val_len = torch.clamp(val_len, min=1.0)
            x = tot_sum / val_len.unsqueeze(-1)  # (b, 1, d)
        else:
            x = restrict(x)
        
        if squeeze and self.mode != 'attn_pool':
            assert x.size(1) == 1
            x = x.squeeze(1)
        
        # import pdb; pdb.set_trace()

        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x.squeeze())
            x_rc = self.output_transform(x_rc.squeeze())
            x = (x + x_rc) / 2
        elif self.mode == 'attn_pool':
            x = self.output_transform(x.squeeze())
        else:
            x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
}

model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
