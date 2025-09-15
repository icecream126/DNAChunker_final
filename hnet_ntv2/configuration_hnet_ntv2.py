"""Configuration for HNet-NTV2 hybrid model.

This configuration combines the Nucleotide Transformer v2 architecture with HNet-style tokenization.
"""

from typing import Optional, Union
from transformers import PretrainedConfig


class HNetNTV2Config(PretrainedConfig):
    """Configuration for HNet-NTV2 hybrid model."""
    model_type = "hnet_ntv2"

    def __init__(
        self,
        # NTV2 model parameters
        d_model: int = 1024,  # NTV2 uses 1024 hidden size
        n_layer: int = 24,   # NTV2 uses 24 layers
        vocab_size: int = 12,  # Character tokenizer vocabulary size (7 special tokens + 5 characters)
        # Note: This will be padded to 16 by the model to match CaduceusTokenizer
        n_head: int = 16,     # NTV2 uses 16 attention heads
        intermediate_size: int = 8192,  # NTV2 MLP intermediate size
        max_position_embeddings: int = 1024,  # Match dataset max_length
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        
        # HNet-specific parameters
        bidirectional: bool = True,
        bidirectional_strategy: Union[str, None] = "add",
        bidirectional_weight_tie: bool = True,
        target_ratio: float = 0.25,
        motif_ratio: float = 0.0,
        
        # Layer configuration for HNet chunking
        n_enc_layer: int = 2,
        n_main_layer: int = 20,  # Most layers are in the main transformer
        n_dec_layer: int = 2,
        
        # HNet tokenization
        tokenizer_type: str = "hnet",  # Use HNet-style tokenization
        use_rotary_embeddings: bool = True,  # NTV2 uses rotary embeddings
        
        # Training parameters
        pad_vocab_size_multiple: int = 8,  # Required to match CaduceusTokenizer padding
        initializer_cfg: Optional[dict] = None,
        
        # Model behavior
        output_hidden_states: bool = False,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # NTV2 parameters
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        
        # HNet parameters
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.target_ratio = target_ratio
        self.motif_ratio = motif_ratio
        
        # Layer configuration
        self.n_enc_layer = n_enc_layer
        self.n_main_layer = n_main_layer
        self.n_dec_layer = n_dec_layer
        
        # Tokenization
        self.tokenizer_type = tokenizer_type
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # Training
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_cfg = initializer_cfg
        
        # Model behavior
        self.output_hidden_states = output_hidden_states
        
        # Ensure total layers match
        self.total_layers = self.n_enc_layer + self.n_main_layer + self.n_dec_layer
