"""Caduceus config for Hugging Face.

"""

from typing import Optional, Union

from transformers import PretrainedConfig


class HNetConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "hnet"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            target_ratio: float = 0.3,  # Legacy parameter for backward compatibility
            target_ratio_stage1: float = 0.5,  # Coarse chunking ratio
            target_ratio_stage2: float = 0.3,  # Super-coarse chunking ratio
            motif_ratio: float = 0.0,
            
            # Layer configuration
            n_enc_layer: int = 1,
            n_main_layer: int = 2,
            n_dec_layer: int = 1,

            # Transformer block params
            transformer_n_head: int = 8,
            transformer_mlp_mult: int = 4,
            tokenizer_type: str = "default",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.target_ratio = target_ratio
        self.target_ratio_stage1 = target_ratio_stage1
        self.target_ratio_stage2 = target_ratio_stage2
        self.n_enc_layer = n_enc_layer
        self.n_main_layer = n_main_layer
        self.n_dec_layer = n_dec_layer
        self.transformer_n_head = transformer_n_head
        self.transformer_mlp_mult = transformer_mlp_mult
        self.tokenizer_type = tokenizer_type
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        
        self.n_layer = self.n_enc_layer + self.n_main_layer + self.n_dec_layer