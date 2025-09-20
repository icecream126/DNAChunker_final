"""DNA Embedding Model.

Backbones from LM pre-training models, used for downstream tasks.
"""

from functools import partial

import torch
import torch.nn as nn
from flash_attn.utils.generation import GenerationMixin
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.models.mixer_seq_simple import _init_weights as _init_weights_mamba

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None


from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import Caduceus
from caduceus_hnet.configuration_caduceus_hnet import CaduceusHNetConfig
from caduceus_hnet.modeling_caduceus_hnet import CaduceusHNet
from src.models.sequence.long_conv_lm import LMBackbone
from src.models.sequence.long_conv_lm import _init_weights
from hnet_mlm.configuration_hnet import HNetConfig
from hnet_mlm.modeling_hnet import HNetTransformer as HNet
from hnet_twostage.configuration_hnet import HNetConfig as HNetTwostageConfig
from hnet_twostage.modeling_hnet import HNetTransformer as HNetTwostageTransformer


class DNAEmbeddingModel(nn.Module, GenerationMixin):
    """DNA Embedding Model.

    Same as ConvLMHeadModel (in long_conv_lm.py), except no decoder head, we just pass back the hidden states for
    downstream tasks.
    """

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 checkpoint_mlp=False,
                 checkpoint_mixer=False,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, sequence_parallel=True,
                 device=None, dtype=None, return_hidden_state=False, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model  # for decoder
        self.process_group = process_group
        self.return_hidden_state = return_hidden_state
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs, **kwargs
        )

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        """DNA Embedding Model forward pass."""
        hidden_states = self.backbone(input_ids, position_ids=position_ids,
                                      inference_params=inference_params)
        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        return hidden_states, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


class DNAEmbeddingModelMamba(DNAEmbeddingModel):
    """Custom DNA Embedding Model that is compatible with open-source Mamba repo."""

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            conjoin_train=False,
            conjoin_test=False,
            device=None,
            dtype=None,
    ):
        super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
        self.config = config
        d_model = config.d_model
        self.d_model = d_model  # for decoder
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights_mamba,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        """Mamba backbone-specific forward pass that does not use `position_ids`."""
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        return hidden_states, None


class DNAEmbeddingModelCaduceus(DNAEmbeddingModel):
    """Custom DNA Embedding Model that is compatible with Caduceus models."""

    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
            conjoin_train=False,
            conjoin_test=False,
    ):
        super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
        self.config = config
        self.d_model = config.d_model  # for decoder
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(
            config=config,
            **factory_kwargs,
        )

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        """Caduceus backbone-specific forward pass that does not use `position_ids`."""
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            hidden_states = self.caduceus(input_ids, return_dict=False)
            num_chan = hidden_states.shape[-1]
            return torch.stack(
                [hidden_states[..., :num_chan // 2], torch.flip(hidden_states[..., num_chan // 2:], dims=[1, 2])],
                dim=-1
            ), None
        if self.conjoin_train or (self.conjoin_test and not self.training):  # For conjoining / post-hoc conjoining
            assert input_ids.ndim == 3, "Input must be 3D tensor, where channels corresponds to forward and rc strands"
            hidden_states = self.caduceus(input_ids[..., 0], return_dict=False)
            hidden_states_rc = self.caduceus(input_ids[..., 1], return_dict=False)
            # Stack along channel dimension (dim=-1)
            return torch.stack([hidden_states, hidden_states_rc], dim=-1), None

        return self.caduceus(input_ids, return_dict=False), None


class DNAEmbeddingModelCaduceusHNet(DNAEmbeddingModel):
    """Custom DNA Embedding Model that is compatible with Caduceus-HNet models."""

    def __init__(
            self,
            config: CaduceusHNetConfig,
            device=None,
            dtype=None,
            conjoin_train=False,
            conjoin_test=False,
    ):
        super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
        self.config = config
        self.d_model = config.d_model  # for decoder
        factory_kwargs = {"device": device, "dtype": dtype}

        self.caduceus = CaduceusHNet(
            config=config,
            **factory_kwargs,
        )
        # for layer in self.caduceus.backbone.layers[:self.caduceus.backbone.n_enc_layer]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # for param in self.caduceus.backbone.routing_module.parameters():
        #     param.requires_grad = False

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        """Caduceus-HNet backbone-specific forward pass that does not use `position_ids`."""
        out = self.caduceus(input_ids, return_dict=False)
        # hid_states, attn_mask =  out['hidden_states'][1]
        # hid_states = hid_states * (attn_mask.squeeze().unsqueeze(-1)).float()
        # hid_states = self.caduceus(input_ids, return_dict=False)[0]
        return out[0], None

# class DNAEmbeddingModelCaduceusHNetMotif(DNAEmbeddingModel):
#     """Custom DNA Embedding Model that is compatible with Caduceus-HNet models."""

#     def __init__(
#             self,
#             config: CaduceusHNetConfig,
#             device=None,
#             dtype=None,
#             conjoin_train=False,
#             conjoin_test=False,
#     ):
#         super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
#         self.config = config
#         self.d_model = config.d_model  # for decoder
#         factory_kwargs = {"device": device, "dtype": dtype}
#         self.caduceus = CaduceusHNet(
#             config=config,
#             **factory_kwargs,
#         )

#         self.conjoin_train = conjoin_train
#         self.conjoin_test = conjoin_test

#     def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
#         """Caduceus-HNet backbone-specific forward pass that does not use `position_ids`."""
#         # import pdb; pdb.set_trace()
#         # if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
#         #     hidden_states = self.caduceus_hnet(input_ids, return_dict=False)
#         #     num_chan = hidden_states.shape[-1]
#         #     return torch.stack(
#         #         [hidden_states[..., :num_chan // 2], torch.flip(hidden_states[..., num_chan // 2:], dims=[1, 2])],
#         #         dim=-1
#         #     ), None
#         # if self.conjoin_train or (self.conjoin_test and not self.training):  # For conjoining / post-hoc conjoining
#         #     assert input_ids.ndim == 3, "Input must be 3D tensor, where channels corresponds to forward and rc strands"
#         #     hidden_states = self.caduceus_hnet(input_ids[..., 0], return_dict=False)
#         #     hidden_states_rc = self.caduceus_hnet(input_ids[..., 1], return_dict=False)
#         #     # Stack along channel dimension (dim=-1)
#         #     return torch.stack([hidden_states, hidden_states_rc], dim=-1), None

#         return self.caduceus(input_ids, return_dict=False)[0], None

class DNAEmbeddingModelHNet(DNAEmbeddingModel):
    """Custom DNA Embedding Model that is compatible with Caduceus-HNet models."""

    def __init__(
            self,
            config: HNetConfig,
            device=None,
            dtype=None,
            conjoin_train=False,
            conjoin_test=False,
    ):
        super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
        self.config = config
        self.d_model = config.d_model  # for decoder
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = HNet(
            config=config,
            **factory_kwargs,
        )

        # # freeze encoder part
        # for param in self.caduceus.backbone.encoder_layers.parameters():
        #     param.requires_grad = False
        
        # # freeze tokenization part
        # for param in self.caduceus.backbone.routing_module.parameters():
        #     param.requires_grad = False

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        # out = self.caduceus(input_ids, return_dict=True, output_hidden_states=True)
        # hid_states, attn_mask =  out['hidden_states'][1]
        # hid_states = hid_states * (~attn_mask.squeeze().unsqueeze(-1)).float()
        out = self.caduceus(input_ids, return_dict=False)
        # hid_states, attn_mask =  out['hidden_states'][1]
        # hid_states = hid_states * (attn_mask.squeeze().unsqueeze(-1)).float()
        # hid_states = self.caduceus(input_ids, return_dict=False)[0]
        return out[0], None
        # return hid_states, None

class DNAEmbeddingModelHNetTwostage(DNAEmbeddingModel):
    """Custom DNA Embedding Model that is compatible with Caduceus-HNet models."""

    def __init__(
            self,
            config: HNetTwostageConfig,
            device=None,
            dtype=None,
            conjoin_train=False,
            conjoin_test=False,
    ):
        super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
        self.config = config
        self.d_model = config.d_model  # for decoder
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = HNetTwostageTransformer(
            config=config,
            **factory_kwargs,
        )

        # # freeze encoder part
        # for param in self.caduceus.backbone.encoder_layers.parameters():
        #     param.requires_grad = False
        
        # # freeze tokenization part
        # for param in self.caduceus.backbone.routing_module.parameters():
        #     param.requires_grad = False

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):  # state for the repo interface
        out = self.caduceus(input_ids, return_dict=True, output_hidden_states=True)
        hid_states, chunk_lengths =  out['hidden_states'][1]
        final_states = out['hidden_states'][0]

        pad_num = (input_ids==4).float().sum(-1)
        true_val_token_num = (chunk_lengths - pad_num).float()

        if (true_val_token_num <= 0).any():
            print("true val token num is less than 0")
            print(true_val_token_num.min())
            print(true_val_token_num.max())
            raise ValueError
        # remove the righthand 
        
        attn_mask  = torch.arange(hid_states.shape[1], device=hid_states.device)[None, :] >= true_val_token_num[:, None]
        hid_states = hid_states * (~attn_mask).float().unsqueeze(-1)
        # import pdb; pdb.set_trace()
        hid_states_pool = hid_states.sum(dim=1) / (~attn_mask).sum(dim=1).unsqueeze(-1)
        hid_states_expand = hid_states_pool.unsqueeze(1).expand_as(final_states)
        # hid_states = hid_states * (~attn_mask.squeeze().unsqueeze(-1)).float()
        # out = self.caduceus(input_ids, return_dict=True)
        # hid_states, attn_mask =  out['hidden_states'][1]
        # hid_states = hid_states * (attn_mask.squeeze().unsqueeze(-1)).float()
        # hid_states = self.caduceus(input_ids, return_dict=False)[0]
        # return out[0], None
        return hid_states_expand, None


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True):
    """

    Modifies state dict loading with custom function.  This is necessary because the head of
    a lm outputs logits for vocab, but we just need the embeddings for downstream tasks.

    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes, then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # consumes prefix from pretrained model, if necessary
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )

    model_new_params_dict = model.state_dict()
    updated_model_state_dict = {}
    # loop through scratch model keys (pretrained may have extra stuff)
    for key in sorted(model_new_params_dict.keys()):
        loaded_params = state_dict.get(key, None)
        if loaded_params is None:
            # This should never happen, it should be there!
            print("Missing key in pretrained model!", key)
            used_params = model_new_params_dict[key]

        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif "decoder" in key:
            print("found decoder key / parameter, load from scratch", key)
            used_params = model_new_params_dict[key]
        else:
            print('key: shape MATCH, loading', key)  # load matched weights
            used_params = loaded_params

        # we need to pass back a state dict with the '.model' prefix!!!!!
        key_with_prefix = 'model.' + key
        updated_model_state_dict[key_with_prefix] = used_params

    if freeze_backbone:
        print("freezing model backbone params!")
        # note, decoder not included in backbone
        for name, param in model.named_parameters():
            param.requires_grad = False

    # we have updated the new model state dict with pretrained now
    return updated_model_state_dict
