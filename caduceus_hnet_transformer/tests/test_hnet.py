"""Tests for HNet modules."""

import pytest
import torch

from caduceus_hnet import CaduceusHNetForMaskedLM, CaduceusHNetConfig




@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("n_layer", [4])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_hnet_forward_pass(batch_size, seq_len, n_layer, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.random.manual_seed(0)

    # Setup CaduceusHNetConfig
    config = CaduceusHNetConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=12,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        target_ratio=0.25,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    model = CaduceusHNetForMaskedLM(
        config=config,
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=1, high=config.vocab_size, size=(batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Test forward pass
    outputs = model(input_ids, labels=labels, return_dict=True)

    # Check that loss is a single value
    assert outputs.loss.shape == torch.Size([])
    assert outputs.loss.dtype == dtype

    # Check that logits have the correct shape
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
