"""Small differentiable V4 checks for the KT training backport."""

import copy
from contextlib import nullcontext

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM, DeepseekV4Config, DeepseekV4ForCausalLM


def small_config():
    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        sliding_window=16,
        max_position_embeddings=4096,
        qk_rope_head_dim=16,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        use_cache=False,
        attn_implementation="eager",
    )


def small_model():
    torch.manual_seed(42)
    model = DeepseekV4ForCausalLM(small_config())
    with torch.no_grad():
        ids = torch.arange(64)
        model.model.layers[0].mlp.gate.tid2eid.copy_(torch.stack((ids % 4, (ids + 1) % 4), dim=-1))
    return model


def test_auto_registration():
    config = AutoConfig.for_model(
        "deepseek_v4", **{k: v for k, v in small_config().to_dict().items() if k != "model_type"}
    )
    assert isinstance(AutoModelForCausalLM.from_config(config), DeepseekV4ForCausalLM)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_lora_backward_through_all_attention_types(checkpointing):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        small_model(),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_a_proj", "o_b_proj"]),
    )
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    tokens = torch.randint(2, 64, (1, 132))
    labels = tokens.clone()
    labels[:, :8] = -100
    result = model(input_ids=tokens, labels=labels, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        elif "lora_" not in name:
            assert parameter.grad is None, name


def test_reference_save_reload(tmp_path):
    model = small_model().eval()
    tokens = torch.randint(2, 64, (1, 16))
    with torch.no_grad():
        before = model(input_ids=tokens).logits
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    with torch.no_grad():
        after = restored(input_ids=tokens).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def lora_model():
    from peft import LoraConfig, get_peft_model

    targets = [
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_proj",
        "self_attn.o_b_proj",
        "self_attn.compressor.kv_proj",
        "self_attn.compressor.gate_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    ]
    return get_peft_model(small_model(), LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets))


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpointing_preserves_loss_and_adapter_gradients(reentrant):
    reference = lora_model().train()
    # Nonzero B exercises gradients of both adapter factors, including compressors.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
    checkpointed = copy.deepcopy(reference)
    checkpointed.enable_input_require_grads()
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    tokens = torch.randint(2, 64, (1, 132))
    expected = reference(tokens, labels=tokens, use_cache=False).loss
    actual = checkpointed(tokens, labels=tokens, use_cache=False).loss
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for (name, left), (_, right) in zip(reference.named_parameters(), checkpointed.named_parameters()):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None, name
            torch.testing.assert_close(right.grad, left.grad, msg=name)


def test_lora_updates_both_factors_without_updating_base():
    model = lora_model().train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tokens = torch.randint(2, 64, (1, 132))
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.isfinite(parameter.grad).all(), name
            assert not torch.equal(parameter, before[name]), name
        else:
            assert parameter.grad is None, name
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("boundary", [3, 4, 5, 127, 128, 129])
def test_compressed_attention_does_not_leak_future_tokens(boundary):
    model = small_model().eval()
    tokens = torch.randint(2, 64, (1, boundary + 5))
    changed = tokens.clone()
    changed[:, boundary:] = (changed[:, boundary:] + 1) % 64
    with torch.no_grad():
        before = model(tokens).logits[:, :boundary]
        after = model(changed).logits[:, :boundary]
    # Changed routing can change expert GEMM shapes and FP32 rounding.
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    embeddings = model.get_input_embeddings()(tokens).detach().requires_grad_(True)
    hook = model.get_input_embeddings().register_forward_hook(lambda module, args, output: embeddings)
    try:
        score = model(tokens).logits[:, boundary - 1, 0].sum()
        gradient = torch.autograd.grad(score, embeddings)[0]
    finally:
        hook.remove()
    assert gradient[:, :boundary].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, boundary:]) == 0


@pytest.mark.parametrize(
    "autocast",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="April V4 BF16 requires autocast; LF rejects the unsupported recipe",
            ),
        ),
        True,
    ],
)
def test_bf16_loading_preserves_fp32_norms_and_backward(autocast):
    reference = small_model()
    model = DeepseekV4ForCausalLM.from_pretrained(
        None,
        config=reference.config,
        state_dict=reference.state_dict(),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    assert model.model.layers[0].input_layernorm.weight.dtype == torch.float32
    assert model.model.layers[0].self_attn.q_a_proj.weight.dtype == torch.bfloat16
    tokens = torch.randint(2, 64, (1, 132))
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        loss = model(tokens, labels=tokens, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.model.layers[0].self_attn.q_a_proj.weight.grad).all()


@pytest.mark.parametrize("length", [3, 5, 129])
def test_right_padding_preserves_valid_logits(length):
    model = small_model().eval()
    tokens = torch.randint(2, 64, (2, 132))
    mask = torch.ones_like(tokens)
    mask[0, length:] = 0
    with torch.no_grad():
        expected = model(tokens[:1, :length]).logits
        actual = model(tokens, attention_mask=mask).logits[:1, :length]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="April V4 DynamicLayer lacks store_compression_weights; cached generation is outside the supported LF/SGLang path",
)
@pytest.mark.parametrize("length", [3, 4, 127, 128])
def test_cached_decode_matches_full_forward_at_compression_boundaries(length):
    model = small_model().eval()
    tokens = torch.randint(2, 64, (1, length + 1))
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits[:, -1:]
        prefix = model(tokens[:, :-1], use_cache=True)
        actual = model(tokens[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
