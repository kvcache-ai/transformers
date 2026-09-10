"""Small differentiable V4 checks for the KT training backport."""

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
    config = AutoConfig.for_model("deepseek_v4", **{k: v for k, v in small_config().to_dict().items() if k != "model_type"})
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
