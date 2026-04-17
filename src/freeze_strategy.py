import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional


class BiRefNetFreezeStrategy:
    def __init__(self, model: nn.Module):
        self.model = model
        self._original_state = {}

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self._original_state = {
            name: param.requires_grad for name, param in self.model.named_parameters()
        }

    def apply_phase_1(self):
        self.freeze_all()

        for n, p in self.model.bb.patch_embed.named_parameters():
            p.requires_grad = True

        for n, p in self.model.bb.named_parameters():
            if "norm" in n or "relative_position_bias_table" in n:
                p.requires_grad = True

        for p in self.model.squeeze_module.parameters():
            p.requires_grad = True
        for p in self.model.decoder.parameters():
            p.requires_grad = True

    def apply_phase_2(self):
        self.apply_phase_1()

        for layer_idx in [0, 1]:
            for p in self.model.bb.layers[layer_idx].parameters():
                p.requires_grad = True

        for idx, blk in enumerate(self.model.bb.layers[2].blocks):
            if idx % 3 == 0:
                for p in blk.parameters():
                    p.requires_grad = True

    def apply_phase_3(self):
        self.apply_phase_2()

        for idx, blk in enumerate(self.model.bb.layers[2].blocks):
            if idx % 3 == 1:
                for p in blk.parameters():
                    p.requires_grad = True

        for p in self.model.bb.layers[3].parameters():
            p.requires_grad = True

    def get_trainable_params_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_trainable_by_component(self) -> Dict[str, int]:
        stats = defaultdict(int)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parts = name.split(".")
                component = parts[0] if parts else "unknown"
                stats[component] += param.numel()
        return dict(stats)

    def print_summary(self, phase_name: str = ""):
        if phase_name:
            print(f"\n{'=' * 60}")
            print(f"Phase: {phase_name}")
            print(f"{'=' * 60}")

        trainable = self.get_trainable_params_count()
        total = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

        by_component = self.get_trainable_by_component()
        for comp, params in sorted(by_component.items()):
            print(f"  {comp:25s}: {params:>12,} params")

        return trainable


def create_optimizer(
    model, phase: int = 1, lr_decoder: float = 5e-4, lr_encoder: float = 1e-5
):
    encoder_params = []
    decoder_params = []

    encoder_names = {"bb.", "squeeze"}

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(enc_name in name for enc_name in encoder_names):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr_encoder})

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def apply_weight_decay_exemptions(model):
    no_decay = ["norm", "bias", "relative_position_bias_table"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(exemption in name for exemption in no_decay):
            param.register_hook(lambda grad: grad)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from transformers import AutoModelForImageSegmentation

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    print("Loading model...")
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", token=token, trust_remote_code=True
    )

    strategy = BiRefNetFreezeStrategy(model)

    strategy.apply_phase_1()
    strategy.print_summary("Phase 1: Decoder + Squeeze + Embedding")

    strategy.apply_phase_2()
    strategy.print_summary("Phase 2: + Stage 0-1 + Alternating Stage 2")

    strategy.apply_phase_3()
    strategy.print_summary("Phase 3: Full Training")
