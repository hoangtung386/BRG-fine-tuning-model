import torch
from transformers import AutoModelForImageSegmentation


def reinitialize_patch_embed_for_lineart(model):
    """Average RGB patch-embed filters across channels for line-art adaptation."""
    patch_proj = getattr(getattr(model, "bb", None), "patch_embed", None)
    patch_proj = getattr(patch_proj, "proj", None)
    if patch_proj is None or not hasattr(patch_proj, "weight"):
        return

    with torch.no_grad():
        w = patch_proj.weight
        if w.dim() == 4 and w.shape[1] > 1:
            w_avg = w.mean(dim=1, keepdim=True)
            patch_proj.weight.copy_(w_avg.repeat(1, w.shape[1], 1, 1))


def load_model(
    model_name="briaai/RMBG-2.0", device="cuda", use_gradient_checkpointing=False
):
    model = AutoModelForImageSegmentation.from_pretrained(
        model_name, trust_remote_code=True
    )
    reinitialize_patch_embed_for_lineart(model)

    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = model.to(device)
    return model


def unfreeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = True


def create_optimizer(model, lr_decoder=2e-5, lr_encoder=5e-6, weight_decay=0.01):
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "encoder" in name.lower():
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": decoder_params, "lr": lr_decoder},
        ],
        weight_decay=weight_decay,
    )
