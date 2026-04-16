import torch
from transformers import AutoModelForImageSegmentation


def load_model(model_name="briaai/RMBG-2.0", device="cuda"):
    model = AutoModelForImageSegmentation.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = model.to(device)
    return model


def unfreeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = True


def create_optimizer(model, lr=2e-5, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
