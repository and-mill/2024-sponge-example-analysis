import torch
from torchvision import models

MODELS = {
    "ResNet18": {
        "model": models.resnet18,
        "params": {"weights": models.ResNet18_Weights.DEFAULT},
    },
    "ResNet50": {
        "model": models.resnet50,
        "params": {"weights": models.ResNet50_Weights.DEFAULT},
    },
    "ResNet101": {
        "model": models.resnet101,
        "params": {"weights": models.ResNet101_Weights.DEFAULT},
    },
    "DenseNet121": {
        "model": models.densenet121,
        "params": {"weights": models.DenseNet121_Weights.DEFAULT},
    },
    "DenseNet161": {
        "model": models.densenet161,
        "params": {"weights": models.DenseNet161_Weights.DEFAULT},
    },
    "DenseNet201": {
        "model": models.densenet201,
        "params": {"weights": models.DenseNet201_Weights.DEFAULT},
    },
    "MobileNetV2": {
        "model": models.mobilenet_v2,
        "params": {"weights": models.MobileNet_V2_Weights.DEFAULT},
    },
}


def disable_in_place_ops_in_model(m: torch.nn.Module):
    """
    Disable inplace operations in model
    @param m: model
    """
    leaf_nodes = [
        module
        for module in m.modules()
        if type(module) == torch.nn.modules.activation.ReLU
    ]
    for xl in leaf_nodes:
        xl.inplace = False


def get_model(
    model_name: str,
    device: torch.device = torch.device("cpu"),
    disable_inplace: bool = True,
) -> torch.nn.Module:
    """
    Get model by name.

    @param model_name: model_name
    @param device: device
    @param disable_inplace: disable inplace operations
    @return: model
    """
    model_metadata = MODELS[model_name]
    model = model_metadata["model"]
    params = model_metadata["params"]
    model = model(**params)
    if disable_inplace:
        disable_in_place_ops_in_model(model)
    model = model.to(device)
    model.eval()

    return model
