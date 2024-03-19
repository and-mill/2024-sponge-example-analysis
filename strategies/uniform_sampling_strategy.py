import torch

from utils.image_utils import randn_images


def generate(mean: float = 0.1,
             std: float = 2/255,
             amount: int = 1) -> torch.Tensor:
    """
    Simple strategy to generate sponge examples by sampling from uniform noise with tight std.

    We use std = 2/255 like in the paper.
    
    The optimal mean depends on the target model (see Section IV A. in the paper):
    ResNet18: 0.1
    ResNet50: 0.0
    ResNet101: 0.0
    DenseNet121: 0.3
    DenseNet161: 0.0
    DenseNet201: 0.3
    MobileNetV2:  0.0

    Note that randn_images does quantization to ints (clean_ints) to be sure that generated images are valid.

    @param amount: The amount of images to generate
    @return: A tensor of shape (amount, 3, 224, 224) containing the generated images
    """
    images = randn_images(width=224, height=224, mean=mean, std=std, amount=amount, clean_int=True)

    return images
