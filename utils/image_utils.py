from typing import Union

import numpy as np

import PIL
from PIL import Image

import torch
from torchvision import transforms


# for ImageNet trained models
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def clamp_image(
    image: torch.tensor, clamp_min: float = 0.0, clamp_max: float = 1.0, normalized: bool = True
) -> torch.tensor:
    """
    Clamp 4-dim-tensor with 3 channels at dim 1 (1, not 0) for each channel

    min and max refer to bounds in REGULAR SPACE, so [0, 1].
    If normalized is True, these bounds get translated to normalized space.

    @param image: image with 4 dims
    @param clamp_min: min
    @param clamp_max: max
    @param normalized: t is in normalized space
    """
    def get_normalized_channel_bounds(bound: float) -> np.array:
        """
        transform a bound (float) in image space to a 1-dim tensor with 3 elements,
        where each element is the bound in normalized space according to MEAN and STD
        """
        return (np.ones(3) * bound - np.array(MEAN)) / np.array(STD)

    if normalized:
        # needed to clip tensors in normalized space with mins and maxs derived from correct normalization means and stds
        clamp_min = get_normalized_channel_bounds(clamp_min)
        clamp_max = get_normalized_channel_bounds(clamp_max)
        # iterate each channel
        for i in range(len(clamp_min)):
            image[:, i:i + 1, :, :] = torch.clamp(
                image[:, i:i + 1, :, :], min=clamp_min[i], max=clamp_max[i]
            )
        return image
    else:
        return torch.clamp(image, min=clamp_min, max=clamp_max)


def scale_tensor(x: torch.Tensor, scale_min: float = 0.0, scale_max: float = 1.0) -> torch.Tensor:
    """
    Scales the input tensor x to range between min and max

    :param x: tensor
    :param scale_min: min
    :param scale_max: max
    :return: scaled tensor
    """
    x_min = x.min()
    x_max = x.max()
    x_std = (x - x_min) / (x_max - x_min)  # x_std is between 0 and 1
    return x_std * (scale_max - scale_min) + scale_min  # stretched to min and max


def rand_images(
    width: int = 224,
    height: int = 224,
    amount: int = 1,
    clean_int: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.tensor:
    """
    get some random images in torch format (batch_size, 3, crop_w, crop_w)

    @param width: width
    @param width: height
    @param amount: amount
    @param clean_int: convert image to [0,255] integers to get realistic images
    @param device: device
    """
    # make some noise
    input_shape = (amount, 3, width, height)
    images = torch.rand(input_shape, dtype=torch.float32, requires_grad=False).to(
        device
    )

    if clean_int:
        images = images * 255
        images = images.int()
        images = images.float()
        images = images / 255

    return images


def randn_images(width: int,
                 height: int,
                 mean: float = 0.5,
                 std: float = 0.1,
                 amount: int = 1,
                 clean_int: bool = False,
                 device: torch.device = torch.device('cpu'),
                 *args, **kwargs) -> torch.tensor:
    """
    get some normal random images in torch format (bs, 3, crop_w, crop_w)

    @param width: width
    @param height: height
    @param mean: mean of uniform noise
    @param std: std of uniform noise
    @param amount: amount
    @param clean_int: convert image to [0,255] integers to get realistic images
    @param device: device
    @return: images tensor
    """
    # make some noise
    input_shape = (amount, 3, width, height)
    images = torch.normal(mean=mean, std=std, size=input_shape, dtype=torch.float32, requires_grad=False).to(device)

    if clean_int:
        images = images * 255
        images = images.int()
        images = images.float()
        images = images / 255

    return images


def batch_tensor(t: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """
    Get list of batches from larger tensor

    @param t: tensor
    @param batch_size: batch_size
    @return list of tensors
    """
    assert len(t.size()) > 1
    tensor_size = t.size(dim=0)
    # assert tensor_size >= batch_size
    if tensor_size < batch_size:
        batch_size = tensor_size

    batches = []
    for batch_id in range(0, int(tensor_size / batch_size)):
        start_id = batch_id * batch_size
        end_id = (batch_id + 1) * batch_size
        batches.append(t[start_id:end_id])
    if end_id != tensor_size:
        batches.append(t[end_id:tensor_size])

    return batches


class Normalize(transforms.Normalize):
    """
    Wrapper for Normalize with pre-set mean & std.

    @param mean: mean
    @param std: std
    """

    def __init__(
        self,
        mean: list[float, float, float] = MEAN,
        std: list[float, float, float] = STD,
        *args,
        **kwargs
    ):
        super().__init__(mean=mean, std=std, *args, **kwargs)


class UnNormalize(object):
    """
    see https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    """

    def __init__(
        self,
        mean: list[float, float, float] = MEAN,
        std: list[float, float, float] = STD,
    ):
        self.mean = mean
        self.std = std
        self.un_normalize = transforms.Normalize(
            mean=[
                -self.mean[0] / self.std[0],
                -self.mean[1] / self.std[1],
                -self.mean[2] / self.std[2],
            ],
            std=[1 / self.std[0], 1 / self.std[1], 1 / self.std[2]],
        )

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size ((batch_size), C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        inv_tensor = self.un_normalize(tensor)
        return inv_tensor


def get_preprocessing(
    scale_w: int = 256,
    crop_w: int = 224,
    normalize: bool = False,
    to_tensor: bool = True,
    mean: tuple[float, float, float] = MEAN,  # imagenet
    std: tuple[float, float, float] = STD,  # imagenet
):
    """
    Get torchvision image transform composition.
    Can only do square outputs.
    If crop is True, images will be scaled to 256x256, then center cropped to input shape[0].
    If not, it will be scaled to input shape.
    If normalize is True, images will be normalized to mean and std.

    @param scale_w: scale_w
    @param crop_w: crop_w
    @param normalize: normalize
    @param to_tensor: to_tensor
    @param mean: mean
    @param std: std
    """
    transform_list = [transforms.Resize((scale_w, scale_w), antialias=True)]
    if crop_w is not None:
        transform_list.append(transforms.CenterCrop(crop_w))
    if to_tensor:
        transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def open_image_as_pil(path: str) -> PIL.Image:
    """
    Open image at path and return PIL image

    @param path: path
    @return: PIL image
    """
    return PIL.Image.open(path).convert("RGB")


def open_image_as_tensor(path: str) -> torch.tensor:
    """
    Open image at path and return tensor image

    @param path: path
    @return: tensor image
    """
    return transforms.ToTensor()(open_image_as_pil(path))


def std_filter(
    img: Union[torch.Tensor, np.array], width: int, device: torch.device = torch.device("cpu")
) -> Union[torch.Tensor, np.array]:
    """
    torch custom filter which outputs std for each window, then mean

    @param img: either torch tensor or numpy array. Either single (3 dims) batch (4 dims).
    @param width: filter width
    @param device: device
    @return either torch tensor or np nd array both with shape <batch_size> (1 in case there is no batch dim)
    """

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        img = img.to(device)
        was_numpy = True
    elif torch.is_tensor(img):
        was_numpy = False
    else:
        raise Exception("only numpy and torch allowed in std_filter")

    # torch
    # make batch
    if img.dim() == 3:
        img = img.unsqueeze(0)
    # make greyscale
    img = img.mean(1, keepdims=True)

    # custom std
    # make batch
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # make greyscale
    img = img.mean(1, keepdims=True)

    # custom std convolution
    x = torch.nn.functional.unfold(img, kernel_size=(width, width))
    x = x.unsqueeze(1)
    x = x.permute((0, 1, 3, 2))
    x = x.std(3)
    new_img_width = img.size()[2] - width + 1
    batch_size = img.size()[0]
    x = x.reshape((batch_size, 1, new_img_width, new_img_width))

    x = x.mean((1, 2, 3))

    if was_numpy:
        x = x.detach().cpu().numpy()

    return x
