import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

from .funcs import torchvision_isinstance

EPS = 1e-3

class ImageStandardization(Transform):

    def _transform(self, inpt, params):
        if not torchvision_isinstance(inpt, (torch.Tensor, tv_tensors.Image)):
            return inpt

        std, mean = torch.std_mean(inpt, dim = (-1, -2), keepdim = True)

        return (inpt - mean) / (std + EPS)

