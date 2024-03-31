import torch
import torch.nn as nn
from torch.nn.functional import relu, one_hot, tanh, softmax, softmin
from params import *

class TreeNetV1(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Define architecture layers here
        pass

    def forward(self, x):
        return x
    

ARCHITECTURES = {
    "TreeNetV1": TreeNetV1
}