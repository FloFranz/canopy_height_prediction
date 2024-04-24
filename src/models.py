import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, ModuleList, MSELoss
from torchvision.transforms.v2 import RandomCrop, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms.v2 import CenterCrop
from torch.nn.functional import relu, one_hot, tanh, softmax, softmin, interpolate

from params import *

class Augmentation(Module):
	def __init__(self, cropsize) -> None:
		super().__init__()
		self.cropsize = cropsize
		self.aug_crop = RandomCrop(cropsize)
		self.aug_flip_h = RandomHorizontalFlip()
		self.aug_flip_v = RandomVerticalFlip()
	
	def forward(self, x):
		x = self.aug_crop(x)
		x = self.aug_flip_h(x)
		x = self.aug_flip_v(x)
		return x


# ---------------- ---------------- #
# Code below from https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.conv2 = Conv2d(outChannels, outChannels, 3)
        
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		x = relu(self.conv1(x))
		x = self.conv2(x)
		return x

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
		)
		self.pool = MaxPool2d(2)
		
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		
        # loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			
			blockOutputs.append(x)
			x = self.pool(x)
		
        # return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures


class UNet(Module):
	def __init__(
			self, 
            image_dim
		):
		super().__init__()

		self.aug = Augmentation(image_dim)

		# initialize the encoder and decoder
		self.encoder = Encoder((3, 8, 16, 32))
		self.decoder = Decoder((32, 16, 8))
		self.head    = Conv2d(8, 1, 1)

		self.image_dim = image_dim
		
		test_img = torch.zeros((1, 3, *image_dim))
		x = self.encoder(test_img)
		x = self.decoder(x[::-1][0], x[::-1][1:])
		self.target_dim = x.shape[2:]
		self.target_cropper = CenterCrop(self.target_dim)
		
		self.loss_func = MSELoss()
		
	def forward(self, x):
		# Input tensor x contains both input and target, which will be split after augmentation.
		# Augment
		x = self.aug(x)
		aug_target = x[:, 3:4, :, :]
		aug_target = self.target_cropper(aug_target)
		x = x[:, 0:3, :, :]

		# grab the features from the encoder
		x = self.encoder(x)
		
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		x = self.decoder(
			x[::-1][0],
			x[::-1][1:]
		)

		x = self.head(x)
		
		return aug_target, x
# ---------------- ---------------- #


class TreeNetV1(nn.Module):
    def __init__(self, image_crop) -> None:
        super().__init__()
        self.image_crop = image_crop
        # Define architecture layers here

    def forward(self, x):
        return x
    

ARCHITECTURES = {
    "TreeNetV1": TreeNetV1,
	"UNet": UNet
}