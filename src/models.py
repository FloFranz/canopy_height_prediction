import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, ModuleList
from torchvision.transforms import CenterCrop
from torch.nn.functional import relu, one_hot, tanh, softmax, softmin, interpolate
from params import *

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
		return self.conv2(relu(self.conv1(x)))

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
            image_dim,
			enc_channels=(3, 16, 32, 64),
            dec_channels=(64, 32, 16),
		):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(enc_channels)
		self.decoder = Decoder(dec_channels)
		
		self.image_dim = image_dim
		
	def forward(self, x):
		# grab the features from the encoder
		enc_features = self.encoder(x)
		
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		dec_features = self.decoder(
			enc_features[::-1][0],
			enc_features[::-1][1:]
		)
		
		return dec_features
# ---------------- ---------------- #


class TreeNetV1(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Define architecture layers here
        pass

    def forward(self, x):
        return x
    

ARCHITECTURES = {
    "TreeNetV1": TreeNetV1,
	"UNet": UNet
}