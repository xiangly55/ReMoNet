import torch
import torch.nn as nn

import math 

from collections import OrderedDict

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock_skipcon(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock_skipcon, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*3, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock_skipcon_7channel(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock_skipcon_7channel, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*7, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)





class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x



class DenBlock_1frame(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=1):
		super(DenBlock_1frame, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in0 - x

		return x


class tf_layer_conv2d(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_ch=64, out_ch=64, kernel_size=3, use_bias=False, activation='leaky_relu', padding='same'):
		super(tf_layer_conv2d, self).__init__()
		
		if padding == 'same':
			padding = kernel_size // 2
		else:
			padding = 0
		self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=use_bias)
		if activation == 'leaky_relu':
			self.activation = nn.LeakyReLU()
		else:
			self.activation = nn.ReLU()


	def forward(self, x):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x = self.conv(x)
		x = self.activation(x)

		return x




class NoiseEstimationNet(nn.Module):
	""" Definition of the Noise Estimation Subnetwork.
	Inputs of constructor:
		num_input_frames: Use how many frames to estimate noise map, default use only one
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
	Outputs:
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=1, mid_ch=32):
		super(NoiseEstimationNet, self).__init__()
		self.noise_estimation_block = nn.Sequential(
			nn.Conv2d(num_input_frames*3, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1, stride=1, bias=False),
			nn.ReLU(inplace=True),
		)
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
		   Output: 
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		noise_estimation = self.noise_estimation_block(x)
		return noise_estimation


class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x


class ViDeNN_Spatial(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=1):
		super(ViDeNN_Spatial, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.input_layer = nn.Sequential(nn.Conv2d(3*num_input_frames, 128, kernel_size=3, padding=1), \
							tf_layer_conv2d(in_ch=128, out_ch=64, kernel_size=3, use_bias=False, activation='leaky_relu', padding='same')) 
		module_list = []
		for i in range(3, 20):
			module_list.append(tf_layer_conv2d(in_ch=64, out_ch=64, kernel_size=3, use_bias=False, activation='leaky_relu', padding='same'))
		
		self.hidden = nn.Sequential(*module_list)
		self.out_layer = nn.Conv2d(64, 3, [3, 3], bias=False, padding=1)
		# self.temp1 = DenBlock(num_input_frames=3)
		# self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		# self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		# (_, inp_middle, _) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		# x20 = self.temp1(x0, x1, x2, noise_map)
		# x21 = self.temp1(x1, x2, x3, noise_map)
		# x22 = self.temp1(x2, x3, x4, noise_map)
		out = self.input_layer(x)
		out = self.hidden(out)
		out = self.out_layer(out)
		#Second stage

		return x - out



class ViDeNN_Temp(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(ViDeNN_Temp, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.input_layer = nn.Sequential(nn.Conv2d(3*num_input_frames, 128, kernel_size=3, padding=1), \
							tf_layer_conv2d(in_ch=128, out_ch=64, kernel_size=3, use_bias=False, activation='leaky_relu', padding='same')) 
		module_list = []
		for i in range(3, 20):
			module_list.append(tf_layer_conv2d(in_ch=64, out_ch=64, kernel_size=3, use_bias=False, activation='leaky_relu', padding='same'))
		
		self.hidden = nn.Sequential(*module_list)
		self.out_layer = nn.Conv2d(64, 3, [3, 3], bias=False, padding=1)
		# self.temp1 = DenBlock(num_input_frames=3)
		# self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		# self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(_, inp_middle, _) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		# x20 = self.temp1(x0, x1, x2, noise_map)
		# x21 = self.temp1(x1, x2, x3, noise_map)
		# x22 = self.temp1(x2, x3, x4, noise_map)
		out = self.input_layer(x)
		out = self.hidden(out)
		out = self.out_layer(out)
		#Second stage

		return inp_middle - out



class ViDeNN(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(ViDeNN, self).__init__()
		self.num_in_frames = num_input_frames
		self.spatial = ViDeNN_Spatial(num_input_frames=1)
		self.temp = ViDeNN_Temp(num_input_frames=num_input_frames)

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def freeze_module(self, module_name='spatial', requires_grad=True):
		if module_name == 'spatial':
			module = self.spatial
		else:
			module = self.temp
		for n, p in module.named_parameters():
			p.requires_grad = requires_grad


	def forward(self, x, spatial_only=False):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_in_frames))
		
		if spatial_only:
			x1 = self.spatial(x1)
			return x1

		x0 = self.spatial(x0)
		x1 = self.spatial(x1)
		x2 = self.spatial(x2)
		
		x_spatial = torch.cat([x0, x1, x2], dim=1)
		# First stage
		# x20 = self.temp1(x0, x1, x2, noise_map)
		# x21 = self.temp1(x1, x2, x3, noise_map)
		# x22 = self.temp1(x2, x3, x4, noise_map)
		x_temp = self.temp(x_spatial)
		#Second stage

		return x_temp



def sequential(*args):
	"""Advanced nn.Sequential.
	Args:
		nn.Sequential, nn.Module
	Returns:
		nn.Sequential
	"""
	if len(args) == 1:
		if isinstance(args[0], OrderedDict):
			raise NotImplementedError('sequential does not support OrderedDict input.')
		return args[0]  # No sequential is needed.
	modules = []
	for module in args:
		if isinstance(module, nn.Sequential):
			for submodule in module.children():
				modules.append(submodule)
		elif isinstance(module, nn.Module):
			modules.append(module)
	return nn.Sequential(*modules)



def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'v':
			L.append(nn.Upsample(scale_factor=4, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return sequential(*L)

class DnCNN(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	# https://github.com/cszn/KAIR/blob/master/models/
	def __init__(self, num_input_frames=1, in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR'):
		super(DnCNN, self).__init__()
		self.num_input_frames = num_input_frames
		bias = True
		m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
		m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
		m_tail = conv(nc, out_nc, mode='C', bias=bias)

		self.model = sequential(m_head, *m_body, m_tail)


	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		# (_, inp_middle, _) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		# x20 = self.temp1(x0, x1, x2, noise_map)
		# x21 = self.temp1(x1, x2, x3, noise_map)
		# x22 = self.temp1(x2, x3, x4, noise_map)
		n = self.model(x)
		#Second stage

		return x - n


class SepConv(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
		super(SepConv, self).__init__()
		groups = min(in_ch, out_ch)
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False),
			nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=False),
		)

	def forward(self, x):
		return self.convblock(x)

class EncoderBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(EncoderBlock, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, out_ch//4, kernel_size=5),
			nn.ReLU(),
			SepConv(out_ch//4, out_ch, kernel_size=5),
		)

	def forward(self, x):
		return x + self.convblock(x)


class DownsampleBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(DownsampleBlock, self).__init__()
		self.conv1 = nn.Sequential(
			SepConv(in_ch, out_ch//4, kernel_size=5, stride=2),
			nn.ReLU(),
			SepConv(out_ch//4, out_ch, kernel_size=5),
		)
		self.conv2 = SepConv(in_ch, out_ch, kernel_size=3, stride=2)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x)
		return x1 + x2

class DecoderBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(DecoderBlock, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, out_ch, kernel_size=3),
			nn.ReLU(),
			SepConv(out_ch, out_ch, kernel_size=3),
		)

	def forward(self, x):
		return x + self.convblock(x)

class UpsampleBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self):
		super(UpsampleBlock, self).__init__()
		# self.conv = nn.ConvTranspose2d(in_ch, out_ch, stride=2, kernel_size=2)
		self.ps = nn.PixelShuffle(2)

	def forward(self, x):
		# return self.conv(x)
		return self.ps(x)





class UNetSkipBlock(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_ch, out_ch):
		super(UNetSkipBlock, self).__init__()
		self.in_layer = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
		self.encoder1 = nn.Sequential(
			DownsampleBlock(16, 64),
			EncoderBlock(64, 64),
			)
		self.encoder2 = nn.Sequential(
			DownsampleBlock(64, 256),
			EncoderBlock(256, 256),
			)
		# self.encoder3 = nn.Sequential(
		# 	DownsampleBlock(128, 128),
		# 	EncoderBlock(128, 256),
		# 	)
		# self.encoder4 = nn.Sequential(
		# 	DownsampleBlock(256, 256),
		# 	EncoderBlock(256, 512),
		# 	)
		
		# self.skip_sepconv1 = SepConv(256, 64)
		# self.skip_sepconv2 = SepConv(128, 32)
		# self.skip_sepconv3 = SepConv(64, 32)
		# self.skip_sepconv4 = SepConv(16, 16)
		
		# self.decoder1 = nn.Sequential(
		# 	DecoderBlock(512, 64),
		# 	UpsampleBlock(64, 64),
		# )
		# self.decoder2 = nn.Sequential(
		# 	DecoderBlock(64, 32),
		# 	UpsampleBlock(32, 32),
		# )
		self.decoder3 = nn.Sequential(
			DecoderBlock(256, 256),
			UpsampleBlock(),
		)
		self.decoder4 = nn.Sequential(
			DecoderBlock(64, 64),
			UpsampleBlock(),
		)
		
		self.out_layer = nn.Conv2d(16, out_ch, [3, 3], bias=False, padding=1)
		# self.temp1 = DenBlock(num_input_frames=3)
		# self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		# self.reset_params()
		# self.reset_params()


	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		x1 = self.in_layer(x)		
		# [N, 16, 96, 96]
		x2 = self.encoder1(x1)
		# [N, 64, 48, 48]
		x3 = self.encoder2(x2)
		# ([N, 128, 24, 24])
		# x4 = self.encoder3(x3)
		# # [N, 256, 12, 12]
		# x5 = self.encoder4(x4)
		# [N, 512, 6, 6]
		# skip1 = self.skip_sepconv1(x4)
		# # [N, 64, 12, 12]
		# skip2 = self.skip_sepconv2(x3)
		# skip3 = self.skip_sepconv3(x2)
		# skip4 = self.skip_sepconv4(x1)
		skip3 = x2
		skip4 = x1

		# out1 = self.decoder1(x5)
		# [N, 64, 12, 12]
		# out1 = out1 + skip1
		# out2 = self.decoder2(out1)
		# [N, 32, 24, 24]
		
		# deal with out2.shape != skip2.shape
		# if out2.shape != skip2.shape:
		# 	out2 = out2[:,:,:skip2.shape[2], :skip2.shape[3]]
		# out2 = out2 + skip2
		out2 = x3
		out3 = self.decoder3(out2)
		# [N, 32, 48, 48]
		
		out3 = out3 + skip3
		out4 = self.decoder4(out3)
		# [10, 16, 96, 96]
		out4 = out4 + skip4
		out = self.out_layer(out4)
		if out.shape == x.shape:
			out = out + x
		return out



def split_channel(img, ch=1):
	# (64,9,96,96)
	img_split = img.split(3, dim=1)
	img_split = torch.stack(img_split, dim=ch)
	return img_split






class ReMoNet(nn.Module):

	def __init__(self, num_input_frames=5, num_output_frames=5):
		super(ReMoNet, self).__init__()
		self.num_input_frames = num_input_frames
		self.num_output_frames = num_output_frames
		# define / retrieve model parameters
		factor = 1
		filters = 16
		kernel_size = 3
		# layers = 7
		self.state_dim = 32
		self.hidden_frames = 3
		self.hidden_stride = 1
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.act = nn.ReLU()
		self.conv1 = nn.Conv2d(3*self.hidden_frames + 3*self.hidden_frames + self.state_dim, filters, kernel_size, padding=int(kernel_size/2))
		# self.conv_list = nn.ModuleList([nn.Conv2d(filters, filters, kernel_size, padding=int(kernel_size/2)) for _ in range(layers-2)])
		self.unetblock = UNetSkipBlock(in_ch=filters, out_ch=filters)
		self.conv_out = nn.Conv2d(filters, 3*self.hidden_frames + self.state_dim, kernel_size, padding=int(kernel_size/2))

		self.cell_output_seq_len = math.ceil((num_input_frames) / self.hidden_stride) * self.hidden_frames
		
		fusion_hidden_dim = 32

		self.conv_fusion = UNetSkipBlock(in_ch=3*self.cell_output_seq_len, out_ch=3*self.num_output_frames)

	def cell(self, x, fb, state):

		# retrieve parameters
		factor = 1

		# define network
		res = x  # keep x for residual connection

		# input = torch.cat([x[:, 0], x[:, 1], x[:, 2],
		#                    shuffle_down(fb, factor),
		#                    state], -3)

		inp = torch.cat([x[:, 0], x[:, 1], x[:, 2],
						fb, state], -3)
		# x.shape torch.Size([32, 3, 3, 96, 96])
		# fb.shape torch.Size([32, 3, 96, 96])
		# state.shape torch.Size([32, 128, 96, 96])
		# inp.shape torch.Size([32, 140, 96, 96])
		
		# first convolution                   
		x = self.act(self.conv1(inp))

		# main convolution block
		# for layer in self.conv_list:
		# 		x = self.act(layer(x))
		x = self.unetblock(x)

		x = self.conv_out(x)
		
		# out = shuffle_up(x[..., :3*factor**2, :, :] + res.repeat(1, factor**2, 1, 1), factor)
		out = x[..., :3*self.hidden_frames, :, :] 
		out = out + res.view(out.shape)

		state = self.act(x[..., 3*self.hidden_frames:, :, :])

		return out, state

	def forward(self, x):
		# [n,5,3,96,96]
		# retrieve device
		# device = params["device"]

		if len(x.shape) < 5:
			x = split_channel(x)
		# retrieve parameters
		factor = 1
		

		# x = split_channel(x)
		seq = []
		for i in range(x.shape[1]):                

			if i == 0:
				# out = shuffle_up(torch.zeros_like(x[:, 0]).repeat(1, factor**2, 1, 1), factor)
				out = torch.zeros_like(x[:, 0]).repeat(1, self.hidden_frames, 1, 1)
				state = torch.zeros_like(x[:, 0, 0:1, ...]).repeat(1, self.state_dim, 1, 1)

				out, state = self.cell(torch.cat([x[:, i:i+1], x[:, i:i+2]], 1), out, state)

			elif i == x.shape[1]-1:
				
				out, state = self.cell(torch.cat([x[:, i-1:i+1], x[:, i:i+1]], 1), out, state)

			else:
				
				out, state = self.cell(x[:, i-1:i+2], out, state)

			seq.append(out)

		seq_stack = torch.stack(seq, 1)
		n, c, t, h, w = seq_stack.shape
		seq_stack = seq_stack.view(n, c*t, h, w)

		seq_fuse = self.conv_fusion(seq_stack)
		seq_fuse = seq_fuse.view(x.shape)
		return seq_fuse






class PKT(nn.Module):
	"""Probabilistic Knowledge Transfer for deep representation learning
	Code from author: https://github.com/passalis/probabilistic_kt"""
	def __init__(self):
		super(PKT, self).__init__()

	def forward(self, f_s, f_t):
		bs = f_s.shape[0]

		return self.cosine_similarity_loss(f_s.view(bs, -1), f_t.view(bs, -1))

	@staticmethod
	def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
		# Normalize each vector by its norm
		output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
		output_net = output_net / (output_net_norm + eps)
		output_net[output_net != output_net] = 0

		target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
		target_net = target_net / (target_net_norm + eps)
		target_net[target_net != target_net] = 0

		# Calculate the cosine similarity
		model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
		target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

		# Scale cosine similarity to 0..1
		model_similarity = (model_similarity + 1.0) / 2.0
		target_similarity = (target_similarity + 1.0) / 2.0

		# Transform them into probabilities
		model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
		target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

		# Calculate the KL-divergence
		loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

		return loss



class Similarity(nn.Module):
	"""Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
	def __init__(self):
		super(Similarity, self).__init__()

	def forward(self, g_s, g_t):
		loss = 0.0
		for f_s, f_t in zip(g_s, g_t):
			loss += self.similarity_loss(f_s, f_t)
		# loss = [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
		return loss
		# return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

	def similarity_loss(self, f_s, f_t):
		bsz = f_s.shape[0]
		f_s = f_s.view(bsz, -1)
		f_t = f_t.view(bsz, -1)

		G_s = torch.mm(f_s, torch.t(f_s))
		# G_s = G_s / G_s.norm(2)
		G_s = torch.nn.functional.normalize(G_s)
		G_t = torch.mm(f_t, torch.t(f_t))
		# G_t = G_t / G_t.norm(2)
		G_t = torch.nn.functional.normalize(G_t)

		G_diff = G_t - G_s
		loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
		return loss



if __name__ == '__main__':
	TEST_SHAPE = (540, 960)
	TEST_NUM_INPUT_FRAMES = 5
	from torchstat import stat
	from pthflops import count_ops
	from thop import profile
	from thop import clever_format

	model_name = 'ReMoNet'

	# import time
	# inp =  torch.randn(1, 3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1]).cuda()
	# model_gpu = eval(model_name + "()").cuda()
	# start = time.time()
	# model_gpu(inp)
	# end = time.time()
	# print("Time used {:.3}s".format(end-start))


	############## step 1: specifiy models
	model = eval(model_name + "()")
	# model = FastDVDnet_TSM1()
	# model = FastDVDnet_MultiOut1()

	############## 
	# https://github.com/Lyken17/pytorch-OpCounter
	# noisyframe = torch.randn(1, 3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1])
	# # sigma_noise = torch.randn(1, 1, noisyframe.shape[2], noisyframe.shape[3])
	# # inp = (noisyframe, sigma_noise)
	# # inp = noisyframe
	# # calculate_FLOPs(model, inp)
	# macs, params=profile(model, inputs=[noisyframe])
	# macs, params = clever_format([macs, params], "%.3f")
	# print(macs, params)

	##############  
	# https://github.com/Swall0w/torchstat
	# stat(model, (3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1]))

	# ##############	
	# https://github.com/1adrianb/pytorch-estimate-flops
	# bug
	# noisyframe = torch.randn(1, 3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1])
	# sigma_noise = torch.randn(1, 1, noisyframe.shape[2], noisyframe.shape[3])
	# inp = (noisyframe, sigma_noise)
	# count_ops(model, noisyframe)

	##############
	# https://github.com/zhijian-liu/torchprofile
	# requires torch >= 1.4
	# from torchprofile import profile_macs

	# noisyframe = torch.randn(1, 3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1])
	# # sigma_noise = torch.randn(1, 1, noisyframe.shape[2], noisyframe.shape[3])
	# # inp = (noisyframe, sigma_noise)
	# macs = profile_macs(model, noisyframe)
	# macs = clever_format(macs, "%.3f")
	# print(macs)

	# ###################
	# https://github.com/sovrasov/flops-counter.pytorch
	from ptflops import get_model_complexity_info
	noisyframe = (3*TEST_NUM_INPUT_FRAMES, TEST_SHAPE[0], TEST_SHAPE[1])
	macs, params = get_model_complexity_info(model, noisyframe, as_strings=True,
								print_per_layer_stat=True, verbose=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))