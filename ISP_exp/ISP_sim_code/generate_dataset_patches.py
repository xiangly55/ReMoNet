"""
Modified on https://github.com/12dmodel/camera_sim
"""

import sys
import os
sys.path.append('/data1/xiangliuyu/fastdvdnet_data/')

import tifffile
import cv2
import skimage
import numpy as np
from PIL import Image

import argparse
import glob
import json
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches_2d

from torchvision import transforms
from torchvision.transforms import functional as F
import torch
from torch.autograd import Variable
from torch import FloatTensor

from pipeline import ImageDegradationPipeline
from constants import XYZ2sRGB, ProPhotoRGB2XYZ
import time
import h5py

def format_opencv2numpy(img):
	# BGR TO RGB
	# 255 -> [0,1] float
	img = img[:,:,::-1]
	img = img.astype('double') / 255.0
	return img

def format_numpy2opencv(img):
	# convert RGB to BGR
	# convert float 0-1 to 0-255
	img = img[:,:,::-1]
	img = np.clip(img*255.0, 0, 255).astype(np.uint8)
	return img

def float2uint(img):
	return np.clip(img*255.0, 0, 255).astype(np.uint8)

def numpy2tensor(arr):
	if len(arr.shape) < 3:
		arr = np.expand_dims(arr, -1)
	return FloatTensor(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def tensor2numpy(t, idx=None):
	t = torch.clamp(t, 0, 1)
	if idx is None:
		t = t[0, ...]
	else:
		t = t[idx, ...]
	return t.permute(1, 2, 0).cpu().squeeze().numpy()


class ISPCamSim():
	def __init__(self, max_gaussian_noise=0.1, min_gaussian_noise=0, \
					max_poisson_noise=0.02, min_poisson_noise=0, max_exposure=0, min_exposure=0, dwn_factor=1):
		self.max_gaussian_noise = max_gaussian_noise
		self.min_gaussian_noise = min_gaussian_noise
		self.max_poisson_noise = max_poisson_noise
		self.min_poisson_noise = min_poisson_noise
		self.max_exposure = max_exposure
		self.min_exposure = min_exposure
		self.dwn_factor = dwn_factor

		# Define pipeline
		self.poisson_k = np.random.uniform(self.min_poisson_noise, self.max_poisson_noise)
		self.read_noise_sigma = np.random.uniform(self.min_gaussian_noise, self.max_gaussian_noise)
		# TODO: random sample in the log space

		dwn_factor = self.dwn_factor
		self.exp_adjustment = np.random.uniform(self.min_exposure, self.max_exposure)
		configs_prepreprocess = [
					# ('UndoProPhotoRGBGamma', {}),
					# Convert to sRGB
					# ('ColorSpaceConversionMatrix', {'matrix': torch.matmul(XYZ2sRGB, ProPhotoRGB2XYZ)}),
					# MIT-Adobe 5k is in the ProPhoto-RGB Space
					('UndosRGBGamma', {}),
			]

		configs_preprocess = [
			# Blur and downsample to reduce noise
			# ('GaussianBlur', {'sigma_x': dwn_factor}),
			('PytorchResizing', {'resizing_factor': 1.0/dwn_factor, 'mode': 'nearest'})
		]
		configs_degrade = [
			('ExposureAdjustment', {'nstops': self.exp_adjustment}),
			# ('MotionBlur', {'amt': [3, 2], 'direction': [0, 45,]}),
			('BayerMosaicking', {}),
			# Add artificial noise.
			('PoissonNoise',{'sigma': FloatTensor([self.poisson_k] * 3), 'mosaick_pattern': 'bayer'}),
			('GaussianNoise',{'sigma': FloatTensor([self.read_noise_sigma] * 3), 'mosaick_pattern': 'bayer'}),
			('PixelClip', {}),
			('ExposureAdjustment', {'nstops': -self.exp_adjustment}),
			('PixelClip', {}),
			('AHDDemosaickingNonDifferentiable', {}),
			('PixelClip', {}),
		]
		configs_nondegrade = [
			('ExposureAdjustment', {'nstops': self.exp_adjustment}),
			# ('MotionBlur', {'amt': [3, 2], 'direction': [0, 45,]}),
			('BayerMosaicking', {}),
			# # Add artificial noise.
			# ('PoissonNoise',{'sigma': FloatTensor([poisson_k] * 3), 'mosaick_pattern': 'bayer'}),
			# ('GaussianNoise',{'sigma': FloatTensor([read_noise_sigma] * 3), 'mosaick_pattern': 'bayer'}),
			('PixelClip', {}),
			('ExposureAdjustment', {'nstops': -self.exp_adjustment}),
			('PixelClip', {}),
			('AHDDemosaickingNonDifferentiable', {}),
			('PixelClip', {}),
		]
		configs_gamma = [
				# ('DenoisingBilateral',{'sigma_s': 1.0, 'sigma_r': 0.1}),
				('PixelClip', {}),
				('sRGBGamma', {}),
			]

		self.pipeline_prepreprocess = ImageDegradationPipeline(configs_prepreprocess)
		self.pipeline_preprocess = ImageDegradationPipeline(configs_preprocess)
		self.pipeline_degrade = ImageDegradationPipeline(configs_degrade)
		self.pipeline_nondegrade = ImageDegradationPipeline(configs_nondegrade)
		self.pipeline_gamma = ImageDegradationPipeline(configs_gamma)

	def random_noise_level(self):
		epsilon = 1e-10
		min_log = np.log(self.min_poisson_noise + epsilon)
		self.poisson_k = np.exp( min_log + np.random.rand(1)*(np.log([self.max_poisson_noise]) - min_log))
		min_log = np.log(self.min_gaussian_noise + epsilon)
		self.read_noise_sigma = np.exp( min_log + np.random.rand(1)*(np.log([self.max_gaussian_noise]) - min_log))

		configs_degrade = [
			('ExposureAdjustment', {'nstops': self.exp_adjustment}),
			# ('MotionBlur', {'amt': [3, 2], 'direction': [0, 45,]}),
			('BayerMosaicking', {}),
			# Add artificial noise.
			('PoissonNoise',{'sigma': FloatTensor([self.poisson_k] * 3), 'mosaick_pattern': 'bayer'}),
			('GaussianNoise',{'sigma': FloatTensor([self.read_noise_sigma] * 3), 'mosaick_pattern': 'bayer'}),
			('PixelClip', {}),
			('ExposureAdjustment', {'nstops': -self.exp_adjustment}),
			('PixelClip', {}),
			('AHDDemosaickingNonDifferentiable', {}),
			('PixelClip', {}),
		]
		self.pipeline_degrade = ImageDegradationPipeline(configs_degrade)

		return self.poisson_k, self.read_noise_sigma

	def add_noise(self, img):
		# raw_im = cv2.imread(path).astype('float32') / 255.0
		# raw_im = raw_im[:, :, ::-1].copy()
		# raw_im = FloatTensor(raw_im).permute(2, 0, 1).unsqueeze(0)
		t1 = time.time()
		demosaicked = self.pipeline_prepreprocess(img)
		t2 = time.time()
		# preprocessed = self.pipeline_preprocess(demosaicked)
		preprocessed = demosaicked 
		degraded = self.pipeline_degrade(preprocessed)
		t3 = time.time()
		nondegraded = self.pipeline_nondegrade(preprocessed)
		
		t4 = time.time()
		noisy = self.pipeline_gamma(degraded)
		t5 = time.time()
		gt = self.pipeline_gamma(nondegraded)
		# gt = img
		t6 = time.time()
		# print('time', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
		return gt, noisy



class VideoRandomCropAddNoise(transforms.RandomCrop):
	def __init__(self, size, max_gaussian_noise, min_gaussian_noise, max_poisson_noise, min_poisson_noise, 
						max_exposure, min_exposure, dwn_factor):
		super(VideoRandomCropAddNoise, self).__init__(size)
		self.isp = ISPCamSim(max_gaussian_noise, min_gaussian_noise, max_poisson_noise, min_poisson_noise, \
					max_exposure, min_exposure, dwn_factor)
	
	
	def random_noise_level(self):
		poisson_k, read_noise_sigma = self.isp.random_noise_level()
		return poisson_k, read_noise_sigma


	def __call__(self, seq):
		i, j, h, w = self.get_params(seq[0], self.size)
		seq_cropped = []
		seq_cropped_noisy = []
		for img in seq:
			img_cropped = F.crop(img, i, j, h, w)

			img_cropped_tensor = transforms.ToTensor()(img_cropped)
			img_cropped_gt, img_cropped_noisy = self.isp.add_noise(img_cropped_tensor)
			seq_cropped.append(img_cropped_gt)
			seq_cropped_noisy.append(img_cropped_noisy)


		seq_tensor = torch.cat(seq_cropped, dim=0)
		seq_noisy_tensor = torch.cat(seq_cropped_noisy, dim=0)
		return seq_tensor, seq_noisy_tensor





if __name__ == "__main__":
		

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--im_folder', help='path to input images', default='/data1/xiangliuyu/fastdvdnet_data/frame_clean/')
	parser.add_argument('--out_dir', help='path to place output', default='/data1/xiangliuyu/fastdvdnet_data/')
	parser.add_argument('--total_patch', type=int, help='total number of patches to generate', default=256000)

	parser.add_argument('--patch_per_image', type=int, default=50, help='Number of patch to generate from a single degradation of an image')
	parser.add_argument('--patch_sz', type=int, default=96, help='Patch size (square patch for now)')
	parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length')
	parser.add_argument('--fraction_train', type=float, default=1.0, help='Fraction of images to use as training')

	parser.add_argument('--input_ext', default='png', help='path to place output')
	parser.add_argument('--max_exposure', type=float, default=0.0, help='maximum exposure adjustment in stops')
	parser.add_argument('--min_exposure', type=float, default=0.0, help='minimum exposure adjustment in stops')
	parser.add_argument('--max_gaussian_noise', type=float, default=0.1, help='maximum gaussian noise std (on range 0 - 1)')
	parser.add_argument('--min_gaussian_noise', type=float, default=0, help='minimum gaussian noise std (on range 0 - 1)')
	parser.add_argument('--max_poisson_noise', type=float, default=0.02, help='maximum poisson noise mult (See image_processing.PoissonNoise for detail)')
	parser.add_argument('--min_poisson_noise', type=float, default=0, help='minimum poisson noise mult (See image_processing.PoissonNoise for detail)')
	parser.add_argument('--skip_degraded', action="store_true", help='Whether to skip degraded images.')
	parser.add_argument('--dwn_factor', type=float, default=1, help='Factor to downsample.')
	args = parser.parse_args()


	# im_names = glob.glob(os.path.join(args.im_folder, '*.' + args.input_ext))
	
	# im_names = sorted([os.path.basename(i) for i in im_names])
	video_folder_list = os.listdir(args.im_folder)
	video_num = len(video_folder_list)
	video_frame_num_list = np.zeros(video_num)
	for i in range(video_num):
		video_full_path = os.path.join(args.im_folder, video_folder_list[i])
		video_frame_num_list[i] = len(os.listdir(video_full_path))


	n_count = 0
	img_idx = 0

	out_path = os.path.join(args.out_dir, 'train.h5')
	f = h5py.File(out_path, 'w')

	transform = VideoRandomCropAddNoise(args.patch_sz, args.max_gaussian_noise, args.min_gaussian_noise, args.max_poisson_noise, args.min_poisson_noise, \
					args.max_exposure, args.min_exposure, args.dwn_factor)

	progress_bar = tqdm(total=args.total_patch)
	
	gt_all_patches = []
	noisy_all_patches = []
	noise_level = []
	while n_count < args.total_patch:
		# if img_idx < args.fraction_train * len(im_names):
		#     base_dir = train_dir
		# else:
		#     base_dir = test_dir

		# target_dir = os.path.join(base_dir, 'images', 'target')
		# degraded_dir = os.path.join(base_dir, 'images', 'degraded')
		# meta_dir = os.path.join(base_dir, 'meta')
		# if img_idx >= len(im_names):
		#     sample_idx = np.random.choice(range(len(im_names)))
		# else:
		#     sample_idx = img_idx

		# random sample
		poisson_k, read_noise_sigma = transform.random_noise_level()
		
		video_id = np.random.choice(list(range(video_num)))
		# make sure to have enough adjacent frames
		frame_id = np.random.choice(list( range(int(video_frame_num_list[video_id]) - args.sequence_length )))
		video_path = video_folder_list[video_id]

		seq = []
		seq_noisy = []
		
		for i in range(args.sequence_length):
			frame_path = 'frame_{}.png'.format(str(frame_id + i))
			full_path = os.path.join(args.im_folder, video_path, frame_path)
			# img, _, _ = open_image(full_path,\
			# 								gray_mode=self.gray_mode,\
			# 								expand_if_needed=False,\
			# 								expand_axis0=False)
			img = Image.open(full_path)
			seq.append(img)
		
		seq_gt, seq_noisy = transform(seq)
		# tensor, [5, 3, 96, 96]


		noisy_all_patches.append(float2uint(seq_noisy.numpy()))
		gt_all_patches.append(float2uint(seq_gt.numpy()))
		noise_level.append(np.array([poisson_k, read_noise_sigma]))
		n_count += 1
		progress_bar.update(1)

	noisy_all_patches_np = np.stack(noisy_all_patches[i] for i in range(len(noisy_all_patches)))
	gt_all_patches_np = np.stack(gt_all_patches[i] for i in range(len(gt_all_patches)))
	noise_level_np = np.stack(noise_level[i] for i in range(len(noise_level)))
	noise_level_np = noise_level_np.squeeze()

	print('Creating datasets ...')
	f.create_dataset('noisy_patches',shape=(args.total_patch, args.sequence_length*3, 3, args.patch_sz, args.patch_sz), \
								dtype=np.uint8, data=noisy_all_patches_np, compression="gzip",compression_opts=9)
	f.create_dataset('gt_patches',shape=(args.total_patch, args.sequence_length, 3, args.patch_sz,args.patch_sz), \
								dtype=np.uint8, data=gt_all_patches_np, compression="gzip",compression_opts=9)
	f.create_dataset('noise_level',shape=(args.total_patch, 2), dtype=np.float, data=noise_level_np, compression="gzip", compression_opts=9)

	f.close()
	progress_bar.close()