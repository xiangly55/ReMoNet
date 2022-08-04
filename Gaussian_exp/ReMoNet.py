import torch
import torch.nn.functional as F

def temp_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	# No noise map here
	# out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)
	out = torch.clamp(model(noisyframe), 0., 1.)
	if len(out.shape) == 5:
		out = out.view(out.size(0), -1, out.size(3), out.size(4))

	out = out.view(-1, 3, out.size(2), out.size(3))
	
	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_ReMoNet(seq, noise_std, temp_psz, temp_osz, model_temporal):
	r"""Denoises a sequence of frames with ReMoNet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		temp_osz: size of output temporal size
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	ctrl_out_fr_idx = temp_osz // 2

	for fridx in range(ctrl_out_fr_idx, numframes, temp_osz):
		print('denoising frame {} - {}'.format(fridx-temp_osz//2, fridx+temp_osz//2))
		# load input frames
		# if not inframes:
		# # if list not yet created, fill it with temp_patchsz frames
		# 	for idx in range(temp_psz):
		# 		# use [2,1,0,1,2] frames to denoise the 0-th frame
		# 		relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
		# 		inframes.append(seq[relidx])
		# else:
		# 	# delete left-most frame
		# 	del inframes[0]
		# 	relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
		# 	# add right-most frame
		# 	inframes.append(seq[relidx])
		inframes = []
		for idx in range(temp_psz):
			relidx = abs(fridx - ctrlfr_idx + idx)
			relidx = min(relidx, 2*numframes-1-relidx)
			inframes.append(seq[relidx])

		# stack 5 consecutive frames
		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		end_idx = min(fridx+ctrl_out_fr_idx+1, denframes.shape[0])
		length = end_idx - (fridx-ctrl_out_fr_idx)
		denframes[fridx-ctrl_out_fr_idx:end_idx] = temp_denoise(model_temporal, inframes_t, noise_map)[:length]

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

