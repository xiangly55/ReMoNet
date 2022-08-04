import torch
import torch.nn.functional as F

def temp_denoise(model, noisyframe, sigma_noise):
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
	out = torch.clamp(model(noisyframe), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_ReMoNet(seq, noise_std, temp_psz, temp_osz, model_temporal):

	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	noise_map = noise_std.expand((1, 1, H, W))

	ctrl_out_fr_idx = temp_osz // 2

	for fridx in range(ctrl_out_fr_idx, numframes, temp_osz):
		print('denoising frame {} - {}'.format(fridx-temp_osz//2, fridx+temp_osz//2))

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

	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	return denframes

