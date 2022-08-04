import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models_ReMoNet import ReMoNet, Similarity
from dataset import ValDataset
from dataloaders import train_dali_loader
# from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from utils import *
from train_common_ReMoNet import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint




def split_channel(img, ch=1):
	# (64,9,96,96)
	img_split = img.split(3, dim=1)
	img_split = torch.stack(img_split, dim=ch)
	return img_split



def main(**args):
	r"""Performs the main training loop
	"""

	# Load dataset
	print('> Loading datasets ...')
	dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
	loader_train = train_dali_loader(batch_size=args['batch_size'],\
									file_root=args['trainset_dir'],\
									sequence_length=args['temp_patch_size'],\
									crop_size=args['patch_size'],\
									epoch_size=args['max_number_patches'],\
									random_shuffle=True,\
									temp_stride=3)

	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)
	log_running_scripts(args['log_dir'], __file__)

	# Define GPU devices
	# device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization
	

	# Create model
	# model = ReMoNet(num_input_frames=args['temp_patch_size'], num_output_frames=args['temp_out_size'])
	model_1 = ReMoNet()
	model_2 = ReMoNet()
	# model = model.cuda()
	model_1 = nn.DataParallel(model_1).cuda()
	model_2 = nn.DataParallel(model_2).cuda()

	# Define loss
	distill_criterion = Similarity()
	distill_criterion.cuda()
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()


	# Optimizer
	optimizer_1 = optim.Adam(model_1.parameters(), lr=args['lr'])
	optimizer_2 = optim.Adam(model_2.parameters(), lr=args['lr'])

	if args['lr_schedule'] == 'CosineAnnealingLR':
		scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_1, args['epochs'])
		scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_2, args['epochs'])
	elif args['lr_schedule'] == 'MultiStepLR':
		scheduler_1 = optim.lr_scheduler.MultiStepLR(optimizer_1, gamma=0.1, milestones=args['milestone'])
		scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer_2, gamma=0.1, milestones=args['milestone'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model_1, optimizer_1)
	_, _ = resume_training(args, model_2, optimizer_2)

	# Training
	start_time = time.time()
	
	# freeze temp module, train spatial module first

	for epoch in range(start_epoch, args['epochs']):

		# train
		scheduler_1.step()
		scheduler_2.step()
		current_lr = scheduler_1.get_lr()
		
		for i, data in enumerate(loader_train, 0):

			# Pre-training step
			model_1.train()
			model_2.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer_1.zero_grad()
			optimizer_2.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			# img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
			img_train, gt_train = normalize_augment_multiOut(data[0]['data'], ctrl_fr_idx, args['temp_out_size'])
			
			N, _, H, W = img_train.size()

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
			# draw noise samples from std dev tensor
			noise = torch.zeros_like(img_train)
			noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

			#define noisy input
			imgn_train = img_train + noise

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise = noise.cuda(non_blocking=True)
			noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

			imgn_train, gt_train = split_channel(imgn_train), split_channel(gt_train)
			# Evaluate model and optimize it
			out_train_1 = model_1(imgn_train)
			out_train_2 = model_2(imgn_train)
			# Compute loss
			loss_1 = criterion(gt_train, out_train_1) / (N*2)
			loss_2 = criterion(gt_train, out_train_2) / (N*2)
			loss_mutual = distill_criterion(out_train_1, out_train_2)

			loss = loss_1 + loss_2 + loss_mutual
			loss.backward()
			optimizer_1.step()
			optimizer_2.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:


				# Compute training PSNR
				log_train_psnr(out_train_1, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		model_1.eval()

		# Validation and log images
		validate_and_log(
						model_temp=model_1, \
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						temp_osz=args['temp_out_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
						trainimg=img_train
						)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model_1, args, optimizer_1, training_params, epoch)

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=12, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=60, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[15, 30], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr_schedule", type=str, default='CosineAnnealingLR', \
						help="lr schedule")
	parser.add_argument("--lr", type=float, default=8e-4, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_out_size", "--to", type=int, default=5, help="Temporal output size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs/logs_test", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default="./data/mp4/", \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default="./data/derf_540p_seqs", \
						 help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training ReMoNet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
