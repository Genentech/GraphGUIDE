import torch
import numpy as np
from model.util import sanitize_sacred_arguments

class MNISTProbUNetTimeConcat(torch.nn.Module):

	def __init__(
		self, t_limit, enc_dims=[32, 64, 128, 256], dec_dims=[32, 64, 128],
		time_embed_size=32, data_channels=1
	):
		"""
		Initialize a time-dependent U-net for MNIST, where time embeddings are
		concatenated to image representations.
		Arguments:
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage)
			`time_embed_size`: size of the time embeddings
			`data_channels`: number of channels in input image
		"""
		super().__init__()

		assert len(enc_dims) == 4
		assert len(dec_dims) == 3

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.t_limit = t_limit

		# Encoders: receptive field increases and depth increases
		self.conv_e1 = torch.nn.Conv2d(
			data_channels + time_embed_size, enc_dims[0], kernel_size=3,
			stride=1, bias=False
		)
		self.time_dense_e1 = torch.nn.Linear(2, time_embed_size)
		self.norm_e1 = torch.nn.GroupNorm(4, num_channels=enc_dims[0])
		self.conv_e2 = torch.nn.Conv2d(
			enc_dims[0] + time_embed_size, enc_dims[1], kernel_size=3, stride=2,
			bias=False
		)
		self.time_dense_e2 = torch.nn.Linear(2, time_embed_size)
		self.norm_e2 = torch.nn.GroupNorm(32, num_channels=enc_dims[1])
		self.conv_e3 = torch.nn.Conv2d(
			enc_dims[1] + time_embed_size, enc_dims[2], kernel_size=3, stride=2,
			bias=False
		)
		self.time_dense_e3 = torch.nn.Linear(2, time_embed_size)
		self.norm_e3 = torch.nn.GroupNorm(32, num_channels=enc_dims[2])
		self.conv_e4 = torch.nn.Conv2d(
			enc_dims[2] + time_embed_size, enc_dims[3], kernel_size=3, stride=2,
			bias=False
		)
		self.time_dense_e4 = torch.nn.Linear(2, time_embed_size)
		self.norm_e4 = torch.nn.GroupNorm(32, num_channels=enc_dims[3])

		# Decoders: depth decreases		   
		self.conv_d4 = torch.nn.ConvTranspose2d(
			enc_dims[3] + time_embed_size, dec_dims[2], 3, stride=2, bias=False
		)
		self.time_dense_d4 = torch.nn.Linear(2, time_embed_size)
		self.norm_d4 = torch.nn.GroupNorm(32, num_channels=dec_dims[2])
		self.conv_d3 = torch.nn.ConvTranspose2d(
			dec_dims[2] + enc_dims[2] + time_embed_size, dec_dims[1], 3,
			stride=2, output_padding=1, bias=False
		)
		self.time_dense_d3 = torch.nn.Linear(2, time_embed_size)
		self.norm_d3 = torch.nn.GroupNorm(32, num_channels=dec_dims[1])
		self.conv_d2 = torch.nn.ConvTranspose2d(
			dec_dims[1] + enc_dims[1] + time_embed_size, dec_dims[0], 3,
			stride=2, output_padding=1, bias=False
		)
		self.time_dense_d2 = torch.nn.Linear(2, time_embed_size)
		self.norm_d2 = torch.nn.GroupNorm(32, num_channels=dec_dims[0])
		self.conv_d1 = torch.nn.ConvTranspose2d(
			dec_dims[0] + enc_dims[0], data_channels, 3, stride=1, bias=True
		)

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

		# Loss
		self.bce_loss = torch.nn.BCELoss()

	def forward(self, xt, t):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x 1 x H x W tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				image
		Returns a B x 1 x H x W tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We embed the time as cos((t/T) * (2pi)) and sin((t/T) * (2pi))
		time_embed_args = (t[:, None] / self.t_limit) * (2 * np.pi)
		# Shape: B x 1
		time_embed = self.swish(
			torch.cat([
				torch.sin(time_embed_args), torch.cos(time_embed_args)
			], dim=1)
		)
		# Shape: B x 2

		# Encoding
		enc_1_out = self.swish(self.norm_e1(self.conv_e1(
			torch.cat([
				xt,
				torch.tile(
					self.time_dense_e1(time_embed)[:, :, None, None],
					(1, 1) + xt.shape[2:]
				)
			], dim=1)
		)))
		enc_2_out = self.swish(self.norm_e2(self.conv_e2(
			torch.cat([
				enc_1_out,
				torch.tile(
					self.time_dense_e2(time_embed)[:, :, None, None],
					(1, 1) + enc_1_out.shape[2:]
				)
			], dim=1)
		)))
		enc_3_out = self.swish(self.norm_e3(self.conv_e3(
			torch.cat([
				enc_2_out,
				torch.tile(
					self.time_dense_e3(time_embed)[:, :, None, None],
					(1, 1) + enc_2_out.shape[2:]
				)
			], dim=1)
		)))
		enc_4_out = self.swish(self.norm_e4(self.conv_e4(
			torch.cat([
				enc_3_out,
				torch.tile(
					self.time_dense_e4(time_embed)[:, :, None, None],
					(1, 1) + enc_3_out.shape[2:]
				)
			], dim=1)
		)))

		# Decoding
		dec_4_out = self.swish(self.norm_d4(self.conv_d4(
			torch.cat([
				enc_4_out,
				torch.tile(
					self.time_dense_d4(time_embed)[:, :, None, None],
					(1, 1) + enc_4_out.shape[2:]
				)
			], dim=1)
		)))
		dec_3_out = self.swish(self.norm_d3(self.conv_d3(
			torch.cat([
				dec_4_out, enc_3_out,
				torch.tile(
					self.time_dense_d3(time_embed)[:, :, None, None],
					(1, 1) + dec_4_out.shape[2:]
				)
			], dim=1)
		)))
		dec_2_out = self.swish(self.norm_d2(self.conv_d2(
			torch.cat([
				dec_3_out, enc_2_out,
				torch.tile(
					self.time_dense_d2(time_embed)[:, :, None, None],
					(1, 1) + dec_3_out.shape[2:]
				)
			], dim=1)
		)))
		dec_1_out = self.conv_d1(torch.cat([dec_2_out, enc_1_out], dim=1))
		return torch.sigmoid(dec_1_out)
	
	def loss(self, pred_probs, true_probs):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_probs`: a B x 1 x H x W tensor of predicted probabilities
			`true_probs`: a B x 1 x H x W tensor of true probabilities
		Returns a scalar loss of binary cross-entropy values, averaged across
		all dimensions.
		"""
		return self.bce_loss(pred_probs, true_probs)


class MNISTProbUNetTimeAdd(torch.nn.Module):

	def __init__(
		self, t_limit, enc_dims=[32, 64, 128, 256], dec_dims=[32, 64, 128],
		time_embed_size=32, time_embed_std=30, use_time_embed_dense=False,
		data_channels=1
	):
		"""
		Initialize a time-dependent U-net for MNIST, where time embeddings are
		added to image representations.
		Arguments:
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage)
			`time_embed_size`: size of the time embeddings
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`use_time_embed_dense`: if True, have a dense layer on top of time
				embeddings initially
			`data_channels`: number of channels in input image
		"""
		super().__init__()

		assert len(enc_dims) == 4
		assert len(dec_dims) == 3
		assert time_embed_size % 2 == 0

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.t_limit = t_limit
		self.use_time_embed_dense = use_time_embed_dense

		# Random embedding layer for time; the random weights are set at the
		# start and are not trainable
		self.time_embed_rand_weights = torch.nn.Parameter(
			torch.randn(time_embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		if use_time_embed_dense:
			self.time_embed_dense = torch.nn.Linear(
				time_embed_size, time_embed_size
			)

		# Encoders: receptive field increases and depth increases
		self.conv_e1 = torch.nn.Conv2d(
			data_channels, enc_dims[0], kernel_size=3, stride=1, bias=False
		)
		self.time_dense_e1 = torch.nn.Linear(time_embed_size, enc_dims[0])
		self.norm_e1 = torch.nn.GroupNorm(4, num_channels=enc_dims[0])
		self.conv_e2 = torch.nn.Conv2d(
			enc_dims[0], enc_dims[1], kernel_size=3, stride=2, bias=False
		)
		self.time_dense_e2 = torch.nn.Linear(time_embed_size, enc_dims[1])
		self.norm_e2 = torch.nn.GroupNorm(32, num_channels=enc_dims[1])
		self.conv_e3 = torch.nn.Conv2d(
			enc_dims[1], enc_dims[2], kernel_size=3, stride=2, bias=False
		)
		self.time_dense_e3 = torch.nn.Linear(time_embed_size, enc_dims[2])
		self.norm_e3 = torch.nn.GroupNorm(32, num_channels=enc_dims[2])
		self.conv_e4 = torch.nn.Conv2d(
			enc_dims[2], enc_dims[3], kernel_size=3, stride=2, bias=False
		)
		self.time_dense_e4 = torch.nn.Linear(time_embed_size, enc_dims[3])
		self.norm_e4 = torch.nn.GroupNorm(32, num_channels=enc_dims[3])

		# Decoders: depth decreases		   
		self.conv_d4 = torch.nn.ConvTranspose2d(
			enc_dims[3], dec_dims[2], 3, stride=2, bias=False
		)
		self.time_dense_d4 = torch.nn.Linear(time_embed_size, dec_dims[2])
		self.norm_d4 = torch.nn.GroupNorm(32, num_channels=dec_dims[2])
		self.conv_d3 = torch.nn.ConvTranspose2d(
			dec_dims[2] + enc_dims[2], dec_dims[1], 3, stride=2,
			output_padding=1, bias=False
		)
		self.time_dense_d3 = torch.nn.Linear(time_embed_size, dec_dims[1])
		self.norm_d3 = torch.nn.GroupNorm(32, num_channels=dec_dims[1])
		self.conv_d2 = torch.nn.ConvTranspose2d(
			dec_dims[1] + enc_dims[1], dec_dims[0], 3, stride=2,
			output_padding=1, bias=False
		)
		self.time_dense_d2 = torch.nn.Linear(time_embed_size, dec_dims[0])
		self.norm_d2 = torch.nn.GroupNorm(32, num_channels=dec_dims[0])
		self.conv_d1 = torch.nn.ConvTranspose2d(
			dec_dims[0] + enc_dims[0], data_channels, 3, stride=1, bias=True
		)

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

		# Loss
		self.bce_loss = torch.nn.BCELoss()

	def forward(self, xt, t):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x 1 x H x W tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				image
		Returns a B x 1 x H x W tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We had sampled a vector z from a zero-mean Gaussian of fixed variance
		# We embed the time as cos((t/T) * (2pi) * z) and sin((t/T) * (2pi) * z)
		time_embed_args = (t[:, None] / self.t_limit) * \
			self.time_embed_rand_weights[None, :] * (2 * np.pi)
		# Shape: B x (E / 2)

		time_embed = torch.cat([
			torch.sin(time_embed_args), torch.cos(time_embed_args)
		], dim=1)
		if self.use_time_embed_dense:
			time_embed = self.swish(self.time_embed_dense(time_embed))
		else:
			time_embed = self.swish(time_embed)
		# Shape: B x E
	
		# Encoding
		enc_1_out = self.swish(self.norm_e1(
			self.conv_e1(xt) +
			self.time_dense_e1(time_embed)[:, :, None, None]
		))
		enc_2_out = self.swish(self.norm_e2(
			self.conv_e2(enc_1_out) +
			self.time_dense_e2(time_embed)[:, :, None, None]
		))
		enc_3_out = self.swish(self.norm_e3(
			self.conv_e3(enc_2_out) +
			self.time_dense_e3(time_embed)[:, :, None, None]
		))
		enc_4_out = self.swish(self.norm_e4(
			self.conv_e4(enc_3_out) +
			self.time_dense_e4(time_embed)[:, :, None, None]
		))
			
		# Decoding
		dec_4_out = self.swish(self.norm_d4(
			self.conv_d4(enc_4_out) +
			self.time_dense_d4(time_embed)[:, :, None, None]
		))
		dec_3_out = self.swish(self.norm_d3(
			self.conv_d3(torch.cat([dec_4_out, enc_3_out], dim=1)) +
			self.time_dense_d3(time_embed)[:, :, None, None]
		))
		dec_2_out = self.swish(self.norm_d2(
			self.conv_d2(torch.cat([dec_3_out, enc_2_out], dim=1)) +
			self.time_dense_d2(time_embed)[:, :, None, None]
		))
		dec_1_out = self.conv_d1(torch.cat([dec_2_out, enc_1_out], dim=1))
		return torch.sigmoid(dec_1_out)
	
	def loss(self, pred_probs, true_probs):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_probs`: a B x 1 x H x W tensor of predicted probabilities
			`true_probs`: a B x 1 x H x W tensor of true probabilities
		Returns a scalar loss of binary cross-entropy values, averaged across
		all dimensions.
		"""
		return self.bce_loss(pred_probs, true_probs)
