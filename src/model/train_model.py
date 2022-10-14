import torch
import numpy as np
import tqdm
import os
import sacred
import model.util as util
import feature.graph_conversions as graph_conversions

MODEL_DIR = os.environ.get(
	"MODEL_DIR",
	"/gstore/data/resbioai/tsenga5/branched_diffusion/models/trained_models/misc"
)

train_ex = sacred.Experiment("train")

train_ex.observers.append(
	sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


@train_ex.config
def config():
	# Number of training epochs
	num_epochs = 30

	# Learning rate
	learning_rate = 0.001


@train_ex.command
def train_model(
	model, diffuser, data_loader, num_epochs, learning_rate, _run, t_limit=1000
):
	"""
	Trains a diffusion model using the given instantiated model and discrete
	diffuser object.
	Arguments:
		`model`: an instantiated model which takes in x, t and predicts a
			posterior
		`diffuser`: a DiscreteDiffuser object
		`data_loader`: a DataLoader object that yields batches of data as
			tensors in pairs: x, y
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
		`t_limit`: training will occur between time 1 and `t_limit`
	"""
	run_num = _run._id
	output_dir = os.path.join(MODEL_DIR, str(run_num))

	model.train()
	torch.set_grad_enabled(True)
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch_num in range(num_epochs):
		batch_losses = []
		t_iter = tqdm.tqdm(data_loader)
		for x0, y in t_iter:
			x0 = x0.to(DEVICE).float()
			
			# Sample random times between 1 and t_limit (inclusive)
			t = torch.randint(
				t_limit, size=(x0.shape[0],), device=DEVICE
			) + 1

			# Run diffusion forward to get xt and the posterior parameter to
			# predict
			xt, true_post = diffuser.forward(x0, t)
			
			# Get model-predicted posterior parameter
			pred_post = model(xt, t)
			
			# Compute loss
			loss = model.loss(pred_post, true_post)
			loss_val = loss.item()
			t_iter.set_description("Loss: %.4f" % loss_val)

			if np.isnan(loss_val):
				continue
			
			optim.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optim.step()
			
			batch_losses.append(loss_val)
		
		epoch_loss = np.mean(batch_losses)
		print("Epoch %d average Loss: %.4f" % (epoch_num + 1, epoch_loss))

		_run.log_scalar("train_epoch_loss", epoch_loss)
		_run.log_scalar("train_batch_losses", batch_losses)

		model_path = os.path.join(
			output_dir, "epoch_%d_ckpt.pth" % (epoch_num + 1)
		)
		link_path = os.path.join(output_dir, "last_ckpt.pth")
		
		# Save model
		util.save_model(model, model_path)

		# Create symlink to last epoch
		if os.path.islink(link_path):
			os.remove(link_path)
		os.symlink(os.path.basename(model_path), link_path)


@train_ex.command
def train_graph_model(
	model, diffuser, data_loader, num_epochs, learning_rate, _run, t_limit=1000
):
	"""
	Trains a diffusion model on graphs using the given instantiated model and
	discrete diffuser object.
	Arguments:
		`model`: an instantiated model which takes in x, t and predicts a
			posterior on edges in canonical order
		`diffuser`: a DiscreteDiffuser object
		`data_loader`: a DataLoader object that yields torch-geometric Data
			objects
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
		`t_limit`: training will occur between time 1 and `t_limit`
	"""
	run_num = _run._id
	output_dir = os.path.join(MODEL_DIR, str(run_num))

	model.train()
	torch.set_grad_enabled(True)
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch_num in range(num_epochs):
		batch_losses = []
		t_iter = tqdm.tqdm(data_loader)
		for data in t_iter:

			e0, edge_batch_inds = graph_conversions.pyg_data_to_edge_vector(
				data, return_batch_inds=True
			)  # Shape: E
			
			# Pick some random times t between 1 and t_limit (inclusive), one
			# value for each individual graph
			graph_sizes = torch.diff(data.ptr)
			graph_times = torch.randint(
				t_limit, size=(graph_sizes.shape[0],), device=DEVICE
			) + 1
			
			# Tile the graph times to the size of all nodes
			t_v = graph_times[data.batch].float()  # Shape: V
			t_e = graph_times[edge_batch_inds].float()	# Shape: E

			# Add noise to edges from time 0 to time t
			et, true_post = diffuser.forward(e0[:, None], t_e)
			# Do the noising on E x 1 tensors
			et, true_post = et[:, 0], true_post[:, 0]
			data.edge_index = graph_conversions.edge_vector_to_pyg_data(
				data, et
			)
			# Note: this modifies `data`

			# Get model-predicted posterior parameter
			pred_post = model(data, t_v)
			
			# Compute loss
			loss = model.loss(pred_post, true_post)
			loss_val = loss.item()
			t_iter.set_description("Loss: %.4f" % loss_val)

			if np.isnan(loss_val):
				continue
			
			optim.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optim.step()
			
			batch_losses.append(loss_val)
		
		epoch_loss = np.mean(batch_losses)
		print("Epoch %d average Loss: %.4f" % (epoch_num + 1, epoch_loss))

		_run.log_scalar("train_epoch_loss", epoch_loss)
		_run.log_scalar("train_batch_losses", batch_losses)

		model_path = os.path.join(
			output_dir, "epoch_%d_ckpt.pth" % (epoch_num + 1)
		)
		link_path = os.path.join(output_dir, "last_ckpt.pth")
		
		# Save model
		util.save_model(model, model_path)

		# Create symlink to last epoch
		if os.path.islink(link_path):
			os.remove(link_path)
		os.symlink(os.path.basename(model_path), link_path)
