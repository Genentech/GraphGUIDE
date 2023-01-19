import torch
import numpy as np
import tqdm
import os
import sacred
import model.util as util
import feature.graph_conversions as graph_conversions
import model.generate as generate
import analysis.graph_metrics as graph_metrics
import analysis.mmd as mmd


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
	model, diffuser, data_loader, num_epochs, learning_rate, _run, t_limit=1000,
	compute_mmd=False, val_data_loader=None, mmd_sample_size=200
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
		`compute_mmd`: if True, compute some performance metrics at the end of
			training
		`val_data_loader`: if `compute_mmd` is True, this must be another data
			loader (like `data_loader`) which yields validation-set objects
		`mmd_sample_size`: number of graphs to compute MMD over
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

	# If required, compute MMD metrics
	if compute_mmd:
		num_batches = int(np.ceil(mmd_sample_size / data_loader.batch_size))
		train_graphs_1, train_graphs_2 = [], []
		gen_graphs = []
		data_iter_1, data_iter_2 = iter(data_loader), iter(val_data_loader)
		print("Generating %d graphs over %d batches" % (
			mmd_sample_size, num_batches)
		)
		for i in range(num_batches):
			print("Batch %d/%d" % (i + 1, num_batches))
			data = next(data_iter_1)
			train_graphs_1.extend(
				graph_conversions.split_pyg_data_to_nx_graphs(data)
			)
			data = next(data_iter_2)
			train_graphs_2.extend(
				graph_conversions.split_pyg_data_to_nx_graphs(data)
			)
			edges = graph_conversions.pyg_data_to_edge_vector(data)
			sampled_edges = diffuser.sample_prior(
				edges.shape[0], # Samples will be E x 1
				torch.tile(torch.tensor([t_limit], device=DEVICE), edges.shape)
			)[:, 0]  # Shape: E
			data.edge_index = graph_conversions.edge_vector_to_pyg_data(
				data, sampled_edges
			)
		
			samples = generate.generate_graph_samples(
				model, diffuser, data, t_limit=t_limit, verbose=True
			)
			gen_graphs.extend(
				graph_conversions.split_pyg_data_to_nx_graphs(samples)
			)
	
		train_graphs_1 = train_graphs_1[:mmd_sample_size]
		train_graphs_2 = train_graphs_2[:mmd_sample_size]
		gen_graphs = gen_graphs[:mmd_sample_size]
		assert len(train_graphs_1) == mmd_sample_size
		assert len(train_graphs_2) == mmd_sample_size
		assert len(gen_graphs) == mmd_sample_size
		all_graphs = train_graphs_1 + train_graphs_2 + gen_graphs
			
		# Compute MMD values
		print("MMD (squared) values:")
		square_func = np.square
		kernel_type = "gaussian_total_variation"
		
		degree_hists = mmd.make_histograms(
			graph_metrics.get_degrees(all_graphs), bin_width=1
		)
		degree_mmd_1 = square_func(mmd.compute_maximum_mean_discrepancy(
			degree_hists[:mmd_sample_size], degree_hists[-mmd_sample_size:],
			kernel_type, sigma=1
		))
		degree_mmd_2 = square_func(mmd.compute_maximum_mean_discrepancy(
			degree_hists[:mmd_sample_size],
			degree_hists[mmd_sample_size:-mmd_sample_size],
			kernel_type, sigma=1
		))
		_run.log_scalar("degree_mmd", degree_mmd_1)
		_run.log_scalar("degree_mmd_baseline", degree_mmd_2)
		print("Degree MMD ratio: %.8f/%.8f = %.8f" % (
			degree_mmd_1, degree_mmd_2, degree_mmd_1 / degree_mmd_2
		))
		
		cluster_coef_hists = mmd.make_histograms(
			graph_metrics.get_clustering_coefficients(all_graphs), num_bins=100
		)
		cluster_coef_mmd_1 = square_func(mmd.compute_maximum_mean_discrepancy(
			cluster_coef_hists[:mmd_sample_size],
			cluster_coef_hists[-mmd_sample_size:],
			kernel_type, sigma=0.1
		))
		cluster_coef_mmd_2 = square_func(mmd.compute_maximum_mean_discrepancy(
			cluster_coef_hists[:mmd_sample_size],
			cluster_coef_hists[mmd_sample_size:-mmd_sample_size],
			kernel_type, sigma=0.1
		))
		_run.log_scalar("cluster_coef_mmd", cluster_coef_mmd_1)
		_run.log_scalar("cluster_coef_mmd_baseline", cluster_coef_mmd_2)
		print("Clustering coefficient MMD ratio: %.8f/%.8f = %.8f" % (
			cluster_coef_mmd_1, cluster_coef_mmd_2,
			cluster_coef_mmd_1 / cluster_coef_mmd_2
		))
		
		spectra_hists = mmd.make_histograms(
			graph_metrics.get_spectra(all_graphs),
			bin_array=np.linspace(-1e-5, 2, 200 + 1)
		)
		spectra_mmd_1 = square_func(mmd.compute_maximum_mean_discrepancy(
			spectra_hists[:mmd_sample_size], spectra_hists[-mmd_sample_size:],
			kernel_type, sigma=1
		))
		spectra_mmd_2 = square_func(mmd.compute_maximum_mean_discrepancy(
			spectra_hists[:mmd_sample_size],
			spectra_hists[mmd_sample_size:-mmd_sample_size],
			kernel_type, sigma=1
		))
		_run.log_scalar("spectra_mmd", spectra_mmd_1)
		_run.log_scalar("spectra_mmd_baseline", spectra_mmd_2)
		print("Spectrum MMD ratio: %.8f/%.8f = %.8f" % (
			spectra_mmd_1, spectra_mmd_2, spectra_mmd_1 / spectra_mmd_2
		))
		
		orbit_counts = graph_metrics.get_orbit_counts(all_graphs)
		orbit_counts = np.stack([
			np.mean(counts, axis=0) for counts in orbit_counts
		])
		orbit_mmd_1 = square_func(mmd.compute_maximum_mean_discrepancy(
			orbit_counts[:mmd_sample_size], orbit_counts[-mmd_sample_size:],
			kernel_type, normalize=False, sigma=30
		))
		orbit_mmd_2 = square_func(mmd.compute_maximum_mean_discrepancy(
			orbit_counts[:mmd_sample_size],
			orbit_counts[mmd_sample_size:-mmd_sample_size],
			kernel_type, normalize=False, sigma=30
		))
		_run.log_scalar("orbit_mmd", orbit_mmd_1)
		_run.log_scalar("orbit_mmd_baseline", orbit_mmd_2)
		print("Orbit MMD ratio: %.8f/%.8f = %.8f" % (
			orbit_mmd_1, orbit_mmd_2, orbit_mmd_1 / orbit_mmd_2
		))
