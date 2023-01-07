import torch
import tqdm
import feature.graph_conversions as graph_conversions

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def generate_samples(
	model, diffuser, num_samples=64, t_start=0, t_limit=1000,
	initial_samples=None, return_all_times=False, verbose=False
):
	"""
	Generates samples from a trained posterior model and discrete diffuser. This
	first generates a sample from the prior distribution a `t_limit`, then steps
	backward through time to generate new data points.
	Arguments:
		`model`: a trained model which takes in x, t and predicts a posterior
		`diffuser`: a DiscreteDiffuser object
		`num_samples`: number of objects to return
		`t_start`: last time step to stop at (a smaller positive integer) than
			`t_limit`
		`t_limit`: the time step to start generating at (a larger positive
			integer than `t_start`)
		`initial_samples`: if given, this is a tensor which contains the samples
			to start from initially, to be used instead of sampling from the
			diffuser's defined prior
		`return_all_times`: if True, instead of returning a tensor at `t_start`,
			return a larger tensor where the first dimension is
			`t_limit - t_start + 1`, and a tensor of times; each tensor is the
			reconstruction of the object for that time; the first entry will be
			the object at `t_limit`, and the last entry will be the object at
			`t_start`
		`verbose`: if True, print out progress bar
	Returns a tensor of size `num_samples` x .... If `return_all_times` is True,
	returns a tensor of size T x `num_samples` x ... of reconstructions and a
	tensor of size T for times.
	"""
	# First, sample from the prior distribution at some late time t
	if initial_samples is not None:
		xt = initial_samples
		assert len(xt) == num_samples
	else:
		t = (torch.ones(num_samples) * t_limit).to(DEVICE)
		xt = diffuser.sample_prior(num_samples, t)

	if return_all_times:
		all_xt = torch.empty(
			(t_limit - t_start + 1,) + xt.shape,
			dtype=xt.dtype, device=xt.device
		)
		all_xt[0] = xt
		all_t = torch.arange(t_limit, t_start - 1, step=-1).to(DEVICE)

	# Disable gradient computation in model
	model.eval()
	torch.set_grad_enabled(False)
	
	time_steps = torch.arange(t_limit, t_start, step=-1).to(DEVICE)
	# (descending order)
	
	# Step backward through time starting at xt
	x = xt
	t_iter = tqdm.tqdm(enumerate(time_steps), total=len(time_steps)) if verbose \
		else enumerate(time_steps)
	for t_i, time_step in t_iter:
		t = torch.ones(num_samples).to(DEVICE) * time_step
		post = model(xt, t)
		xt = diffuser.reverse_step(xt, t, post)

		if return_all_times:
			all_xt[t_i + 1] = xt

	if return_all_times:
		return all_xt, all_t
	return xt


def generate_graph_samples(
	model, diffuser, initial_samples, t_start=0, t_limit=1000,
	return_all_times=False, verbose=False
):
	"""
	Generates samples from a trained posterior model and discrete diffuser. This
	first generates a sample from the prior distribution a `t_limit`, then steps
	backward through time to generate new data points.
	Arguments:
		`model`: a trained model which takes in x, t and predicts a posterior on
			edges in canonical order
		`diffuser`: a DiscreteDiffuser object
		`initial_samples`: a torch-geometric Data object which contains the
			samples to start from initially, at `t_limit`
		`t_start`: last time step to stop at (a smaller positive integer) than
			`t_limit`
		`t_limit`: the time step to start generating at (a larger positive
			integer than `t_start`)
		`return_all_times`: if True, instead of returning a single
			torch-geometric Data object at `t_start`, return a list of Data
			objects of length `t_limit - t_start + 1`, and parallel tensor of
			times; each Data object reconstruction of the object for that time;
			the first entry will be the object at `t_limit`, and the last entry
			will be the object at `t_start`
		`verbose`: if True, print out progress bar
	Returns a torch-geometric Data object. If `return_all_times` is True,
	returns a list of T torch-geometric Data objects and a T-tensor of times.
	"""
	xt = initial_samples

	if return_all_times:
		all_xt = [xt]
		all_t = torch.arange(t_limit, t_start - 1, step=-1).to(DEVICE)

	# Disable gradient computation in model
	model.eval()
	torch.set_grad_enabled(False)
	
	time_steps = torch.arange(t_limit, t_start, step=-1).to(DEVICE)
	# (descending order)
	
	# Step backward through time starting at xt
	x = xt
	t_iter = tqdm.tqdm(enumerate(time_steps), total=len(time_steps)) if verbose \
		else enumerate(time_steps)
	for t_i, time_step in t_iter:
		edges = graph_conversions.pyg_data_to_edge_vector(xt)

		t_v = torch.tile(
			torch.tensor([time_step], device=DEVICE), (xt.x.shape[0],)
		)  # Shape: V
		t_e = torch.tile(
			torch.tensor([time_step], device=DEVICE), edges.shape
		)  # Shape: E

		post = model(xt, t_v)
		post_edges = diffuser.reverse_step(
			edges[:, None], t_e, post[:, None]
		)[:, 0]  # Do everything on E x 1 tensors, and then squeeze back

		# Make copy of Data object
		xt = xt.clone()

		xt.edge_index = graph_conversions.edge_vector_to_pyg_data(
			xt, post_edges
		)

		if return_all_times:
			all_xt.append(xt)

	if return_all_times:
		return all_xt, all_t
	return xt
