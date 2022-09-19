import networkx as nx
import torch
import numpy as np
import tqdm

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def create_random_ring_graph(num_nodes, node_dim, max_node_id=None):
	"""
	Creates a ring graph. The node attributes will be a one-hot encoding that
	successively iterates from 0 to `max_node_id - 1`.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
		`max_node_id`: maximum ID to give a node; by default is the size of
			the node attribute vector
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.circulant_graph(num_nodes, offsets=[1])
	
	if not max_node_id:
		max_node_id = node_dim
	id_mat = np.identity(node_dim)
	node_features = {i : id_mat[i % max_node_id] for i in range(num_nodes)}
	nx.set_node_attributes(g, node_features, "feats")
	return g


def create_random_tree(num_nodes, node_dim):
	"""
	Creates a random connected tree. The node attributes will be randomized
	one-hot encodings.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.random_tree(num_nodes)

	id_mat = np.identity(node_dim)
	node_features = {
		i : id_mat[np.random.choice(node_dim)] for i in range(num_nodes)
	}
	nx.set_node_attributes(g, node_features, "feats")
	return g


def get_edges(graph):
	"""
	Returns a binary E-array of the edges in canonical order. Note that
	self-edges are not allowed.
	Arguments:
		`graph`: a NetworkX graph
	"""
	num_nodes = graph.number_of_nodes()
	return nx.to_numpy_array(graph)[np.triu_indices(num_nodes, k=1)]

def set_edges(graph, edges):
	"""
	Sets the edges in the adjacency matrix according to the given edges.
	Note that self-edges are not allowed.
	Arguments:
		`graph`: a NetworkX graph
		`edges`: a binary E-array of edges in canonical order
	"""
	num_nodes = graph.number_of_nodes()
	assert len(edges) == (num_nodes * (num_nodes - 1)) // 2

	edges_to_set = np.stack(np.triu_indices(num_nodes, k=1), axis=1)[edges == 1]

	graph.clear_edges()

	for u, v in edges_to_set:
		graph.add_edge(u, v)

def get_node_features(graph):
	"""
	Returns a V x D array of the node features in canonical order.	
	Arguments:
		`graph`: a NetworkX graph
	"""
	feat_dict = nx.get_node_attributes(graph, "feats")
	return np.vstack([feat_dict[i] for i in range(graph.number_of_nodes())])

def set_node_features(graph, feats):
	"""
	Sets the node features in the graph to the given array.
	Arguments:
		`graph`: a NetworkX graph
		`feats`: a V x D array of node features in canonical order
	"""
	feat_dict = {i : feats[i] for i in range(graph.number_of_nodes())}
	nx.set_node_attributes(graph, feat_dict, "feats")


def beta_func(t):
	"""
	Maps a tensor of times `t` to particular noise levels beta.
	Arguments:
		`t`: a tensor of times
	Returns an equivalently shaped tensor of beta values.
	"""
	return torch.sigmoid((t / 6) - 8)


def noise_graph(edges, node_feats, beta, t_0, t_1):
	"""
	Given a graph at time `t_0`, return a version of the graph that has noise
	added to it at time `t_1`. To add noise in a single time step, make `t_1`
	equal to `t_0 + 1`.
	Arguments:
		`edges`: a binary B x E tensor of edges at time `t_0`
		`node_feats`: a B x V x D tensor of node features at time `t_0`
		`beta`: a function which maps a tensor of times to a noise level beta
		`t_0`: time of current graph, a B-tensor
		`t_1`: time to add noise to, a B-tensor
	Returns a new B x E tensor and B x V x D tensor of edges and node features
	with noise added to it, and leaves the originals untouched.
	"""
	range_lens = t_1 - t_0
	max_range = torch.max(range_lens)
	ranges = torch.tile(
		torch.arange(max_range, device=DEVICE), (t_0.shape[0], 1)
	)  # Shape: B x maxT
	betas = beta(ranges + t_0[:, None])

	# Sample a B x E x T binary tensor, where there is a 1 if a flip occurs
	flip_indicators = torch.bernoulli(
		torch.tile(betas[:, None], (1, edges.shape[1], 1))
	)
	
	# Anything that ran over the range length edge, set to 0
	mask = torch.tile(
		(ranges >= range_lens[:, None])[:, None], (1, edges.shape[1], 1)
	)
	flip_indicators[mask] = 0

	# For each edge, a flip occurs if over time the number of flips is odd
	flip_indicators_sum = torch.remainder(torch.sum(flip_indicators, dim=2), 2)

	# Perform edge flips
	new_edges = edges.clone()
	mask = flip_indicators_sum == 1
	new_edges[mask] = 1 - new_edges[mask]

	new_node_feats = node_feats.clone()

	return new_edges, new_node_feats


class GraphDenoiser(torch.nn.Module):
	def __init__(self, num_nodes, node_dim, hidden_sizes=[40, 30, 20, 10]):
		"""
		Creates a GraphDenoiser neural network.
		Arguments:
			`num_nodes`: number of nodes in the input graph
			`node_dim`: size of the node feature dimension
			`hidden_sizes`: list of hidden-layer sizes to use in both the
				encoding and decoding directions
		"""
		super().__init__()

		num_edges = (num_nodes * (num_nodes - 1)) // 2
	
		# Encoding path
		self.edge_enc = torch.nn.ModuleList()
		dims = [num_edges + 1] + hidden_sizes
		for i in range(len(hidden_sizes)):
			self.edge_enc.append(torch.nn.Linear(dims[i], dims[i + 1]))
			self.edge_enc.append(torch.nn.ReLU())
			self.edge_enc.append(torch.nn.BatchNorm1d(dims[i + 1]))
		
		# Decoding path
		self.edge_dec = torch.nn.ModuleList()
		dims = hidden_sizes[::-1]
		for i in range(len(hidden_sizes) - 1):
			self.edge_dec.append(torch.nn.Linear(dims[i], dims[i + 1]))
			self.edge_dec.append(torch.nn.ReLU())
			self.edge_dec.append(torch.nn.BatchNorm1d(dims[i + 1]))
		self.edge_dec.append(torch.nn.Linear(dims[-1], num_edges))
		self.edge_dec.append(torch.nn.Sigmoid())

		self.bce_loss = torch.nn.BCELoss(reduction="none")

	def forward(self, edges, node_feats, times):
		"""
		Runs forward pass of model for a batch of B items.
		Arguments:
			`edges`: a binary B x E tensor of edges
			`node_feats`: a B x V x D tensor of node features
			`times`: a B tensor of times to denoise from
		Returns a B x E tensor of probabilities and a B x V x D tensor of
		updated node features.
		"""
		edges_and_times = torch.cat([edges, times[:, None]], dim=1)
		x = edges_and_times
		for layer in self.edge_enc:
			x = layer(x)
		for layer in self.edge_dec:
			x = layer(x)
		return x, node_feats

	def loss(self, pred_edges, true_edges, pred_node_feats, true_node_feats):
		"""
		Computes the loss of a batch.
		Arguments:
			`pred_edges`: a B x E tensor of predicted edge probabilities
			`true_edges`: a binary B x E tensor of true edges
			`pred_node_feats`: a B x V x D tensor of predicted node features
			`true_node_feats`: a B x V x D tensor of true node features
		Returns a B-tensor of loss values.
		"""
		return torch.mean(self.bce_loss(pred_edges, true_edges), dim=1)	


class RandomGraphDataset(torch.utils.data.IterableDataset):
	def __init__(self, num_nodes, node_dim, batch_size=32, num_batches=512):
		"""
		Create a PyTorch IterableDataset which gives batches of random graphs.
		Arguments:
			`num_nodes`: number of nodes in the graph
			`node_dim`: size of node feature vector
			`batch_size`: size of batches to yield
			`num_batches`: number of batches to yield in an epoch
		"""
		self.num_nodes = num_nodes
		self.node_dim = node_dim
		self.batch_size = batch_size
		self.num_batches = num_batches

	def get_batch(self):
		"""
		Returns a batch of data generated randomly. This consists of a B x E
		array of binary edges, and a B x V x D array of node features.
		"""
		all_edges, all_node_feats = [], []
		for _ in range(self.batch_size):
			graph = create_random_tree(self.num_nodes, self.node_dim)
			all_edges.append(get_edges(graph))
			all_node_feats.append(get_node_features(graph))
		return np.stack(all_edges), np.stack(all_node_feats)

	def __iter__(self):
		"""
		Returns an iterator over the batches. If the dataset iterator is called
		from multiple workers, each worker will be give a shard of the full
		range.
		"""
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			# In single-processing mode
			start, end = 0, self.num_batches
		else:
			worker_id = worker_info.id
			num_workers = worker_info.num_workers
			shard_size = int(np.ceil(self.num_batches / num_workers))
			start = shard_size * worker_id
			end = min(start + shard_size, self.num_batches)
		return (self.get_batch() for _ in range(start, end))

	def __len__(self):
		return self.num_batches


def train_model(model, data_loader, t_limit, num_epochs, learning_rate):
	"""
	Trains a GraphDenoiser model.
	Arguments:
		`model`: an initialized GraphDenoiser model
		`data_loader`: a DataLoader which iterates over batches of graph edges
			and node features
		`t_limit`: maximum time range for noising
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
	"""
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	model.train()
	torch.set_grad_enabled(True)

	for epoch_num in range(num_epochs):
		losses = []
		t_iter = tqdm.tqdm(
			data_loader, total=len(data_loader.dataset), desc="Loss: -----"
		)

		for edges_0, node_feats_0 in t_iter:
			edges_0 = torch.tensor(edges_0).float().to(DEVICE)
			node_feats_0 = torch.tensor(node_feats_0).float().to(DEVICE)

			# Pick some random times t-1 between 0 and t_limit
			t_1 = torch.randint(
				t_limit, size=(edges_0.shape[0],), device=DEVICE
			)

			# Add noise to graphs from time 0 to time t - 1
			edges_t_1, node_feats_t_1 = noise_graph(
				edges_0, node_feats_0, beta_func, 
				torch.zeros(edges_0.shape[0], device=DEVICE), t_1
			)

			# Add one more step of noise
			edges_t, node_feats_t = noise_graph(
				edges_t_1, node_feats_t_1, beta_func,
				t_1, t_1 + 1
			)

			# Have model try and predict graph at t - 1 from time t
			pred_edges_t_1, pred_node_feats_t_1 = model(
				edges_t, node_feats_t, t_1
			)

			loss = torch.mean(model.loss(
				pred_edges_t_1, edges_t_1, pred_node_feats_t_1, node_feats_t_1
			))
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			t_iter.set_description(
				"Loss: %6.4f" % loss.item()
			)
			losses.append(loss.item())

		print("Epoch %d average loss: %.2f" % (epoch_num + 1, np.mean(losses)))

num_nodes, node_dim = 50, 1
t_limit = 50

dataset = RandomGraphDataset(num_nodes, node_dim)
data_loader = torch.utils.data.DataLoader(
	dataset, batch_size=None, num_workers=4, collate_fn=(lambda x: x)
)

# # Number of expected cycles in prior
# expected_prior_cycles = np.mean([
# 	len(nx.cycle_basis(nx.erdos_renyi_graph(num_nodes, 0.5)))
# 	for _ in range(500)
# ])
# print("Expected cycles in limit: %.2f" % expected_prior_cycles)
# 
# # Check progression of number of cycles
# t_vals = np.arange(1, t_limit)
# num_cycles = np.empty((len(t_vals), dataset.batch_size))
# edges_0, node_feats_0 = next(iter(data_loader))
# edges_0 = torch.tensor(edges_0).to(DEVICE)
# node_feats_0 = torch.tensor(node_feats_0).to(DEVICE)
# t_0 = torch.zeros(edges_0.shape[0], device=DEVICE)
# for i, t in tqdm.tqdm(enumerate(t_vals), total=len(t_vals)):
# 	t_tens = torch.ones(edges_0.shape[0], device=DEVICE) * t
# 	edges_t, _ = noise_graph(edges_0, node_feats_0, beta_func, t_0, t_tens)
# 	for j in range(dataset.batch_size):
# 		g = nx.empty_graph(num_nodes)
# 		set_edges(g, edges_t[j].cpu().numpy())
# 		num_cycles[i, j] = len(nx.cycle_basis(g))
# 
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(20, 8))
# ax.plot(t_vals, np.mean(num_cycles, axis=1))
# ax.set_ylabel("Average number of cycles")
# ax.set_xlabel("t")
# ax.set_title("Actual progression of cycles")
# plt.show()

model = GraphDenoiser(num_nodes, node_dim).to(DEVICE)
train_model(
	model, data_loader,
	t_limit=50,
	num_epochs=3,
	learning_rate=0.001
)

# Take the trained model and go backwards to sample some graphs
print("Sampling some reverse trajectories")
batch_size = 32
num_edges = (num_nodes * (num_nodes - 1)) // 2
edges = np.empty((batch_size, num_edges))
node_feats = np.empty((batch_size, num_nodes, node_dim))
for i in range(batch_size):
	g = nx.erdos_renyi_graph(num_nodes, 0.5)
	edges[i] = get_edges(g)
edges = torch.tensor(edges).float().to(DEVICE)
node_feats = torch.tensor(node_feats).float().to(DEVICE)

for t in tqdm.trange(50 - 1, 1, -1):
	edges, _ = model(
		edges, node_feats, torch.ones(batch_size, device=DEVICE) * t
	)

	graphs = []
	num_cycles = []
	num_components = []
	e = edges.detach().cpu().numpy()
	for i in range(batch_size):
		g = nx.empty_graph(num_nodes)
		set_edges(g, e[i])
		graphs.append(g)
		num_cycles.append(len(nx.cycle_basis(g)))
		num_components.append(nx.number_connected_components(g))
	print(num_cycles)
	print(num_components)
	print("---------")

edges = edges.detach().cpu().numpy()
graphs = []
num_cycles = []
num_components = []
for i in range(batch_size):
	g = nx.empty_graph(num_nodes)
	set_edges(g, edges[i])
	graphs.append(g)
	num_cycles.append(len(nx.cycle_basis(g)))
	num_components.append(nx.number_connected_components(g))

print(num_cycles)
print(num_components)
