import torch
import torch_geometric
import numpy as np
import networkx as nx

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def create_random_tree(num_nodes, node_dim, noise_level=1):
	"""
	Creates a random connected tree. The node attributes will be initialized by
	a random vector plus the distance from a randomly selected source node, plus
	noise.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
		`noise_level`: standard deviation of Gaussian noise to add to distances
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.random_tree(num_nodes)

	node_features = np.empty((num_nodes, node_dim))
	
	# Pick a random source node
	source = np.random.choice(num_nodes)
	
	# Set source node's feature to random vector
	source_feat = np.random.randn(node_dim) * 2 * np.sqrt(num_nodes)
	node_features[source] = source_feat
	
	# Run BFS starting from source node; for each layer, set features
	# to be the source vector plus the distance plus noise
	current_layer = [source]
	distance = 1
	visited = set(current_layer)
	while current_layer:
		next_layer = []
		for node in current_layer:
			for child in g[node]:
				if child not in visited:
					visited.add(child)
					next_layer.append(child)
					node_features[child] = source_feat + distance + (
						np.random.randn(node_dim) * noise_level
					)
		current_layer = next_layer
		distance += 1

	nx.set_node_attributes(
		g, {i : node_features[i] for i in range(num_nodes)}, "feats"
	)
	return g


def create_random_uniform_cliques(
	num_nodes, node_dim, clique_size=6, noise_level=1
):
	"""
	Creates a random graph of disconnected cliques. The node attributes will be
	initialized by a constant vector for each clique, plus some noise. If
	`clique_size` does not divide `num_nodes`, there will be a smaller clique.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
		`clique_size`: size of cliques
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.empty_graph()
	
	clique_count = 0
	while g.number_of_nodes() < num_nodes:
		size = min(clique_size, num_nodes - g.number_of_nodes())
		
		# Create clique
		clique = nx.complete_graph(size)
		
		# Create the core feature vector for the clique
		core = np.ones((node_dim, 1)) * clique_count * \
			(2 * num_nodes / clique_size)
		
		# Add a small bit of noise to for each node in the clique
		node_features = core + (np.random.randn(size, node_dim) * noise_level)
		nx.set_node_attributes(
			clique, {i : node_features[i] for i in range(size)}, "feats"
		)
		
		# Add the clique to the graph
		g = nx.disjoint_union(g, clique)
		clique_count += 1
		
	return g


def create_random_diverse_cliques(
	num_nodes, node_dim, clique_sizes=[3, 4, 5], repeat=False, noise_level=1
):
	"""
	Creates a random graph of disconnected cliques. The node attributes will be
	initialized by a constant vector for each clique (which is the size of the
	clique), plus some noise. Leftover nodes will be singleton nodes.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
		`clique_sizes`: iterable of clique sizes to use
		`repeat`: if False, all clique sizes will be unique
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.empty_graph()
	
	clique_sizes = np.unique(clique_sizes)
	
	sizes_to_make, size_left = [], num_nodes
	if repeat:
		while clique_sizes.size:
			size = np.random.choice(clique_sizes)
			sizes_to_make.append(size)
			size_left -= size
			clique_sizes = clique_sizes[clique_sizes >= size_left]
	else:
		clique_sizes = np.random.permutation(clique_sizes)
		for size in clique_sizes:
			if size <= size_left:
				sizes_to_make.append(size)
				size_left -= size
	sizes_to_make.extend([1] * size_left)
	
	for size in sizes_to_make:
		# Create clique
		clique = nx.complete_graph(size)
		
		# Create the core feature vector for the clique
		core = np.ones((size, 1)) * size
		
		# Add a small bit of noise to for each node in the clique
		node_features = core + (np.random.randn(size, node_dim) * noise_level)
		nx.set_node_attributes(
			clique, {i : node_features[i] for i in range(size)}, "feats"
		)
		
		# Add the clique to the graph
		g = nx.disjoint_union(g, clique)
		
	return g


def create_random_er_degree_graph(
	num_nodes, node_dim, edge_prob=0.2, noise_level=1
):
	"""
	Creates a random Erdos-Renyi graph where the node attributes are a constant
	vector of the degree of the node, plus some noise.
	Arguments:
		`num_nodes`: number of nodes in the graph
		`node_dim`: size of node feature vector
		`edge_prob`: probability of edges in Erdos-Renyi graph
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	g = nx.erdos_renyi_graph(num_nodes, edge_prob)

	degrees = dict(g.degree())
	node_features = np.tile(
		np.array([degrees[n] for n in range(len(g))])[:, None],
		(1, node_dim)
	)
	node_features = node_features + \
		(np.random.randn(num_nodes, node_dim) * noise_level)

	nx.set_node_attributes(
		g, {i : node_features[i] for i in range(num_nodes)}, "feats"
	)

	return g


class RandomGraphDataset(torch.utils.data.Dataset):
	def __init__(
		self, num_nodes, node_dim, graph_type="tree", num_items=1000,
		**kwargs
	):
		"""
		Create a PyTorch IterableDataset which yields random graphs.
		Arguments:
			`num_nodes`: number of nodes in the graph; can be an integer or a
				NumPy array of integers to sample from uniformly
			`node_dim`: size of node feature vector
			`num_items`: number of items to yield in an epoch
			`graph_type`: type of graph to generate; can be "tree",
				"uniform_cliques", "diverse_cliques", or "degree_graph"
			`kwargs`: extra keyword arguments for the graph generator
		"""
		super().__init__()
		
		if type(num_nodes) is int:
			num_nodes = np.array([num_nodes])
			
		self.num_nodes = num_nodes
		self.node_dim = node_dim
		self.num_items = num_items
		self.graph_type = graph_type
		self.kwargs = kwargs

	def __getitem__(self, index):
		"""
		Returns a single data point generated randomly, as a torch-geometric
		Data object. `index` is ignored.
		"""
		num_nodes = np.random.choice(self.num_nodes)
		if self.graph_type == "tree":
			graph = create_random_tree(num_nodes, self.node_dim, **self.kwargs)
		elif self.graph_type == "uniform_cliques":
			graph = create_random_uniform_cliques(
				num_nodes, self.node_dim, **self.kwargs
			)
		elif self.graph_type == "diverse_cliques":
			graph = create_random_diverse_cliques(
				num_nodes, self.node_dim, **self.kwargs
			)
		elif self.graph_type == "degree_graph":
			graph = create_random_er_degree_graph(
				num_nodes, self.node_dim, **self.kwargs
			)

		data = torch_geometric.utils.from_networkx(
			graph, group_node_attrs=["feats"]
		)
		data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
		return data.to(DEVICE)

	def __len__(self):
		return self.num_items


if __name__ == "__main__":
	num_nodes, node_dim = 10, 5

	dataset = RandomGraphDataset(
	    num_nodes, node_dim, num_items=100,
	    graph_type="diverse_cliques", clique_sizes=[3, 4, 5, 6], noise_level=0
	)
	data_loader = torch_geometric.loader.DataLoader(
	    dataset, batch_size=32, shuffle=False
	)
	batch = next(iter(data_loader))
