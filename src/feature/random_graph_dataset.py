import torch
import torch_geometric
import numpy as np
import networkx as nx
import scipy.spatial

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def create_tree(node_dim, num_nodes=10, noise_level=1):
	"""
	Creates a random connected tree. The node attributes will be initialized by
	a random vector plus the distance from a randomly selected source node, plus
	noise.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`noise_level`: standard deviation of Gaussian noise to add to distances
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

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


def create_uniform_cliques(
	node_dim, num_nodes=10, clique_size=6, noise_level=1
):
	"""
	Creates a random graph of disconnected cliques. The node attributes will be
	initialized by a constant vector for each clique, plus some noise. If
	`clique_size` does not divide `num_nodes`, there will be a smaller clique.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`clique_size`: size of cliques
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

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


def create_diverse_cliques(
	node_dim, num_nodes=10, clique_sizes=[3, 4, 5], repeat=False, noise_level=1,
	unity_features=False
):
	"""
	Creates a random graph of disconnected cliques. The node attributes will be
	initialized by a constant vector for each clique (which is the size of the
	clique), plus some noise. Leftover nodes will be singleton nodes.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`clique_sizes`: iterable of clique sizes to use
		`repeat`: if False, all clique sizes will be unique
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
		`unity_features`: if True, use 1 for all features instead of clique size
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

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
		core = np.ones((size, 1)) * (1 if unity_features else size)
		
		# Add a small bit of noise to for each node in the clique
		node_features = core + (np.random.randn(size, node_dim) * noise_level)
		nx.set_node_attributes(
			clique, {i : node_features[i] for i in range(size)}, "feats"
		)
		
		# Add the clique to the graph
		g = nx.disjoint_union(g, clique)
		
	return g


def create_degree_graph(
	node_dim, num_nodes=10, edge_prob=0.2, noise_level=1
):
	"""
	Creates a random Erdos-Renyi graph where the node attributes are a constant
	vector of the degree of the node, plus some noise.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`edge_prob`: probability of edges in Erdos-Renyi graph
		`noise_level`: standard deviation of Gaussian noise to add to node
			features
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

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


def create_planar_graph(node_dim, num_nodes=64):
	"""
	Creates a planar graph using the Delaunay triangulation algorithm.
	All nodes will be given a feature vector of all 1s.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

	# Sample points uniformly at random from unit square
	points = np.random.rand(num_nodes, 2)

	# Perform Delaunay triangulation
	tri = scipy.spatial.Delaunay(points)

	# Create graph and add edges from triangulation result
	g = nx.empty_graph(num_nodes)
	indptr, indices = tri.vertex_neighbor_vertices
	for i in range(num_nodes):
		for j in indices[indptr[i]:indptr[i + 1]]:
			g.add_edge(i, j)
			
	nx.set_node_attributes(
		g, {i : np.ones(node_dim) for i in range(num_nodes)}, "feats"
	)
	
	return g


def create_community_graph(
	node_dim, num_nodes=np.arange(12, 21), num_comms=2,
	intra_comm_edge_prob=0.3, inter_comm_edge_frac=0.05
):
	"""
	Creates a community graph following this paper:
	https://arxiv.org/abs/1802.08773
	The default values give the definition of a "community-small" graph in the
	above paper. Each community is a Erdos-Renyi graph, with a certain set
	number of edges connecting the communities sparsely (drawn uniformly).
	All nodes will be given a feature vector of all 1s.
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`num_comms`: number of communities to create
		`intra_comm_edge_prob`: probability of edge in Erdos-Renyi graph for
			each community
		`inter_comm_edge_frac`: number of edges to draw between each pair of
			communities, as a fraction of `num_nodes`; edges are drawn uniformly
			at random between communities
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		num_nodes = np.random.choice(num_nodes)

	# Create communities
	exp_size = int(num_nodes / num_comms)
	comm_sizes = []
	total_size = 0
	g = nx.empty_graph()
	while total_size < num_nodes:
		size = min(exp_size, num_nodes - total_size)
		g = nx.disjoint_union(
			g, nx.erdos_renyi_graph(size, intra_comm_edge_prob)
		)
		comm_sizes.append(size)
		total_size += size
	
	# Link together communities
	node_inds = np.cumsum(comm_sizes)
	num_inter_edges = int(num_nodes * inter_comm_edge_frac)
	for i in range(num_comms):
		for j in range(i):
			i_nodes = np.arange(node_inds[i - 1] if i else 0, node_inds[i])
			j_nodes = np.arange(node_inds[j - 1] if j else 0, node_inds[j])
			for _ in range(num_inter_edges):
				g.add_edge(
					np.random.choice(i_nodes), np.random.choice(j_nodes)
				)
				
	nx.set_node_attributes(
		g, {i : np.ones(node_dim) for i in range(num_nodes)}, "feats"
	)
	
	return g


def create_sbm_graph(
	node_dim, num_blocks_arr=np.arange(2, 6), block_size_arr=np.arange(20, 41),
	intra_block_edge_prob=0.3, inter_block_edge_prob=0.05
):
	"""
	Creates a stochastic-block-model graph, where the number of blocks and size
	of blocks is drawn randomly.
	All nodes will be given a feature vector of all 1s.
	Arguments:
		`node_dim`: size of node feature vector
		`num_blocks_arr`: iterable containing possible numbers of blocks to have
			(selected uniformly)
		`block_size_arr`: iterable containing possible block sizes for each
			block (selected uniformly per block)
		`intra_block_edge_prob`: probability of edge within blocks
		`inter_block_edge_prob`: probability of edge between blocks
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	num_blocks = np.random.choice(num_blocks_arr)
	block_sizes = np.random.choice(block_size_arr, num_blocks, replace=True)
	num_nodes = np.sum(block_sizes)
	
	# Create matrix of edge probabilities between blocks
	p = np.full((len(block_sizes), len(block_sizes)), inter_block_edge_prob)
	np.fill_diagonal(p, intra_block_edge_prob)
	
	# Create SBM graph
	g = nx.stochastic_block_model(block_sizes, p)
	
	nx.set_node_attributes(
		g, {i : np.ones(node_dim) for i in range(num_nodes)}, "feats"
	)

	# Delete these two attributes, or else conversion to PyTorch Geometric Data
	# object will fail
	del g.graph["partition"]
	del g.graph["name"]
	
	return g


def create_molecule_like_graph(
	node_dim, num_nodes=10, backbone_size_range=(4, 6),
	ornament_size_range=(0, 4), ring_prob=0.5
):
	"""
	Creates a random graph that looks a bit like a molecule. There are two types
	of nodes: backbone nodes (with node features of 0s) and ornamental nodes
	(with node features of 1s). Leftover nodes will be either backbone or
	ornamental nodes (selected uniformly at random). The backbone may be cyclic.
	Each backbone node can be connected to at most 4 other nodes (other backbone
	nodes or ornamental nodes).
	Arguments:
		`node_dim`: size of node feature vector
		`num_nodes`: number of nodes in the graph, or an array to sample from
		`backbone_size_range`: pair of minimum and maximum sizes of backbones to
			choose from
		`ornament_size_range`: pair of minimum and maximum number of ornaments
			to choose from; the actual number of ornaments added may be smaller
			if the backbone selected is too small
		`ring_prob`: probability of generating a ring for the backbone
	Returns a NetworkX Graph with NumPy arrays as node attributes.
	"""
	if type(num_nodes) is not int:
		assert np.max(num_nodes) >= \
			backbone_size_range[1] + ornament_size_range[1]
		num_nodes = np.random.choice(num_nodes)
	else:
		assert num_nodes >= backbone_size_range[1] + ornament_size_range[1]

	backbone_size = np.random.randint(
		backbone_size_range[0], backbone_size_range[1] + 1
	)
	ornament_size = np.random.randint(
		ornament_size_range[0],
		min(ornament_size_range[1], (backbone_size * 2) + 2) + 1
	)
	ring = np.random.random() <= ring_prob

	# Create backbone
	if ring:
		g = nx.circulant_graph(backbone_size, [1])
	else:
		g = nx.empty_graph()
		g.add_node(0)
		for _ in range(backbone_size - 1):
			# Add leaf node to randomly selected node with degree < 4
			degrees = g.degree()
			connector = np.random.choice(
				[i for i in range(len(g)) if degrees[i] < 4]
			)
			new_node = len(g)
			g.add_node(new_node)
			g.add_edge(new_node, connector)

	# Add ornaments	
	capacities = [(n, 4 - m) for n, m in g.degree() if m < 4]
	inds, counts = zip(*capacities)
	node_arr = np.repeat(inds, counts)
	connectors = np.random.choice(node_arr, size=ornament_size, replace=False)
	for connector in connectors:
		new_node = len(g)
		g.add_node(new_node)
		g.add_edge(new_node, connector)

	# Add extra singleton nodes
	extra_nodes = num_nodes - len(g)
	extra_backbone_size = extra_nodes // 2
	extra_ornament_size = extra_nodes - extra_backbone_size
	for _ in range(extra_backbone_size):
		g.add_node(len(g))
	for _ in range(extra_ornament_size):
		g.add_node(len(g))

	node_features = np.concatenate([
		np.zeros((backbone_size, node_dim)),
		np.ones((ornament_size, node_dim)),
		np.zeros((extra_backbone_size, node_dim)),
		np.ones((extra_ornament_size, node_dim))
	], axis=0)

	nx.set_node_attributes(
		g, {i : node_features[i] for i in range(len(g))}, "feats"
	)
	return g


class RandomGraphDataset(torch.utils.data.Dataset):
	def __init__(
		self, node_dim, graph_type="tree", num_items=1000, static=False,
		**kwargs
	):
		"""
		Create a PyTorch IterableDataset which yields random graphs.
		Arguments:
			`node_dim`: size of node feature vector
			`graph_type`: type of graph to generate; can be "tree",
				"uniform_cliques", "diverse_cliques", "degree", "planar",
				"community", or "sbm"
			`num_items`: number of items to yield in an epoch
			`static`: if True, generate `num_items` graphs initially and only
				yield pregenerated graphs
			`kwargs`: extra keyword arguments for the graph generator
		"""
		super().__init__()
		
		self.node_dim = node_dim
		self.num_items = num_items
		self.static = static
		self.kwargs = kwargs

		if graph_type == "tree":
			self.graph_creater = create_tree
		elif graph_type == "uniform_cliques":
			self.graph_creater = create_uniform_cliques
		elif graph_type == "diverse_cliques":
			self.graph_creater = create_diverse_cliques
		elif graph_type == "degree":
			self.graph_creater = create_degree_graph
		elif graph_type == "planar":
			self.graph_creater = create_planar_graph
		elif graph_type == "community":
			self.graph_creater = create_community_graph
		elif graph_type == "sbm":
			self.graph_creater = create_sbm_graph
		elif graph_type == "molecule_like":
			self.graph_creater = create_molecule_like_graph
		else:
			raise ValueError("Unrecognize random graph type: %s" % graph_type)

		if static:
			self.graph_cache = [
				self.graph_creater(node_dim, **kwargs) for _ in range(num_items)
			]


	def __getitem__(self, index):
		"""
		Returns a single data point generated randomly, as a torch-geometric
		Data object. `index` is ignored.
		"""
		if self.static:
			graph = self.graph_cache[index]
		else:
			graph = self.graph_creater(self.node_dim, **self.kwargs)
		data = torch_geometric.utils.from_networkx(
			graph, group_node_attrs=["feats"]
		)
		data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
		return data.to(DEVICE)

	def __len__(self):
		return self.num_items


if __name__ == "__main__":
	node_dim = 5

	dataset = RandomGraphDataset(
		node_dim, num_items=100,
		graph_type="diverse_cliques", num_nodes=np.arange(10, 20),
		clique_sizes=[3, 4, 5, 6], noise_level=0
	)
	data_loader = torch_geometric.loader.DataLoader(
		dataset, batch_size=32, shuffle=False
	)
	batch = next(iter(data_loader))
