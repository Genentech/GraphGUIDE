import torch
import torch_geometric
import numpy as np
import networkx as nx

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def get_nx_edges(graph):
	"""
	Returns a binary E-array of the edges in a NetworkX graph in canonical
	order. Note that self-edges are not allowed.
	Arguments:
		`graph`: a NetworkX graph
	"""
	num_nodes = graph.number_of_nodes()
	return nx.to_numpy_array(graph)[np.triu_indices(num_nodes, k=1)]


def set_nx_edges(graph, edges):
	"""
	Sets the edges in the adjacency matrix of a NetworkX graph according to the
	given edges. Note that self-edges are not allowed.
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


def get_nx_node_features(graph):
	"""
	Returns a V x D array of the node features of a NetworkX graph in canonical
	order.
	Arguments:
		`graph`: a NetworkX graph
	"""
	feat_dict = nx.get_node_attributes(graph, "feats")
	return np.vstack([feat_dict[i] for i in range(graph.number_of_nodes())])


def set_nx_node_features(graph, feats):
	"""
	Sets the node features in the NetworkX graph to the given array.
	Arguments:
		`graph`: a NetworkX graph
		`feats`: a V x D array of node features in canonical order
	"""
	feat_dict = {i : feats[i] for i in range(graph.number_of_nodes())}
	nx.set_node_attributes(graph, feat_dict, "feats")


def pyg_data_to_edge_vector(data, return_batch_inds=False):
	"""
	Returns a binary E-tensor of the edges from a torch-geometric Data object in
	canonical order. Note that self-edges are not allowed.
	Arguments:
		`data`: a torch-geometric Data object
		`return_batch_inds`: if True, also return an E-tensor which contains the
			index of the individual graph each edge belongs to
	Returns a binary E-tensor of edges, and optionally another E-tensor of batch
	indices (type int).
	"""
	# First, get (padded) adjacency matrix of size B x V x V, where V is
	# the maximum number of nodes in any individual graph
	adj_matrix = torch_geometric.utils.to_dense_adj(data.edge_index, data.batch)
	graph_sizes = torch.diff(data.ptr)

	if torch.max(data.batch) >= len(graph_sizes):
		# There are more indices in `batch` than there are pointers, so cut off
		# the excess
		adj_matrix = adj_matrix[:len(graph_sizes)]
	
	# Create boolean mask of only the top upper triangle of each B x V x V
	# matrix, ignoring the diagonal
	triu_mask = torch.triu(torch.ones_like(adj_matrix), diagonal=1) == 1
	for i, size in enumerate(torch.diff(data.ptr)):
		# TODO: make this more efficient
		# For each individual graph, limit the upper-triangle mask to
		# only the size of that graph
		triu_mask[i, :, size:] = False
		
	# Edge vector is all entries in the graph-specific upper triangle of each
	# individual adjacency matrix
	edge_vec = adj_matrix[triu_mask]
	
	if return_batch_inds:
		# Number of edges in each graph:
		edge_counts = ((graph_sizes * (graph_sizes - 1)) / 2).int()
		edge_counts_cumsum = torch.cumsum(edge_counts, dim=0)
		
		# Create binary marker array, which is all 0s except with 1s
		# wherever we switch to a new graph
		markers = torch.zeros(
			edge_counts_cumsum[-1], dtype=torch.int, device=DEVICE
		)
		markers[edge_counts_cumsum[:-1]] = 1
		
		batch_inds = torch.cumsum(markers, dim=0)
		return edge_vec, batch_inds
	return edge_vec


def edge_vector_to_pyg_data(data, edges, reflect=True):
	"""
	Returns the edge-index tensor which would be associated with the
	torch-geometric Data object `data`, if the edges in the edge index attribute
	were set according to the given edges. Note that self-edges are not allowed.
	If `edges` is a scalar 1, then this returns the set of all edges as an
	edge-index tensor.
	Arguments:
		`data`: a torch-geometric Data object
		`edges`: a binary E-tensor of edges in canonical order, or the scalar 1
		`reflect`: by default, each edge will be represented twice in the
			edge-index tensor (no self-edges are allowed); if False, only the
			upper-triangular indices will be present (and thus the tensor's size
			will be halved)
	Returns a 2 x E' edge-index tensor (type long).
	"""
	graph_sizes = torch.diff(data.ptr)
	max_size = torch.max(graph_sizes)
	
	# Create filler adjacency matrix that starts out as all 0s
	adj_matrix = torch.zeros(
		graph_sizes.shape[0], max_size, max_size, device=DEVICE
	)
	
	# Create boolean mask of only the top upper triangle of each B x V x V
	# matrix, ignoring the diagonal
	triu_mask = torch.triu(torch.ones_like(adj_matrix), diagonal=1) == 1
	for i, size in enumerate(torch.diff(data.ptr)):
		# TODO: make this more efficient
		# For each individual graph, limit the upper-triangle mask to
		# only the size of that graph
		triu_mask[i, :, size:] = False
		
	# Set the upper triangle of each graph-specific adjacency matrix to
	# the edges given,
	adj_matrix[triu_mask] = edges
	
	# Symmetrize the matrix
	if reflect:
		adj_matrix = adj_matrix + torch.transpose(adj_matrix, 1, 2)
	
	# Get indices where the adjacency matrix is nonzero (an E x 3 matrix)
	nonzero_inds = adj_matrix.nonzero()
	
	# The indices are for each individual graph, so add the graph-size
	# pointers to each set of indices so later graphs have higher
	# indices based on the sizes of earlier graphs
	edges_to_set = nonzero_inds[:, 1:] + data.ptr[nonzero_inds[:, 0]][:, None]
	
	# Convert to a 2 x E matrix
	edges_to_set = edges_to_set.t().contiguous()
	return torch_geometric.utils.sort_edge_index(edges_to_set).long()


def split_pyg_data_to_nx_graphs(data):
	"""
	Given a torch-geometric Data object, splits the objects into an ordered list
	of NetworkX graphs. The NetworkX graphs will be undirected (no self-edges
	allowed), and node features will be under the attribute "feats".
	Arguments:
		`data`: a batched torch-geometric Data object
	Returns an ordered list of NetworkX graph objects.
	"""
	graphs = []
	pointers = data.ptr.cpu().numpy()
	graph_sizes = np.diff(pointers)
	num_graphs = len(graph_sizes)

	# First, get (padded) adjacency matrix of size B x V x V, where V is
	# the maximum number of nodes in any individual graph
	adj_matrix = torch_geometric.utils.to_dense_adj(data.edge_index, data.batch)

	for i in range(num_graphs):
		graph = nx.empty_graph(graph_sizes[i])
		
		# Get the indices of adjacency matrix, upper triangle only
		edge_indices = torch.triu(
			adj_matrix[i], diagonal=1
		).nonzero().cpu().numpy()
		for u, v in edge_indices:
			graph.add_edge(u, v)

		node_feats = data.x[pointers[i] : pointers[i + 1]].cpu().numpy()
		feat_dict = {i : node_feats[i] for i in range(graph.number_of_nodes())}
		nx.set_node_attributes(graph, feat_dict, "feats")

		graphs.append(graph)
	return graphs


def add_virtual_nodes(data):
	"""
	Given a torch-geometric Data object, adds a virtual node to each individual
	graph in the batch. This modifies `data` in place. The attributes are
	modified as follows:
		`x`: new row of all 0s added for each virtual node
		`edge_index`: edges from each virtual node to all other nodes in its
			graph
		`batch`: new entry with unique index for each virtual node
		`edge_type`: introduces new edge type for each edge added
		`ptr`: unmodified
	Arguments:
		`data`: a batched torch-geometric Data object
	"""
	# Code adapted from `torch_geometric.transforms.virtual_node.VirtualNode`
	# We write our own function to make sure we add a distinct virtual node for
	# each graph in the batch, and in light of the issue here:
	# https://github.com/pyg-team/pytorch_geometric/issues/5818
	updates = {
		"x": [data.x],
		"edge_index": [data.edge_index],
		"batch": [data.batch],
		"edge_type": [
			data.get("edge_type", torch.zeros_like(data.edge_index[0]))
		]
	}
	if "num_nodes" in data:
		num_nodes = data.num_nodes

	graph_sizes = torch.diff(data.ptr)
	num_graphs = len(graph_sizes)
	device = data.x.device

	if not updates["edge_type"][0].numel():
		new_edge_type = 1
	else:
		new_edge_type = int(torch.max(updates["edge_type"][0])) + 1
	next_node_i = len(data.x)
	next_batch_i = num_graphs

	for i in range(num_graphs):
		start_node_i, end_node_i = data.ptr[i], data.ptr[i + 1]
		graph_size = graph_sizes[i]

		# New edges
		new_edges_1 = torch.full((graph_size,), next_node_i, device=device)
		new_edges_2 = torch.arange(start_node_i, end_node_i, device=device)
		new_edges = torch.cat([
			torch.stack([new_edges_1, new_edges_2], dim=0),
			torch.stack([new_edges_2, new_edges_1], dim=0),
		], dim=1)
		updates["edge_index"].append(new_edges)

		new_edge_types_1 = torch.full(
			new_edges_1.shape, new_edge_type, device=device
		)
		new_edge_types_2 = new_edge_types_1 + 1
		new_edge_types = torch.cat([new_edge_types_1, new_edge_types_2], dim=0)
		updates["edge_type"].append(new_edge_types)

		# New virtual node
		new_x = torch.zeros_like(data.x[0])[None]
		updates["x"].append(new_x)
		next_node_i = next_node_i + 1

		# New batch
		new_batch = torch.tensor([next_batch_i], device=device)
		updates["batch"].append(new_batch)
		next_batch_i = next_batch_i + 1

	data.edge_index = torch.cat(updates["edge_index"], dim=1)
	data.edge_type = torch.cat(updates["edge_type"], dim=0)
	data.x = torch.cat(updates["x"], dim=0)
	data.batch = torch.cat(updates["batch"], dim=0)

	if "num_nodes" in data:
		data.num_nodes = num_nodes + num_graphs
