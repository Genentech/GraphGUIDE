import torch
import torch_geometric
import numpy as np
import networkx as nx
from model.util import sanitize_sacred_arguments
import feature.graph_conversions as graph_conversions

class GraphLinkGIN(torch.nn.Module):

	def __init__(
		self, input_dim, t_limit, num_gnn_layers=4, hidden_dim=10,
		time_embed_size=256, virtual_node=False
	):
		"""
		Initialize a time-dependent GNN which predicts bit probabilities for
		each edge.
		Arguments:
			`input_dim`: the dimension of the input node features
			`t_limit`: maximum time horizon
			`num_gnn_layers`: number of GNN layers to have
			`hidden_dim`: the dimension of the hidden node embeddings
			`time_embed_size`: size of the time embeddings
			`virtual_node`: if True, include a virtual node in the architecture
		"""
		super().__init__()
		
		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.t_limit = t_limit
		self.num_gnn_layers = num_gnn_layers
		self.virtual_node = virtual_node
		
		self.time_embed_dense = torch.nn.Linear(3, time_embed_size)
		
		self.swish = lambda x: x * torch.sigmoid(x)
		self.relu = torch.nn.ReLU()
		
		# GNN layers
		self.gnn_layers = torch.nn.ModuleList()
		self.gnn_batch_norms = torch.nn.ModuleList()
		for i in range(num_gnn_layers):
			gnn_nn = torch.nn.Sequential(
				torch.nn.Linear(
					input_dim + time_embed_size if i == 0 else hidden_dim,
					hidden_dim * 2
				),
				self.relu,
				torch.nn.Linear(hidden_dim * 2, hidden_dim)
			)
			gnn_layer = torch_geometric.nn.GINConv(gnn_nn, train_eps=True)
			gnn_batch_norm = torch_geometric.nn.LayerNorm(hidden_dim)

			self.gnn_layers.append(gnn_layer)
			self.gnn_batch_norms.append(gnn_batch_norm)

		# Link prediction
		self.link_dense = torch.nn.Linear(hidden_dim, 1)
		
		# Loss
		self.bce_loss = torch.nn.BCELoss()

	def forward(self, data, t):
		"""
		Forward pass of the network.
		Arguments:
			`data`: a (batched) torch-geometric Data object
			`t`: a V-tensor containing the time to train on for each node; note
				that the time should be the same for nodes belonging to the same
				individual graph
		Returns an E-tensor of probabilities of each edge at time t - 1, where E
		is the total possible number of edges, and is in canonical ordering.
		"""
		if self.virtual_node:
			# Add a virtual node
			data_copy = data.clone()  # Don't modify the original
			graph_conversions.add_virtual_nodes(data_copy)  # Modify `data_copy`
			# Also extend `t` by the number of virtual nodes added (use 0,
			# since the node is fake)
			# TODO: it would be better to use the same time as its host graph
			t = torch.cat([
				t, torch.zeros(len(data.ptr) - 1, device=t.device)
			])
			data = data_copy  # Use our modified Data object

		# Get the time embeddings for `t`
		time_embed_args = t[:, None] / self.t_limit  # Shape: V x 1
		time_embed = self.swish(self.time_embed_dense(
				torch.cat([
					torch.sin(time_embed_args * (np.pi / 2)),
					torch.cos(time_embed_args * (np.pi / 2)),
					time_embed_args
				], dim=1)
		))	# Shape: V x Dt
		
		# Concatenate initial node features and time embedding
		node_embed = torch.cat([data.x.float(), time_embed], dim=1)
		# Shape: V x D
		
		# GNN layers
		for i in range(self.num_gnn_layers):
			node_embed = self.gnn_batch_norms[i](
				self.gnn_layers[i](node_embed, data.edge_index),
				data.batch
			)
			
		# For all possible edges (i.e. node pairs), compute probability
		edge_inds = graph_conversions.edge_vector_to_pyg_data(
			data, 1, reflect=False
		)  # Shape: 2 x E
		node_embed_1 = node_embed[edge_inds[0]]  # Shape: E x D'
		node_embed_2 = node_embed[edge_inds[1]]  # Shape: E x D'
		node_prod = node_embed_1 * node_embed_2
		
		edge_probs = torch.sigmoid(self.link_dense(node_prod))[:, 0]
		
		return edge_probs

	def loss(self, pred_probs, true_probs):
		"""
		Computes the loss of a batch.
		Arguments:
			`pred_probs`: an E-tensor of predicted probabilities
			`true_probs`: an E-tensor of true probabilities
		Returns a scalar loss value.
		"""
		return self.bce_loss(pred_probs, true_probs)


class GraphAttentionLayer(torch.nn.Module):

	def __init__(self, input_dim, num_heads=8, att_hidden_dim=32, dropout=0.1):
		"""
		Initialize a graph attention layer which computes attention between all
		nodes.
		Arguments:
			`input_dim`: the dimension of the input node features, D
			`num_heads`: number of attention heads
			`att_hidden_dim`: the dimension of the hidden node embeddings in the
				GAT
			`dropout`: dropout rate for post-GAT layers
		"""

		super().__init__()

		self.input_dim = input_dim
		self.num_heads = num_heads
		self.att_hidden_dim = att_hidden_dim
		
		self.spectral_dense = torch.nn.Linear(input_dim, input_dim)

		self.gat = torch_geometric.nn.GATv2Conv(
			input_dim * 2, att_hidden_dim, heads=num_heads, edge_dim=1
			# edge_dim is 1 for presence/absence of edge
		)

		self.dropout_1 = torch.nn.Dropout(dropout)
		self.norm_1 = torch_geometric.nn.LayerNorm(
			att_hidden_dim * num_heads
		)
		self.dense_2 = torch.nn.Linear(
			att_hidden_dim * num_heads, input_dim 
		)
		self.dropout_2 = torch.nn.Dropout(dropout)
		self.norm_2 = torch_geometric.nn.LayerNorm(input_dim)

		self.dense_3 = torch.nn.Linear(input_dim, input_dim)
		self.dropout_3 = torch.nn.Dropout(dropout)
		self.norm_3 = torch_geometric.nn.LayerNorm(input_dim)

		self.relu = torch.nn.ReLU()

	def forward(
		self, x, full_edge_index, edge_indicators, batch, spectrum_mats
	):
		"""
		Forward pass of the attention layer.
		Arguments:
			`x`: V x D tensor of node features
			`full_edge_index`: 2 x E tensor of edge indices for the attention
				mechanism, denoting all possible edges within each subgraph in
				the batch
			`edge_feats`: E x 1 tensor of edge features, denoting which edges
				actually exist in each subgraph (e.g. 1 if the edge exists, 0
				otherwise)
			`batch`: V-tensor of which nodes belong to which batch
			`spectrum_mats`: list of B matrices, where each matrix is n x m,
				where n is the number of nodes in each graph and m is the
				number of eigenvectors (columns); this matrix transforms _out_
				of the spectral domain when left-multiplying node features; B is
				the number of graphs in the batch
		Returns a V x D tensor of updated node features.
		"""
		# Perform spectral convolution
		specconv_out = x.clone()
		for i in range(len(spectrum_mats)):
			batch_mask = batch == i
			x_in = x[batch_mask]  # Shape: n x d
			mat = spectrum_mats[i]  # Shape: n x m
			# Transform into spectral domain
			x_spec = torch.matmul(torch.transpose(mat, 0, 1), x_in)
			# Shape: m x d
			# Perform convolution by combining channels across each "frequency"
			x_spec_out = self.spectral_dense(x_spec)  # Shape: m x d
			# Transform back into feature domain
			x_out = torch.matmul(mat, x_spec_out)  # Shape: n x d
			specconv_out[batch_mask] = x_out
		
		# Attention on x and output of spectral convolution
		gat_out = self.gat(
			torch.cat([x, specconv_out], dim=1), full_edge_index,
			edge_attr=edge_indicators
		)
		x_out_1 = self.norm_1(x + self.relu(self.dropout_1(gat_out)))

		# Post-attention dense layers
		x_out_2 = self.norm_2(x_out_1 + self.dropout_2(self.dense_2(x_out_1)))
		x_out_3 = self.norm_3(x_out_2 + self.dropout_3(self.dense_3(x_out_2)))
		return x_out_3


class GraphLinkGAT(torch.nn.Module):

	def __init__(
		self, input_dim, t_limit, num_gnn_layers=4, gat_num_heads=8,
		gat_hidden_dim=32, hidden_dim=256, time_embed_size=256, spectrum_dim=5,
		epsilon=1e-6
	):
		"""
		Initialize a time-dependent GNN which predicts bit probabilities for
		each edge.
		Arguments:
			`input_dim`: the dimension of the input node features
			`t_limit`: maximum time horizon
			`num_gnn_layers`: number of GNN layers to have
			`gat_num_heads`: number of attention heads
			`gat_hidden_dim`: the dimension of the hidden node embeddings in the
				GAT
			`hidden_dim`: size of hidden dimension before and after attention
				layers
			`time_embed_size`: size of the time embeddings
			`spectrum_dim`: number of spectral features to use (i.e. the number
				of eigenvectors to use)
			`epsilon`: small number for numerical stability when computing graph
				Laplacian
		"""
		super().__init__()

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.t_limit = t_limit
		self.num_gnn_layers = num_gnn_layers
		self.spectrum_dim = spectrum_dim
		self.epsilon = epsilon
		
		self.time_embed_dense = torch.nn.Linear(3, time_embed_size)
		
		self.swish = lambda x: x * torch.sigmoid(x)
		self.relu = torch.nn.ReLU()

		# Pre-GNN linear layers
		self.pregnn_dense_1 = torch.nn.Linear(
			input_dim + time_embed_size, hidden_dim
		)
		self.pregnn_dense_2 = torch.nn.Linear(hidden_dim, hidden_dim)
	
		# GNN layers
		self.gnn_layers = torch.nn.ModuleList()
		for i in range(num_gnn_layers):
			self.gnn_layers.append(GraphAttentionLayer(
				hidden_dim if i == 0 else gat_num_heads * gat_hidden_dim,
				num_heads=gat_num_heads, att_hidden_dim=gat_hidden_dim
			))

		# Pre-GNN linear layers
		self.postgnn_dense_1 = torch.nn.Linear(
			gat_hidden_dim * gat_num_heads, hidden_dim
		)
		self.postgnn_dense_2 = torch.nn.Linear(hidden_dim, hidden_dim)
		
		# Link prediction
		self.link_dense = torch.nn.Linear(hidden_dim, 1)
		
		# Loss
		self.bce_loss = torch.nn.BCELoss()

	def forward(self, data, t):
		"""
		Forward pass of the network.
		Arguments:
			`data`: a (batched) torch-geometric Data object
			`t`: a V-tensor containing the time to train on for each node; note
				that the time should be the same for nodes belonging to the same
				individual graph
		Returns an E-tensor of probabilities of each edge at time t - 1, where E
		is the total possible number of edges, and is in canonical ordering.
		"""
		# Get the time embeddings for `t`
		time_embed_args = t[:, None] / self.t_limit  # Shape: V x 1
		time_embed = self.swish(self.time_embed_dense(
				torch.cat([
					torch.sin(time_embed_args * (np.pi / 2)),
					torch.cos(time_embed_args * (np.pi / 2)),
					time_embed_args
				], dim=1)
		))  # Shape: V x Dt
		
		# Concatenate initial node features and time embedding
		node_embed = torch.cat([data.x.float(), time_embed], dim=1)
		# Shape: V x D

		# Pre-GNN dense layers on node features
		node_embed = self.relu(self.pregnn_dense_1(node_embed))
		node_embed = self.relu(self.pregnn_dense_2(node_embed))

		# Create edge_index specifying the full dense subgraphs
		full_edge_index = graph_conversions.edge_vector_to_pyg_data(
			data, 1, reflect=False
		)  # Shape: 2 x E

		# Create edge features, which denotes both which edges are real
		edge_indicators = \
			graph_conversions.pyg_data_to_edge_vector(data)[:, None]
		# Shape: E x 1
	
		# Compute the Laplacian for each graph in the batch
		adj_mat = torch_geometric.utils.to_dense_adj(
			data.edge_index, data.batch
		)
		deg = torch.sum(adj_mat, dim=2)
		sqrt_deg = 1 / torch.sqrt(deg + self.epsilon)
		sqrt_deg_mat = torch.diag_embed(sqrt_deg)
		
		identity = torch.eye(adj_mat.size(1), device=adj_mat.device)[None]
		laplacian = identity - \
			torch.matmul(torch.matmul(sqrt_deg_mat, adj_mat), sqrt_deg_mat)

		# Compute spectrum transformation matrix (i.e. eigenvalues/eigenvectors)
		# This is done separately for each graph, as each graph may have a
		# different size
		spectrum_mats = []
		for i, graph_size in enumerate(torch.diff(data.ptr)):
			# We only compute the eigendecomposition on the graph-size-limited
			# Laplacian, since this function always sorts eigenvalues
			evals, evecs = torch.linalg.eigh(
				laplacian[i, :graph_size, :graph_size]
			)
			# Limit the eigenvectors to the smallest eigenvalues if needed
			if self.spectrum_dim < graph_size:
				evecs = evecs[:, :self.spectrum_dim]
			spectrum_mats.append(evecs)

		# GNN layers
		for i in range(self.num_gnn_layers):
			node_embed = self.gnn_layers[i](
				node_embed, full_edge_index, edge_indicators, data.batch,
				spectrum_mats
			)
			
		# Post-GNN dense layers on node features
		node_embed = self.relu(self.postgnn_dense_1(node_embed))
		node_embed = self.relu(self.postgnn_dense_2(node_embed))

		# For all possible edges (i.e. node pairs), compute probability
		node_embed_1 = node_embed[full_edge_index[0]]  # Shape: E x D'
		node_embed_2 = node_embed[full_edge_index[1]]  # Shape: E x D'
		node_prod = node_embed_1 * node_embed_2
		
		edge_probs = torch.sigmoid(self.link_dense(node_prod))[:, 0]
		
		return edge_probs

	def loss(self, pred_probs, true_probs):
		"""
		Computes the loss of a batch.
		Arguments:
			`pred_probs`: an E-tensor of predicted probabilities
			`true_probs`: an E-tensor of true probabilities
		Returns a scalar loss value.
		"""
		return self.bce_loss(pred_probs, true_probs)
