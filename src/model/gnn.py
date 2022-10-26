import torch
import torch_geometric
import numpy as np
import networkx as nx
from model.util import sanitize_sacred_arguments
import feature.graph_conversions as graph_conversions

class GraphLinkPredictor(torch.nn.Module):

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
