import math
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
import torch_geometric

def assert_correctly_masked(variable, node_mask):
	assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
		'Variables not masked properly.'
	
class PlaceHolder:
	def __init__(self, X, E, y):
		self.X = X
		self.E = E
		self.y = y

	def type_as(self, x: torch.Tensor):
		""" Changes the device and dtype of X, E, y. """
		self.X = self.X.type_as(x)
		self.E = self.E.type_as(x)
		self.y = self.y.type_as(x)
		return self

	def mask(self, node_mask, collapse=False):
		x_mask = node_mask.unsqueeze(-1)		  # bs, n, 1
		e_mask1 = x_mask.unsqueeze(2)			  # bs, n, 1, 1
		e_mask2 = x_mask.unsqueeze(1)			  # bs, 1, n, 1

		if collapse:
			self.X = torch.argmax(self.X, dim=-1)
			self.E = torch.argmax(self.E, dim=-1)

			self.X[node_mask == 0] = - 1
			self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
		else:
			self.X = self.X * x_mask
			self.E = self.E * e_mask1 * e_mask2
			assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
		return self


def encode_no_edge(E):
	assert len(E.shape) == 4
	if E.shape[-1] == 0:
		return E
	no_edge = torch.sum(E, dim=3) == 0
	first_elt = E[:, :, :, 0]
	first_elt[no_edge] = 1
	E[:, :, :, 0] = first_elt
	diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
	E[diag] = 0
	return E

def to_dense(x, edge_index, edge_attr, batch):
	X, node_mask = torch_geometric.utils.to_dense_batch(x=x, batch=batch)
	# node_mask = node_mask.float()
	edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
	# TODO: carefully check if setting node_mask as a bool breaks the continuous case
	max_num_nodes = X.size(1)
	if edge_index.numel() == 0:
		# We have to do this check otherwise things fail
		E = torch_geometric.utils.to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr)
	else:
		E = torch_geometric.utils.to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
	E = encode_no_edge(E)

	return PlaceHolder(X=X, E=E, y=None), node_mask


class Xtoy(nn.Module):
	def __init__(self, dx, dy):
		""" Map node features to global features """
		super().__init__()
		self.lin = nn.Linear(4 * dx, dy)

	def forward(self, X):
		""" X: bs, n, dx. """
		m = X.mean(dim=1)
		mi = X.min(dim=1)[0]
		ma = X.max(dim=1)[0]
		std = X.std(dim=1)
		z = torch.hstack((m, mi, ma, std))
		out = self.lin(z)
		return out


class Etoy(nn.Module):
	def __init__(self, d, dy):
		""" Map edge features to global features. """
		super().__init__()
		self.lin = nn.Linear(4 * d, dy)

	def forward(self, E):
		""" E: bs, n, n, de
			Features relative to the diagonal of E could potentially be added.
		"""
		m = E.mean(dim=(1, 2))
		mi = E.min(dim=2)[0].min(dim=1)[0]
		ma = E.max(dim=2)[0].max(dim=1)[0]
		std = torch.std(E, dim=(1, 2))
		z = torch.hstack((m, mi, ma, std))
		out = self.lin(z)
		return out

class XEyTransformerLayer(nn.Module):
	""" Transformer that updates node, edge and global features
		d_x: node features
		d_e: edge features
		dz : global features
		n_head: the number of heads in the multi_head_attention
		dim_feedforward: the dimension of the feedforward network model after self-attention
		dropout: dropout probablility. 0 to disable
		layer_norm_eps: eps value in layer normalizations.
	"""
	def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
				 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
				 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
		kw = {'device': device, 'dtype': dtype}
		super().__init__()

		self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

		self.linX1 = Linear(dx, dim_ffX, **kw)
		self.linX2 = Linear(dim_ffX, dx, **kw)
		self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
		self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
		self.dropoutX1 = Dropout(dropout)
		self.dropoutX2 = Dropout(dropout)
		self.dropoutX3 = Dropout(dropout)

		self.linE1 = Linear(de, dim_ffE, **kw)
		self.linE2 = Linear(dim_ffE, de, **kw)
		self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
		self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
		self.dropoutE1 = Dropout(dropout)
		self.dropoutE2 = Dropout(dropout)
		self.dropoutE3 = Dropout(dropout)

		self.lin_y1 = Linear(dy, dim_ffy, **kw)
		self.lin_y2 = Linear(dim_ffy, dy, **kw)
		self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
		self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
		self.dropout_y1 = Dropout(dropout)
		self.dropout_y2 = Dropout(dropout)
		self.dropout_y3 = Dropout(dropout)

		self.activation = F.relu

	def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
		""" Pass the input through the encoder layer.
			X: (bs, n, d)
			E: (bs, n, n, d)
			y: (bs, dy)
			node_mask: (bs, n) Mask for the src keys per batch (optional)
			Output: newX, newE, new_y with the same shape.
		"""

		newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

		newX_d = self.dropoutX1(newX)
		X = self.normX1(X + newX_d)

		newE_d = self.dropoutE1(newE)
		E = self.normE1(E + newE_d)

		new_y_d = self.dropout_y1(new_y)
		y = self.norm_y1(y + new_y_d)

		ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
		ff_outputX = self.dropoutX3(ff_outputX)
		X = self.normX2(X + ff_outputX)

		ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
		ff_outputE = self.dropoutE3(ff_outputE)
		E = self.normE2(E + ff_outputE)

		ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
		ff_output_y = self.dropout_y3(ff_output_y)
		y = self.norm_y2(y + ff_output_y)

		return X, E, y


class NodeEdgeBlock(nn.Module):
	""" Self attention layer that also updates the representations on the edges. """
	def __init__(self, dx, de, dy, n_head, **kwargs):
		super().__init__()
		assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
		self.dx = dx
		self.de = de
		self.dy = dy
		self.df = int(dx / n_head)
		self.n_head = n_head

		# Attention
		self.q = Linear(dx, dx)
		self.k = Linear(dx, dx)
		self.v = Linear(dx, dx)

		# FiLM E to X
		self.e_add = Linear(de, dx)
		self.e_mul = Linear(de, dx)

		# FiLM y to E
		self.y_e_mul = Linear(dy, dx)			# Warning: here it's dx and not de
		self.y_e_add = Linear(dy, dx)

		# FiLM y to X
		self.y_x_mul = Linear(dy, dx)
		self.y_x_add = Linear(dy, dx)

		# Process y
		self.y_y = Linear(dy, dy)
		self.x_y = Xtoy(dx, dy)
		self.e_y = Etoy(de, dy)

		# Output layers
		self.x_out = Linear(dx, dx)
		self.e_out = Linear(dx, de)
		self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

	def forward(self, X, E, y, node_mask):
		"""
		:param X: bs, n, d		  node features
		:param E: bs, n, n, d	  edge features
		:param y: bs, dz		   global features
		:param node_mask: bs, n
		:return: newX, newE, new_y with the same shape.
		"""
		x_mask = node_mask.unsqueeze(-1)		# bs, n, 1
		e_mask1 = x_mask.unsqueeze(2)			# bs, n, 1, 1
		e_mask2 = x_mask.unsqueeze(1)			# bs, 1, n, 1

		# 1. Map X to keys and queries
		Q = self.q(X) * x_mask			 # (bs, n, dx)
		K = self.k(X) * x_mask			 # (bs, n, dx)
		assert_correctly_masked(Q, x_mask)
		# 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

		Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
		K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

		Q = Q.unsqueeze(2)								# (bs, 1, n, n_head, df)
		K = K.unsqueeze(1)								# (bs, n, 1, n head, df)

		# Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
		Y = Q * K
		Y = Y / math.sqrt(Y.size(-1))
		assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

		E1 = self.e_mul(E) * e_mask1 * e_mask2						  # bs, n, n, dx
		E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

		E2 = self.e_add(E) * e_mask1 * e_mask2						  # bs, n, n, dx
		E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

		# Incorporate edge features to the self attention scores.
		Y = Y * (E1 + 1) + E2				   # (bs, n, n, n_head, df)

		# Incorporate y to E
		newE = Y.flatten(start_dim=3)					   # bs, n, n, dx
		ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
		ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
		newE = ye1 + (ye2 + 1) * newE

		# Output E
		newE = self.e_out(newE) * e_mask1 * e_mask2		 # bs, n, n, de
		assert_correctly_masked(newE, e_mask1 * e_mask2)

		# Compute attentions. attn is still (bs, n, n, n_head, df)
		attn = F.softmax(Y, dim=2)

		V = self.v(X) * x_mask						  # bs, n, dx
		V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
		V = V.unsqueeze(1)									   # (bs, 1, n, n_head, df)

		# Compute weighted values
		weighted_V = attn * V
		weighted_V = weighted_V.sum(dim=2)

		# Send output to input dim
		weighted_V = weighted_V.flatten(start_dim=2)			# bs, n, dx

		# Incorporate y to X
		yx1 = self.y_x_add(y).unsqueeze(1)
		yx2 = self.y_x_mul(y).unsqueeze(1)
		newX = yx1 + (yx2 + 1) * weighted_V

		# Output X
		newX = self.x_out(newX) * x_mask
		assert_correctly_masked(newX, x_mask)

		# Process y based on X axnd E
		y = self.y_y(y)
		e_y = self.e_y(E)
		x_y = self.x_y(X)
		new_y = y + x_y + e_y
		new_y = self.y_out(new_y)				# bs, dy

		return newX, newE, new_y


class GraphTransformer(nn.Module):
	"""
	n_layers : int -- number of layers
	dims : dict -- contains dimensions for each feature type
	"""
	def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
				 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
		super().__init__()
		self.n_layers = n_layers
		self.out_dim_X = output_dims['X']
		self.out_dim_E = output_dims['E']
		self.out_dim_y = output_dims['y']

		self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
									  nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

		self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
									  nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

		self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
									  nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

		self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
															de=hidden_dims['de'],
															dy=hidden_dims['dy'],
															n_head=hidden_dims['n_head'],
															dim_ffX=hidden_dims['dim_ffX'],
															dim_ffE=hidden_dims['dim_ffE'])
										for i in range(n_layers)])

		self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
									   nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

		# Note: we change the activation function here to sigmoid!
		self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), nn.Sigmoid(),
									   nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

		self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
									   nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

	def forward(self, X, E, y, node_mask):
		bs, n = X.shape[0], X.shape[1]

		diag_mask = torch.eye(n)
		diag_mask = ~diag_mask.type_as(E).bool()
		diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

		X_to_out = X[..., :self.out_dim_X]
		E_to_out = E[..., :self.out_dim_E]
		y_to_out = y[..., :self.out_dim_y]

		new_E = self.mlp_in_E(E)
		new_E = (new_E + new_E.transpose(1, 2)) / 2
		after_in = PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
		X, E, y = after_in.X, after_in.E, after_in.y

		for layer in self.tf_layers:
			X, E, y = layer(X, E, y, node_mask)

		X = self.mlp_out_X(X)
		E = self.mlp_out_E(E)
		y = self.mlp_out_y(y)

		X = (X + X_to_out)
		E = (E + E_to_out) * diag_mask
		y = y + y_to_out

		E = 1/2 * (E + torch.transpose(E, 1, 2))

		return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class DiGressGNN(nn.Module):

	def __init__(self, input_dim, t_limit):
		super().__init__()
		self.creation_args = {}
		self.t_limit = t_limit
		self.model = GraphTransformer(
			5,
			{'X': input_dim, 'E': 1, 'y': 1},
			{'X': 256, 'E': 128, 'y': 128},
			{'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
			{'X': input_dim, 'E': 1, 'y': 1},
			nn.ReLU(), nn.ReLU()
		)
		self.sigmoid = torch.nn.Sigmoid()
		self.bce_loss = torch.nn.BCELoss()

	def forward(self, data, t):
		# Add extra attribute
		data.edge_attr = torch.ones(data.edge_index.shape[1], 1, device=data.x.device)

		# Convert to proper format
		dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
		dense_data = dense_data.mask(node_mask)
		X, E = dense_data.X.float(), dense_data.E.float()

		# Encode time as y
		y = (t[data.ptr[:-1]] / self.t_limit)[:, None]

		# Extract out edge predictions
		pred_object = self.model(X, E, y, node_mask)
		edge_preds = pred_object.E[:, :, :, 0]  # Shape: B x V_max x V_max

		# Convert edge predictions into canonical tensor; do it the same way as
		# pyg_data_to_edge_vector
		graph_sizes = torch.diff(data.ptr)

		if torch.max(data.batch) >= len(graph_sizes):
			edge_preds = edge_preds[:len(graph_sizes)]
		
		# Create boolean mask of only the top upper triangle of each B x V x V
		# matrix, ignoring the diagonal
		triu_mask = torch.triu(torch.ones_like(edge_preds), diagonal=1) == 1
		for i, size in enumerate(torch.diff(data.ptr)):
			# For each individual graph, limit the upper-triangle mask to
			# only the size of that graph
			triu_mask[i, :, size:] = False
			
		# Edge vector is all entries in the graph-specific upper triangle of each
		# individual adjacency matrix
		edge_vec = edge_preds[triu_mask]

		# Finally, pass through a sigmoid
		return self.sigmoid(edge_vec)
	
	def loss(self, pred_probs, true_probs):
		return self.bce_loss(pred_probs, true_probs)

if __name__ == "__main__":
	import networkx as nx

	if torch.cuda.is_available():
		DEVICE = "cuda"
	else:
		DEVICE = "cpu"

	batch_size = 32
	num_nodes = 10
	node_dim = 5
	t_limit = 1

	# Prepare a data and time object just as the model would see
	edge_index = []
	for i in range(batch_size):
		edges = torch.tensor(list(nx.erdos_renyi_graph(num_nodes, 0.5).edges)) + (i * num_nodes)
		edge_index.append(edges)
		edge_index.append(torch.flip(edges, (1,)))
	edge_index = torch.concat(edge_index, dim=0)
	batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes, 0)
	ptr = torch.concat([
		torch.zeros(1, device=batch.device),
		torch.where(torch.diff(batch) != 0)[0] + 1,
		torch.tensor([len(batch)], device=batch.device)
	]).long()
	
	data = torch_geometric.data.Data(
		x=torch.rand((batch_size * num_nodes, node_dim)),
		edge_index=edge_index,
		batch=batch,
		ptr=ptr
	).to(DEVICE)
	t = torch.rand(len(data.x)).to(DEVICE)

	# Run the model
	model = DiGressGNN(node_dim, t_limit).to(DEVICE)
	edge_pred = model(data, t)
