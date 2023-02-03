import pandas as pd
import rdkit.Chem
import torch
import torch_geometric
import networkx as nx

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


ZINC250K_PATH = "/gstore/home/tsenga5/discrete_graph_diffusion/data/250k_rndm_zinc_drugs_clean_3.csv"


def smiles_to_networkx(smiles):
	"""
	Converts a SMILES string to a NetworkX graph. The graph will retain the
	atomic number and bond type for nodes and edges (respectively), under the
	keys `atomic_num` and `bond_type` (respectively).
	Arguments:
		`smiles`: a SMILES string
	Returns a NetworkX graph.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	g = nx.Graph()
	for atom in mol.GetAtoms():
		g.add_node(
			atom.GetIdx(),
			atomic_num=atom.GetAtomicNum()
		)
	for bond in mol.GetBonds():
		g.add_edge(
			bond.GetBeginAtomIdx(),
			bond.GetEndAtomIdx(),
			bond_type=bond.GetBondType()
		)
	return g


ATOM_MAP = torch.tensor([6, 7, 8, 9, 16, 17, 35, 53, 15])
BOND_MAP = torch.tensor([1, 2, 3, 12])

def smiles_to_pyg_data(smiles, ignore_edge_attr=False):
	"""
	Converts a SMILES string to a torch-geometric Data object. The data object
	will have node attributes and edge attributes under `x` and `edge_attr`,
	respectively.
	Arguments:
		`smiles`: a SMILES string
		`ignore_edge_attr`: if True, no edge attributes will be included
	Returns a torch-geometric Data object.
	"""
	g = smiles_to_networkx(smiles)
	data = torch_geometric.utils.from_networkx(g)
	
	# Set atom features
	atom_inds = torch.argmax(
		(data.atomic_num.view(-1, 1) == ATOM_MAP).int(), dim=1
	)
	data.x = torch.nn.functional.one_hot(atom_inds, num_classes=len(ATOM_MAP))

	if not ignore_edge_attr:
		# Set bond features
		# For aromatic bonds, set them to be both single and double
		aromatic_mask = data.bond_type == BOND_MAP[-1]
		bond_inds = torch.argmax(
			(data.bond_type.view(-1, 1) == BOND_MAP).int(), dim=1
		)
		bond_inds[aromatic_mask] = 0
		data.edge_attr = torch.nn.functional.one_hot(
			bond_inds, num_classes=(len(BOND_MAP) - 1)
		)
		data.edge_attr[aromatic_mask, 1] = 1
	
	del data.atomic_num
	del data.bond_type
	
	return data


class ZINCDataset(torch.utils.data.Dataset):
	def __init__(self, connectivity_only=False):
		"""
		Create a PyTorch IterableDataset which yields random graphs.
		Arguments:
			`connectivity_only`: if True, only connectivity information is
				retained, and no edge attributes will be included
		"""
		super().__init__()

		self.connectivity_only = connectivity_only
		self.node_dim = len(ATOM_MAP)
		
		zinc_table = pd.read_csv(ZINC250K_PATH, sep=",", header=0)
		zinc_table["smiles"] = zinc_table["smiles"].str.strip()
		self.all_smiles = zinc_table["smiles"]	

	def __getitem__(self, index):
		"""
		Returns the torch-geometric Data object representing the molecule at
		index `index` in `self.all_smiles`.
		"""
		data = smiles_to_pyg_data(self.all_smiles[index])
		data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
		return data.to(DEVICE)

	def __len__(self):
		return len(self.all_smiles)


if __name__ == "__main__":
	dataset = ZINCDataset(connectivity_only=True)
	data_loader = torch_geometric.loader.DataLoader(
		dataset, batch_size=32, shuffle=False
	)
	batch = next(iter(data_loader))
