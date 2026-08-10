import torch
from rdkit import Chem
from torch_geometric.data import Data

POSSIBLE_ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"]


def one_hot_encode(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features(atom):
    features = one_hot_encode(atom.GetSymbol(), POSSIBLE_ATOMS)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    features.append(int(atom.IsInRing()))
    hybridization = one_hot_encode(
        str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    )
    features.extend(hybridization)
    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
