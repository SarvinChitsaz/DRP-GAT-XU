import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class DrugEncoderGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return x


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x


class SimpleFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, drug_vec, cell_vec):
        combined = torch.cat([drug_vec, cell_vec], dim=1)
        return F.relu(self.fc(combined))


class DrugResponseModelGAT(nn.Module):
    def __init__(self, atom_feature_dim, gene_feature_dim, hidden_dim=128):
        super().__init__()
        self.drug_encoder = DrugEncoderGAT(atom_feature_dim, hidden_dim)
        self.cell_encoder = CellLineEncoder(gene_feature_dim, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, drug_x, drug_edge_index, drug_batch, cell_expression):
        drug_vec = self.drug_encoder(drug_x, drug_edge_index, drug_batch)
        cell_vec = self.cell_encoder(cell_expression)
        fused = self.fusion(drug_vec, cell_vec)
        prediction = self.output_layer(fused)
        return prediction.squeeze(-1)


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
