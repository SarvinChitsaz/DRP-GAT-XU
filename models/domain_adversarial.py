import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.model import DrugEncoderGAT, CellLineEncoder, SimpleFusion


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim, num_domains, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_domains)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class DrugResponseModelDomainAdversarial(nn.Module):
    def __init__(self, atom_feature_dim, gene_feature_dim, num_domains, hidden_dim=128, alpha=1.0):
        super().__init__()
        self.drug_encoder = DrugEncoderGAT(atom_feature_dim, hidden_dim)
        self.cell_encoder = CellLineEncoder(gene_feature_dim, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.grl = GradientReversalLayer(alpha=alpha)
        self.domain_classifier = DomainClassifier(hidden_dim, num_domains)

    def forward(self, drug_x, drug_edge_index, drug_batch, cell_expression):
        drug_vec = self.drug_encoder(drug_x, drug_edge_index, drug_batch)
        cell_vec = self.cell_encoder(cell_expression)
        fused = self.fusion(drug_vec, cell_vec)
        prediction = self.output_layer(fused).squeeze(-1)
        reversed_fused = self.grl(fused)
        domain_pred = self.domain_classifier(reversed_fused)
        return prediction, domain_pred
