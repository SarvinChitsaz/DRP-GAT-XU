import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class DrugResponseDataset(Dataset):
    def __init__(self, dataframe, drug_graphs, cell_line_expression):
        self.data = dataframe.reset_index(drop=True)
        self.drug_graphs = drug_graphs
        self.cell_line_expression = cell_line_expression

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_graph = self.drug_graphs[row["DRUG_NAME"]]
        expression = torch.tensor(
            self.cell_line_expression.loc[row["SANGER_MODEL_ID"]].values,
            dtype=torch.float,
        )
        label = torch.tensor(row["LN_IC50"], dtype=torch.float)
        return drug_graph, expression, label


def collate_fn(batch):
    graphs = [item[0] for item in batch]
    expressions = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    drug_batch = Batch.from_data_list(graphs)
    return drug_batch, expressions, labels


class DomainAdversarialDataset(Dataset):
    def __init__(self, dataframe, drug_graphs, cell_line_expression, domain_to_idx):
        self.data = dataframe.reset_index(drop=True)
        self.drug_graphs = drug_graphs
        self.cell_line_expression = cell_line_expression
        self.domain_to_idx = domain_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_graph = self.drug_graphs[row["DRUG_NAME"]]
        expression = torch.tensor(
            self.cell_line_expression.loc[row["SANGER_MODEL_ID"]].values,
            dtype=torch.float,
        )
        label = torch.tensor(row["LN_IC50"], dtype=torch.float)
        domain_label = torch.tensor(
            self.domain_to_idx[row["CANCER_TYPE"]], dtype=torch.long
        )
        return drug_graph, expression, label, domain_label


def domain_collate_fn(batch):
    graphs = [item[0] for item in batch]
    expressions = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    domain_labels = torch.stack([item[3] for item in batch])
    drug_batch = Batch.from_data_list(graphs)
    return drug_batch, expressions, labels, domain_labels
