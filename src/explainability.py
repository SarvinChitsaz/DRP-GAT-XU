import torch
from configs.config import DEVICE

def compute_gene_importance(model, drug_graphs, cell_line_expression, gene_columns, drug_name, cell_line_id, top_n=10):
    graph = drug_graphs[drug_name]
    expression = torch.tensor(
        cell_line_expression.loc[cell_line_id].values, dtype=torch.float
    ).unsqueeze(0)
    expression_grad = expression.clone().requires_grad_(True)
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

    model.eval()
    prediction = model(graph.x, graph.edge_index, batch, expression_grad)
    prediction.backward()

    importance = expression_grad.grad.abs().squeeze(0)

    import pandas as pd
    importance_df = pd.DataFrame({
        "gene": gene_columns,
        "importance": importance.tolist(),
    }).sort_values("importance", ascending=False).head(top_n)

    return prediction.item(), importance_df
