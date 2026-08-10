import os
import pickle
import torch
from torch.utils.data import DataLoader

from configs.config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    ATOM_FEATURE_DIM,
    GENE_FEATURE_DIM,
    HIDDEN_DIM,
    DEVICE,
)
from data.preprocessing import build_dataset, normalize_expression
from data.dataset import DrugResponseDataset, collate_fn
from models.model import DrugResponseModelGAT
from src.train import train_model
from src.eval import evaluate_model
from src.metrics import bootstrap_r2_ci
from src.visualize import plot_model_comparison


def run_pipeline():
    final_dataset, top_genes, train_cell_lines, val_cell_lines, test_cell_lines = build_dataset()

    with open(f"{PROCESSED_DIR}/drug_graphs.pkl", "rb") as f:
        drug_graphs = pickle.load(f)
    drug_graphs = {str(k): v for k, v in drug_graphs.items()}

    valid_drugs = set(drug_graphs.keys())
    final_dataset_with_smiles = final_dataset[final_dataset["DRUG_NAME"].isin(valid_drugs)]

    cell_line_expression_normalized, _, _ = normalize_expression(
        final_dataset_with_smiles, top_genes, train_cell_lines
    )

    train_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(train_cell_lines)]
    val_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(val_cell_lines)]
    test_df = final_dataset_with_smiles[final_dataset_with_smiles["SANGER_MODEL_ID"].isin(test_cell_lines)]

    train_dataset = DrugResponseDataset(train_df, drug_graphs, cell_line_expression_normalized)
    val_dataset = DrugResponseDataset(val_df, drug_graphs, cell_line_expression_normalized)
    test_dataset = DrugResponseDataset(test_df, drug_graphs, cell_line_expression_normalized)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = DrugResponseModelGAT(
        atom_feature_dim=ATOM_FEATURE_DIM,
        gene_feature_dim=GENE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    model, optimizer, train_losses, val_losses = train_model(model, train_loader, val_loader)

    r2, predictions, labels = evaluate_model(model, test_loader)
    ci_lower, ci_upper = bootstrap_r2_ci(predictions, labels)

    print(f"Test R2: {r2:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "gene_feature_dim": GENE_FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "test_r2": r2,
        "test_r2_ci": (ci_lower, ci_upper),
    }
    torch.save(checkpoint, f"{CHECKPOINT_DIR}/drug_response_model_gat.ckpt")

    os.makedirs(f"{RESULTS_DIR}/model_comparison", exist_ok=True)
    plot_model_comparison(
        ["GAT (random split)"],
        [r2],
        ["seagreen"],
        f"{RESULTS_DIR}/model_comparison/model_comparison.png",
    )


if __name__ == "__main__":
    run_pipeline()
