import torch
from sklearn.metrics import r2_score
from configs.config import DEVICE

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in test_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            predictions = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    r2 = r2_score(all_labels, all_predictions)
    return r2, all_predictions, all_labels


def evaluate_domain_adversarial(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, expressions, labels in test_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            predictions, _ = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    r2 = r2_score(all_labels, all_predictions)
    return r2, all_predictions, all_labels
