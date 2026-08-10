import torch
from configs.config import MC_DROPOUT_SAMPLES

def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def predict_with_uncertainty(model, drug_x, drug_edge_index, drug_batch, expression, n_samples=MC_DROPOUT_SAMPLES):
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(drug_x, drug_edge_index, drug_batch, expression)
            predictions.append(pred.item())

    predictions = torch.tensor(predictions)
    mean_prediction = predictions.mean().item()
    std_prediction = predictions.std().item()

    return mean_prediction, std_prediction


def predict_sensitivity_probability_smooth(model, drug_x, drug_edge_index, drug_batch, expression, threshold, n_samples=MC_DROPOUT_SAMPLES, temperature=1.0):
    mean_prediction, std_prediction = predict_with_uncertainty(
        model, drug_x, drug_edge_index, drug_batch, expression, n_samples
    )

    distance = (threshold - mean_prediction) / temperature
    prob_sensitive = torch.sigmoid(torch.tensor(distance)).item()

    return mean_prediction, std_prediction, prob_sensitive
