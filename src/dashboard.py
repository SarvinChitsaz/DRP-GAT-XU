import torch
import pandas as pd
from src.uncertainty import predict_sensitivity_probability_smooth

def build_patient_dashboard(model, patient_cell_line, drug_graphs, cell_line_expression, drug_thresholds, temperature, n_samples=30):
    all_drug_names = list(drug_graphs.keys())
    patient_expr = torch.tensor(
        cell_line_expression.loc[patient_cell_line].values, dtype=torch.float
    ).unsqueeze(0)

    rows = []
    for drug_name in all_drug_names:
        graph = drug_graphs[drug_name]
        batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
        drug_threshold = drug_thresholds.get(drug_name, drug_thresholds.median())

        mean_pred, std_pred, prob_sens = predict_sensitivity_probability_smooth(
            model, graph.x, graph.edge_index, batch, patient_expr, drug_threshold, n_samples, temperature
        )

        rows.append({
            "drug": drug_name,
            "probability_sensitive": prob_sens,
            "uncertainty_std": std_pred,
            "predicted_LN_IC50": mean_pred,
        })

    dashboard_df = pd.DataFrame(rows).sort_values("probability_sensitive", ascending=False)
    return dashboard_df


def validate_top1_recommendations(model, patient_ids, test_df, drug_graphs, cell_line_expression, drug_thresholds, temperature, n_samples=20):
    results = []

    for patient_id in patient_ids:
        patient_cancer = test_df[test_df["SANGER_MODEL_ID"] == patient_id]["CANCER_TYPE"].iloc[0]
        ranking = build_patient_dashboard(
            model, patient_id, drug_graphs, cell_line_expression, drug_thresholds, temperature, n_samples
        )
        top_drug = ranking.iloc[0]["drug"]

        actual_data = test_df[
            (test_df["SANGER_MODEL_ID"] == patient_id) & (test_df["DRUG_NAME"] == top_drug)
        ]
        if len(actual_data) > 0:
            actual_ln_ic50 = actual_data["LN_IC50"].iloc[0]
            threshold = drug_thresholds.get(top_drug, drug_thresholds.median())
            was_sensitive = bool(actual_ln_ic50 < threshold)
        else:
            was_sensitive = None

        results.append({
            "patient": patient_id,
            "cancer_type": patient_cancer,
            "top_recommended_drug": top_drug,
            "predicted_probability": ranking.iloc[0]["probability_sensitive"],
            "actually_sensitive": was_sensitive,
        })

    return pd.DataFrame(results)
