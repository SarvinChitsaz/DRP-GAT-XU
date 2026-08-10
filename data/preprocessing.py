import os
import pandas as pd
from sklearn.model_selection import train_test_split
from configs.config import RAW_DIR, PROCESSED_DIR, TOP_N_GENES, RANDOM_SEED


def load_and_clean_expression():
    gdsc_expr_raw = pd.read_csv(
        f"{RAW_DIR}/rnaseq_merged_rsem_tpm_20260323.csv", low_memory=False
    )

    gene_symbols = gdsc_expr_raw["model_id"].iloc[3:].values
    expression_values = gdsc_expr_raw.iloc[3:, 3:]
    expression_values.index = gene_symbols

    expression_values_dedup = expression_values.loc[
        ~expression_values.index.duplicated(keep="first")
    ]

    gdsc_expression_clean = expression_values_dedup.transpose()
    gdsc_expression_clean.index.name = "SANGER_MODEL_ID"
    gdsc_expression_clean = gdsc_expression_clean.reset_index()

    gene_columns_all = [
        col for col in gdsc_expression_clean.columns if col != "SANGER_MODEL_ID"
    ]
    gdsc_expression_clean[gene_columns_all] = gdsc_expression_clean[
        gene_columns_all
    ].apply(pd.to_numeric, errors="coerce")

    return gdsc_expression_clean, gene_columns_all


def build_dataset():
    gdsc2 = pd.read_excel(f"{RAW_DIR}/GDSC2_fitted_dose_response_27Oct23.xlsx")
    gdsc_expression_clean, gene_columns_all = load_and_clean_expression()

    common_cell_lines = set(gdsc2["SANGER_MODEL_ID"]) & set(
        gdsc_expression_clean["SANGER_MODEL_ID"]
    )
    unique_cell_lines_all = sorted(common_cell_lines)

    train_cell_lines, temp_cell_lines = train_test_split(
        unique_cell_lines_all, test_size=0.3, random_state=RANDOM_SEED
    )
    val_cell_lines, test_cell_lines = train_test_split(
        temp_cell_lines, test_size=0.5, random_state=RANDOM_SEED
    )

    train_expression_only = gdsc_expression_clean[
        gdsc_expression_clean["SANGER_MODEL_ID"].isin(train_cell_lines)
    ]
    gene_variance_train = (
        train_expression_only[gene_columns_all].var().sort_values(ascending=False)
    )
    top_genes = gene_variance_train.head(TOP_N_GENES).index.tolist()

    gdsc_expression_subset = gdsc_expression_clean[["SANGER_MODEL_ID"] + top_genes]
    final_dataset = gdsc2.merge(
        gdsc_expression_subset, on="SANGER_MODEL_ID", how="inner"
    )
    final_dataset["DRUG_NAME"] = final_dataset["DRUG_NAME"].astype(str)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    final_dataset.to_parquet(f"{PROCESSED_DIR}/final_dataset.parquet", index=False)

    return final_dataset, top_genes, train_cell_lines, val_cell_lines, test_cell_lines


def normalize_expression(final_dataset_with_smiles, top_genes, train_cell_lines):
    cell_line_expression = (
        final_dataset_with_smiles.drop_duplicates(subset="SANGER_MODEL_ID")
        .set_index("SANGER_MODEL_ID")[top_genes]
    )

    train_expression_values = cell_line_expression.loc[
        cell_line_expression.index.isin(train_cell_lines)
    ]
    cell_line_expression = cell_line_expression.fillna(train_expression_values.mean())

    expression_mean = train_expression_values.values.mean()
    expression_std = train_expression_values.values.std()
    cell_line_expression_normalized = (
        cell_line_expression - expression_mean
    ) / expression_std

    return cell_line_expression_normalized, expression_mean, expression_std
