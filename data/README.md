# Dataset

This project uses the GDSC2 release of the Genomics of Drug Sensitivity in Cancer database.

Dataset ownership, citation requirements, and license terms belong to the original data provider.

## Source

https://www.cancerrxgene.org/downloads/bulk_download

## Required Files

Place the following files in `data/raw/gdsc/`:

```
data/raw/gdsc/
├── GDSC2_fitted_dose_response_27Oct23.xlsx
├── rnaseq_merged_rsem_tpm_20260323.csv
└── model_list_20260709.csv
```

## Preprocessing

Running `data/preprocessing.py` will:

1. Clean and transpose the raw expression matrix.
2. Split cell lines into train/validation/test (70/15/15) at the cell-line level.
3. Select the 1,000 most variable genes using train-only statistics.
4. Normalize expression using train-only mean/standard deviation.
5. Save the resulting dataset to `data/processed/`.

Gene selection and normalization statistics are computed strictly from the training split to avoid data leakage into validation/test.

## Dataset Composition

- 245 drugs with resolved molecular structure (SMILES)
- 944 cancer cell lines with matched RNA-seq expression
- 206,145 drug–cell line response pairs across 42 cancer types

## Output

After preprocessing, the following file is created:

```
data/processed/final_dataset.parquet
```

This file contains the merged drug response and gene expression data used for training, validation, and testing.
