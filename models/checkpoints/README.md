# Model Checkpoints

Trained model weights are not included directly in this repository due to file size limitations.

## Available Checkpoints

| File | Description |
|---|---|
| `drug_response_model_gat.ckpt` | Main GAT model (random split), Test R²=0.768 |
| `drug_response_model_domain_baseline.ckpt` | Domain generalization baseline (unseen tissue), Test R²=0.706 |
| `drug_response_model_domain_adversarial.ckpt` | Domain-adversarial variant (unseen tissue), Test R²=0.707 |

## Download

Download from Google Drive: `[ADD_LINK_HERE]`

After downloading, place each file in this directory.

## Loading a Checkpoint

```python
import torch
from models.model import DrugResponseModelGAT

model = DrugResponseModelGAT(atom_feature_dim=20, gene_feature_dim=1000, hidden_dim=128)
checkpoint = torch.load("models/checkpoints/drug_response_model_gat.ckpt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
