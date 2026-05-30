# Classification Report: xlm-roberta (LoRA)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.6596 |
| F1-Macro | 0.6626 |
| Eval Loss | 0.6428 |
| Training Time | 4m 24s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | xlm-roberta-base |
| Method | LoRA |
| MAX_LENGTH | 64 |
| BATCH_SIZE | 32 |
| NUM_EPOCHS | 3 |
| LEARNING_RATE | 2e-05 |
| TRAIN_SUBSAMPLE | 0.25 |
| Train samples | 11403 |

## Detailed Classification Report
```
              precision    recall  f1-score   support

    negative     0.6083    0.8593    0.7123      3972
     neutral     0.7852    0.4784    0.5945      5937
    positive     0.6050    0.7785    0.6809      2375

    accuracy                         0.6596     12284
   macro avg     0.6662    0.7054    0.6626     12284
weighted avg     0.6931    0.6596    0.6493     12284
```
