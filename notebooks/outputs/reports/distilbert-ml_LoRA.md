# Classification Report: distilbert-ml (LoRA)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.6222 |
| F1-Macro | 0.6192 |
| Eval Loss | 0.8348 |
| Training Time | 2m 11s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | distilbert-base-multilingual-cased |
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

    negative     0.5932    0.7082    0.6456      3972
     neutral     0.6867    0.5575    0.6154      5937
    positive     0.5584    0.6400    0.5964      2375

    accuracy                         0.6222     12284
   macro avg     0.6128    0.6352    0.6192     12284
weighted avg     0.6317    0.6222    0.6215     12284
```
