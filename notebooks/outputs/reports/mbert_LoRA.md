# Classification Report: mbert (LoRA)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.6345 |
| F1-Macro | 0.6339 |
| Eval Loss | 0.7735 |
| Training Time | 4m 15s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | bert-base-multilingual-cased |
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

    negative     0.5868    0.7747    0.6678      3972
     neutral     0.7169    0.5306    0.6098      5937
    positive     0.5922    0.6598    0.6242      2375

    accuracy                         0.6345     12284
   macro avg     0.6320    0.6550    0.6339     12284
weighted avg     0.6507    0.6345    0.6313     12284
```
