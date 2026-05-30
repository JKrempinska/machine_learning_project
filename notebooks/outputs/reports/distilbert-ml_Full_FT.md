# Classification Report: distilbert-ml (Full FT)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.5986 |
| F1-Macro | 0.5978 |
| Eval Loss | 0.4527 |
| Training Time | 7m 34s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | distilbert-base-multilingual-cased |
| Method | Full FT |
| MAX_LENGTH | 64 |
| BATCH_SIZE | 32 |
| NUM_EPOCHS | 3 |
| LEARNING_RATE | 2e-05 |
| TRAIN_SUBSAMPLE | 0.25 |
| Train samples | 11403 |

## Detailed Classification Report
```
              precision    recall  f1-score   support

    negative     0.5414    0.8321    0.6560      3972
     neutral     0.7315    0.4208    0.5342      5937
    positive     0.5606    0.6526    0.6031      2375

    accuracy                         0.5986     12284
   macro avg     0.6112    0.6352    0.5978     12284
weighted avg     0.6370    0.5986    0.5869     12284
```
