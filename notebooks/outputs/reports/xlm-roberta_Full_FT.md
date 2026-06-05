# Classification Report: xlm-roberta (Full FT)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.6552 |
| F1-Macro | 0.6588 |
| Eval Loss | 0.4130 |
| Training Time | 15m 24s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | xlm-roberta-base |
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

    negative     0.6104    0.8361    0.7056      3972
     neutral     0.7784    0.4780    0.5923      5937
    positive     0.5912    0.7958    0.6784      2375

    accuracy                         0.6552     12284
   macro avg     0.6600    0.7033    0.6588     12284
weighted avg     0.6879    0.6552    0.6456     12284
```
