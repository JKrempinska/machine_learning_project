# Classification Report: mbert (Full FT)

## Summary
| Metric | Value |
|--------|-------|
| Accuracy | 0.6269 |
| F1-Macro | 0.6283 |
| Eval Loss | 0.5507 |
| Training Time | 34m 2s |

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | bert-base-multilingual-cased |
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

    negative     0.6013    0.7525    0.6685      3972
     neutral     0.7226    0.4992    0.5905      5937
    positive     0.5444    0.7360    0.6259      2375

    accuracy                         0.6269     12284
   macro avg     0.6227    0.6626    0.6283     12284
weighted avg     0.6489    0.6269    0.6225     12284
```
