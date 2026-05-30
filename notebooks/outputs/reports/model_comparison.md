| Model         | Method   |   Accuracy |   F1-Macro |   Loss |   Train Time(m) |
|:--------------|:---------|-----------:|-----------:|-------:|----------------:|
| xlm-roberta   | LoRA     |     0.6596 |     0.6626 | 0.6428 |             4.4 |
| xlm-roberta   | Full FT  |     0.6552 |     0.6588 | 0.413  |            15.4 |
| mbert         | LoRA     |     0.6345 |     0.6339 | 0.7735 |             4.3 |
| mbert         | Full FT  |     0.6269 |     0.6283 | 0.5507 |            34   |
| distilbert-ml | LoRA     |     0.6222 |     0.6192 | 0.8348 |             2.2 |
| distilbert-ml | Full FT  |     0.5986 |     0.5978 | 0.4527 |             7.6 |