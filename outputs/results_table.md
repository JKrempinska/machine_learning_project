# Sentiment Analysis — All Models Results
**Dataset:** cardiffnlp/tweet_eval (SemEval-2017)  
**Sorted by:** Accuracy (descending)

| Model | Method | Framework | Accuracy | Loss | Train (sec) |
|:------|:-------|:----------|:--------:|-----:|------------:|
| XLM-RoBERTa | LoRA | Transformers | 0.6596 | 0.6428 | 261 |
| XLM-RoBERTa | Full FT | Transformers | 0.6539 | 0.3948 | 913 |
| mBERT | LoRA | Transformers | 0.6345 | 0.7735 | 262 |
| Bi-LSTM | TensorFlow/Keras | TensorFlow/Keras | 0.6370 | 0.7839 | 102 |
| mBERT | Full FT | Transformers | 0.6269 | 0.5507 | 706 |
| DistilBERT-ML | LoRA | Transformers | 0.6222 | 0.8348 | 131 |
| DistilBERT-ML | Full FT | Transformers | 0.5973 | 0.4528 | 449 |
| ED-LSTM | TensorFlow/Keras | TensorFlow/Keras | 0.4558 | 0.9996 | 54 |
| LSTM | TensorFlow/Keras | TensorFlow/Keras | 0.4531 | 0.9859 | 52 |
