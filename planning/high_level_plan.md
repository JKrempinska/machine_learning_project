## 👥 Suggested Role Distribution (3 People)

1. **Lead Data Engineer:** Responsible for text preprocessing (tokenization, lemmatization), handling the unlabeled data (unsupervised pre-training), and GitHub repo structure.
2. **Modeler (Deep Learning):** Responsible for the "Advanced Techniques" (Transformers/BERT), hyperparameter tuning, and MLflow integration.
3. **Analyst/Communicator:** Responsible for the baseline models (TF-IDF + XGBoost), **Interpretability (SHAP/LIME)**, and the final presentation/notebook documentation.

---

## 📅 Semester Timeline & Milestones

### Phase 1: Infrastructure & Exploration (Weeks 1-4)

* **Setup:** Create the GitHub repo using the structure provided in your guide. Set up the `requirements.txt` with `transformers`, `datasets`, `mlflow`, and `shap`.
* **Data Ingestion:** Write a script to script to parse the `pos/neg` folders into a clean Pandas DataFrame.
* **EDA:** Analyze review lengths, word clouds for sentiment, and distribution of star ratings (the `id_rating.txt` filenames).

### Phase 2: Baselines & MLOps Setup (Weeks 5-8)

* **Technique #1 (Baseline):** Implement a TF-IDF vectorizer + Logistic Regression or XGBoost.
* **MLOps:** Connect this pipeline to **MLflow**. Log the N-gram range and $C$ (regularization) parameters.
* **Unlabeled Data:** Start exploring the 50,000 unlabeled reviews. Use them for "Unsupervised Pre-training" (e.g., Word2Vec or fine-tuning a masked language model).

### Phase 3: Advanced Modeling (Weeks 9-12)

* **Technique #2 (Deep Learning):** Fine-tune a pre-trained Transformer like **BERT** or **RoBERTa** using the `HuggingFace` library.
* **Technique #3 (Advanced/Hybrid):** Implement **Aspect-Based Sentiment Analysis** (identifying sentiment toward specific actors/directors) or an **Ensemble** (stacking the BERT predictions with XGBoost features).
* **Interpretability:** Apply **SHAP (Partition Explainer)** to the Transformer model to see which specific words (e.g., "masterpiece" vs. "disaster") triggered the classification.

### Phase 4: Finalization & Delivery (Weeks 13-15)

* **Validation:** Ensure the code is reproducible. Run the "Restart & Run All" test on your Jupyter Notebook.
* **Presentation:** Create slides showing the SHAP visualizations.
* **Submission:** Ensure everything is pushed to GitHub 48 hours before your date.

---

## 🛠 Project-Specific Technical Strategy

### Addressing the "3 Advanced Techniques" Requirement:

To ensure you hit the 20% "Use of advanced techniques" grade:

1. **Transfer Learning:** Fine-tuning a `distilbert-base-uncased` model.
2. **Semi-Supervised Learning:** Using the 50k unlabeled reviews to create domain-specific word embeddings.
3. **Integrated Gradients/SHAP:** Using advanced attribution methods specifically designed for NLP to explain model "attention."

### Interpretation Tip:

Since you are using movie reviews, your **SHAP** analysis will be very visual. You can produce "Force Plots" that highlight words in red (positive) or blue (negative) directly within the text of a review. This makes for a fantastic "Live Demo."

> **Warning on Data Leakage:** Ensure you do not use the star rating (found in the filename) as a feature in your model, as that is directly derived from the label you are trying to predict!
