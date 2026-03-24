# MLflow Python Quick Start & Guidelines

## What is MLflow?
MLflow is an open-source platform to manage the machine learning lifecycle. The most important component is **MLflow Tracking**, which lets you record and compare parameters, metrics, and models from your training runs. 

Quick start: https://mlflow.org/docs/latest/ml/getting-started/quickstart/

---

## 1. Installation

Install MLflow via pip. It's recommended to do this within your project's virtual environment.

```bash
pip install mlflow
```

---

## 2. Quick Start: Manual Logging

The core workflow involves starting a "run", logging your parameters (inputs), logging your metrics (outputs/performance), and saving your model.

Here is a basic example using `scikit-learn`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Generate some dummy data
X = np.random.rand(100, 2)
y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Set an experiment name (Groups runs together)
mlflow.set_experiment("My_First_MLflow_Project")

# 3. Start an MLflow run
with mlflow.start_run(run_name="Random_Forest_Run"):
    
    # Define parameters
    n_estimators = 50
    max_depth = 5
    
    # Log parameters to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    
    # Log the model itself
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Run completed. MSE: {mse}")
```

---

## 3. The "Easy Way": Autologging

If you are using popular frameworks (like Scikit-learn, XGBoost, TensorFlow, PyTorch, etc.), MLflow has an `autolog` feature that automatically logs parameters, metrics, and models without needing manual `log_param` or `log_metric` calls.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

mlflow.set_experiment("Autologging_Experiment")

with mlflow.start_run():
    # Just train your model normally! 
    # MLflow will automatically capture parameters, metrics, and the model.
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Metrics are automatically logged when you call score or predict (depending on the framework)
    score = model.score(X_test, y_test)
```

---

## 4. Viewing Your Results (The MLflow UI)

Once you've run your Python script, MLflow saves the data locally in an `mlruns` folder. To view your "lab notebook":

1. Open your terminal.
2. Navigate to the folder where you ran your Python script (in our case most likely `notebooks`).
3. Run the following command:

```bash
mlflow ui
```

4. Open your web browser and go to `http://127.0.0.1:5000`. You will see the MLflow dashboard where you can compare your runs, view charts, and download your saved models.

5. If you want to close ML Flow UI in terminal press `Ctrl` + `C` (`control` + `C` on MacOS)

---

## 5. Guidelines and Best Practices

* **Group by Experiments:** Use `mlflow.set_experiment("Name")` to group related runs. Don't dump everything into the "Default" experiment.
* **Name Your Runs:** Pass a `run_name` to `mlflow.start_run()` (e.g., "baseline_model" or "tuned_xgboost"). It makes finding specific runs much easier later.
* **Log Artifacts:** You aren't limited to just metrics and parameters. You can save plots, CSV files, or images using `mlflow.log_artifact("path/to/file.png")`.
* **Track Your Environment:** MLflow automatically logs a `conda.yaml` or `requirements.txt` with your model so you know exactly what versions of libraries were used to train it. Always utilize `log_model()` so this is captured.
* **Don't Log Everything Manually:** Rely on `autolog()` where possible to keep your code clean, and only use manual logging for custom metrics specific to your business logic.
