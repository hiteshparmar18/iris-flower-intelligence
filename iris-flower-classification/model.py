import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# MODEL STORAGE
# --------------------------------------------------
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# TRAIN & SAVE MODELS
# --------------------------------------------------
def train_and_save_models():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(
            max_iter=300,
            multi_class="auto"
        )
    }

    accuracies = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, f"{name}.pkl")
        )

    # Save test data for evaluation (confusion matrix)
    joblib.dump(
        (X_test, y_test),
        os.path.join(MODEL_DIR, "test_data.pkl")
    )

    return accuracies, iris.target_names, iris.feature_names

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
def load_models():
    models = {
        "KNN": joblib.load(os.path.join(MODEL_DIR, "KNN.pkl")),
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "Logistic Regression.pkl")
        )
    }

    X_test, y_test = joblib.load(
        os.path.join(MODEL_DIR, "test_data.pkl")
    )

    return models, X_test, y_test
