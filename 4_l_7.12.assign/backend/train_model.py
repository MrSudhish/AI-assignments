# backend/train_model.py
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset_loader import load_dataset

# ========== CONFIG ==========
DATASET_PATH = "C:/Users/mrsud/Desktop/AI/dataset"   # your dataset path
MODELS_DIR = "models"  # saved under backend/models
RANDOM_STATE = 42
# ==========================

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def print_sep():
    print("-" * 60)

def main():
    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)
    print("Loaded:", X.shape, y.shape)

    # Encode labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", le.classes_)

    # Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Train / Val / Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    # Optional: scale features (helps many models)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    ensure_dir(MODELS_DIR)

    # --- Model 1: K-Nearest Neighbors (baseline) ---
    print_sep(); print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_s, y_train)
    y_val_pred = knn.predict(X_val_s)
    print("KNN Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    joblib.dump(knn, os.path.join(MODELS_DIR, "knn_model.joblib"))
    print("Saved:", os.path.join(MODELS_DIR, "knn_model.joblib"))

    # --- Model 2: Logistic Regression (multinomial) ---
    print_sep(); print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="saga", n_jobs=-1)
    lr.fit(X_train_s, y_train)
    y_val_pred = lr.predict(X_val_s)
    print("Logistic Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    joblib.dump(lr, os.path.join(MODELS_DIR, "logreg_model.joblib"))
    print("Saved:", os.path.join(MODELS_DIR, "logreg_model.joblib"))

    # --- Model 3: Gaussian Naive Bayes ---
    print_sep(); print("Training GaussianNB...")
    gnb = GaussianNB()
    gnb.fit(X_train_s, y_train)
    y_val_pred = gnb.predict(X_val_s)
    print("GNB Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    joblib.dump(gnb, os.path.join(MODELS_DIR, "gnb_model.joblib"))
    print("Saved:", os.path.join(MODELS_DIR, "gnb_model.joblib"))

    # --- Evaluate best model on Test set (choose by validation accuracy) ---
    print_sep(); print("Evaluating on TEST set...")

    # Quick selection based on val accuracy (you could pick manually)
    models = {
        "knn": knn,
        "logreg": lr,
        "gnb": gnb
    }
    val_accuracies = {}
    for name, m in models.items():
        acc = accuracy_score(y_val, m.predict(X_val_s))
        val_accuracies[name] = acc

    best_name = max(val_accuracies, key=val_accuracies.get)
    best_model = models[best_name]
    print("Best on validation:", best_name, val_accuracies)

    # Test set metrics
    y_test_pred = best_model.predict(X_test_s)
    print(f"TEST Accuracy for {best_name}:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    # Save metadata: label encoder + scaler
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("Saved label encoder and scaler.")

    print("All done.")

if __name__ == "__main__":
    main()
