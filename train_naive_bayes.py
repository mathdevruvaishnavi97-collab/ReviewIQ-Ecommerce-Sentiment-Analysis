
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

model_folder = Path("models")
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

train_file = model_folder / "train_data.joblib"
test_file = model_folder / "test_data.joblib"

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 12: NAIVE BAYES")
print("=" * 60)

if not train_file.exists() or not test_file.exists():
    raise FileNotFoundError(
        "Training or testing data is missing. "
        "Run split_and_extract_features.py first."
    )

print("\nLoading training and testing data...")

X_train, y_train = joblib.load(train_file)
X_test, y_test = joblib.load(test_file)

print("Training samples:", f"{X_train.shape[0]:,}")
print("Testing samples:", f"{X_test.shape[0]:,}")

# Sentiment label mapping
# 0 = Negative, 1 = Positive, 2 = Neutral
class_labels = [0, 1, 2]
class_names = ["Negative", "Positive", "Neutral"]

print("\nClasses used: Negative, Positive, Neutral")
print("\nCreating Multinomial Naive Bayes model...")

model = MultinomialNB(alpha=1.0)

print("Training started. Please wait...")
model.fit(X_train, y_train)
print("Training completed.")

print("\nPredicting testing reviews...")
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

# Macro averaging gives equal weight to all three classes
precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\nMODEL EVALUATION RESULTS")
print("-" * 40)
print(f"Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1-score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
)

# Save the trained model
model_path = model_folder / "naive_bayes_model.joblib"
joblib.dump(model, model_path, compress=3)

# Save evaluation metrics
metrics = pd.DataFrame([{
    "Model": "Naive Bayes",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics_path = results_folder / "naive_bayes_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# Create and save three-class confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(values_format="d")
plt.title("Naive Bayes - Three-Class Confusion Matrix")
plt.tight_layout()

chart_path = results_folder / "naive_bayes_confusion_matrix.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nFiles saved successfully:")
print("Model:", model_path.resolve())
print("Metrics:", metrics_path.resolve())
print("Confusion matrix:", chart_path.resolve())

print("\nStep 12 completed successfully.")
