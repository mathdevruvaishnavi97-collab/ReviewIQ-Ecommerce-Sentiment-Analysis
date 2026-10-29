
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
input_file = Path("dataset") / "preprocessed_reviews.csv"
model_folder = Path("models")
results_folder = Path("results")

model_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 10: TRAIN-TEST SPLIT AND TF-IDF")
print("=" * 60)

if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

# Load the cleaned reviews and sentiment labels
print("\nLoading preprocessed reviews...")

df = pd.read_csv(
    input_file,
    usecols=["Cleaned_Text", "Sentiment"]
)

df = df.dropna(subset=["Cleaned_Text", "Sentiment"])
df["Cleaned_Text"] = df["Cleaned_Text"].astype(str)
df = df[df["Cleaned_Text"].str.strip() != ""]

# Convert sentiment labels into numbers
# 0 = Negative, 1 = Positive, 2 = Neutral
df["Sentiment_Label"] = df["Sentiment"].map({
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2
})

df = df.dropna(subset=["Sentiment_Label"])
df["Sentiment_Label"] = df["Sentiment_Label"].astype("int8")

print("Total reviews:", f"{len(df):,}")

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts().to_string())

# Split reviews before learning the TF-IDF vocabulary
train_text, test_text, y_train, y_test = train_test_split(
    df["Cleaned_Text"],
    df["Sentiment_Label"],
    test_size=0.20,
    random_state=42,
    stratify=df["Sentiment_Label"]
)

print("\nDataset split completed.")
print("Training reviews:", f"{len(train_text):,}")
print("Testing reviews:", f"{len(test_text):,}")

print("\nTraining sentiment distribution:")
print(y_train.map({
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}).value_counts().to_string())

print("\nTesting sentiment distribution:")
print(y_test.map({
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}).value_counts().to_string())

# Learn vocabulary and IDF weights from training reviews only
print("\nFitting TF-IDF on training reviews...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype="float32"
)

X_train = vectorizer.fit_transform(train_text)
print("Training TF-IDF completed.")

# Apply the same fitted vectorizer to the testing reviews
print("Transforming testing reviews...")
X_test = vectorizer.transform(test_text)

print("\nFeature extraction completed.")
print("Training matrix shape:", X_train.shape)
print("Testing matrix shape:", X_test.shape)
print("Training matrix format:", X_train.format)
print("Testing matrix format:", X_test.format)

# Save training and testing data
print("\nSaving files...")

joblib.dump(
    (X_train, y_train.to_numpy()),
    model_folder / "train_data.joblib",
    compress=3
)

joblib.dump(
    (X_test, y_test.to_numpy()),
    model_folder / "test_data.joblib",
    compress=3
)

joblib.dump(
    vectorizer,
    model_folder / "tfidf_vectorizer.joblib"
)

pd.DataFrame({
    "Feature": vectorizer.get_feature_names_out()
}).to_csv(
    results_folder / "tfidf_feature_names.csv",
    index=False
)

print("\nSaved successfully:")
print(model_folder / "train_data.joblib")
print(model_folder / "test_data.joblib")
print(model_folder / "tfidf_vectorizer.joblib")
print(results_folder / "tfidf_feature_names.csv")

print("\nStep 10 completed successfully.")
