
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
input_file = Path("dataset") / "preprocessed_reviews.csv"
output_folder = Path("results")
model_folder = Path("models")

output_folder.mkdir(exist_ok=True)
model_folder.mkdir(exist_ok=True)

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 9: TF-IDF FEATURE EXTRACTION")
print("=" * 60)

if not input_file.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found: {input_file}"
    )

# Use unigrams and bigrams.
# Limit vocabulary size to control memory usage.
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype="float32"
)

print("\nReading cleaned reviews...")

df = pd.read_csv(
    input_file,
    usecols=["Cleaned_Text", "Sentiment"]
)

df["Cleaned_Text"] = df["Cleaned_Text"].fillna("").astype(str)

# Remove empty reviews, if any
df = df[df["Cleaned_Text"].str.strip() != ""].reset_index(drop=True)

print("Reviews available for TF-IDF:", f"{len(df):,}")
print("\nLearning vocabulary and converting text to numbers...")

X = vectorizer.fit_transform(df["Cleaned_Text"])
y = df["Sentiment"].map({"Negative": 0, "Positive": 1}).to_numpy()

print("\nTF-IDF extraction completed.")
print("Number of reviews:", f"{X.shape[0]:,}")
print("Number of TF-IDF features:", f"{X.shape[1]:,}")
print("Matrix shape:", X.shape)
print("Matrix format:", X.format)
print("Matrix data type:", X.dtype)

# Save the TF-IDF vectorizer
vectorizer_path = model_folder / "tfidf_vectorizer.joblib"
joblib.dump(vectorizer, vectorizer_path)

# Save the sparse feature matrix and sentiment labels
features_path = model_folder / "tfidf_features.joblib"
joblib.dump((X, y), features_path, compress=3)

# Save feature names for inspection
feature_names = vectorizer.get_feature_names_out()
feature_path = output_folder / "tfidf_feature_names.csv"

pd.DataFrame({"Feature": feature_names}).to_csv(
    feature_path,
    index=False
)

print("\nSaved files:")
print("Vectorizer:", vectorizer_path.resolve())
print("Feature matrix:", features_path.resolve())
print("Feature names:", feature_path.resolve())

print("\nSample TF-IDF features:")
for word, value in sorted(
    zip(feature_names, X[0].toarray()[0]),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{word}: {value:.4f}")

print("\nStep 9 completed successfully.")
