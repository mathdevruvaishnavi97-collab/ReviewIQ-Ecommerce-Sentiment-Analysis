
import re
import joblib
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load the trained three-class Linear SVM model
model_path = MODEL_DIR / "svm_model.joblib"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        f"Trained SVM model not found: {model_path}. "
        "Run train_svm.py first."
    )

if not vectorizer_path.exists():
    raise FileNotFoundError(
        f"TF-IDF vectorizer not found: {vectorizer_path}. "
        "Run split_and_extract_features.py first."
    )

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Preserve important sentiment words such as "not" and "never".
# Keep this list consistent with preprocess_text.py.
STOP_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "with", "and", "or", "but", "it", "this",
    "that", "i", "we", "you", "he", "she", "they",
    "my", "our", "your", "their", "as", "by", "from"
}


def clean_review(text):
    """Clean a new review using the training pipeline's basic rules."""
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove selected stopwords while preserving sentiment words
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)


def predict_sentiment(review):
    """Predict Negative, Neutral, or Positive sentiment."""
    cleaned_review = clean_review(review)

    if not cleaned_review:
        print("Please enter a review containing meaningful text.")
        return

    # Convert the review into TF-IDF features
    features = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = int(model.predict(features)[0])

    sentiment_labels = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }

    sentiment = sentiment_labels.get(
        prediction,
        "Unknown"
    )

    print("\nReview:", review)
    print("Cleaned review:", cleaned_review)
    print("Predicted sentiment:", sentiment)


if __name__ == "__main__":
    print("=" * 50)
    print("PRODUCT REVIEW SENTIMENT PREDICTION")
    print("Using the trained three-class Linear SVM model")
    print("Classes: Negative | Neutral | Positive")
    print("=" * 50)

    while True:
        review = input(
            "\nEnter a product review (or type 'exit' to stop): "
        ).strip()

        if review.lower() == "exit":
            print("Prediction program closed.")
            break

        predict_sentiment(review)
