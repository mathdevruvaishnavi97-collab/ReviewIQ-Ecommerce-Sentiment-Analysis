
import re
import pandas as pd
from pathlib import Path

# Input and output files
input_file = Path("dataset") / "processed_reviews.csv"
output_file = Path("dataset") / "preprocessed_reviews.csv"

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 8: TEXT PREPROCESSING")
print("=" * 60)

# Check whether the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {input_file}"
    )

# Common English words to remove.
# Important sentiment words such as not, no, and never are preserved.
stop_words = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "as", "it", "its", "this",
    "that", "these", "those", "and", "or", "but", "if",
    "then", "than", "so", "very", "just", "have", "has",
    "had", "do", "does", "did", "will", "would", "could",
    "should", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "their", "what", "which",
    "who", "when", "where", "how"
}

def preprocess_text(text):
    """Clean a customer review while preserving sentiment words."""

    # Handle missing values safely
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove web addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Split into words and remove selected common words
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words with single spaces
    return " ".join(words)


# Read the dataset in chunks to reduce memory usage
chunk_size = 25000
total_reviews = 0

with pd.read_csv(
    input_file,
    usecols=["Text", "Score", "Sentiment"],
    chunksize=chunk_size
) as reader:

    first_chunk = True

    for chunk in reader:
        # Save a few original examples for verification
        if first_chunk:
            print("\nExample reviews before preprocessing:")
            for review in chunk["Text"].head(3):
                print("-", str(review)[:200])
            first_chunk = False

        # Clean each review
        chunk["Cleaned_Text"] = chunk["Text"].apply(
            preprocess_text
        )

        # Keep the original text, cleaned text, score, and sentiment
        chunk = chunk[
            ["Text", "Cleaned_Text", "Score", "Sentiment"]
        ]

        # Write the first chunk with headers; append the rest
        chunk.to_csv(
            output_file,
            mode="w" if total_reviews == 0 else "a",
            header=(total_reviews == 0),
            index=False
        )

        total_reviews += len(chunk)

print("\nText preprocessing completed.")
print("Total reviews processed:", f"{total_reviews:,}")
print("Output file:", output_file.resolve())
print("Output file exists:", output_file.exists())

# Display examples from the saved dataset
sample_df = pd.read_csv(output_file, nrows=3)

print("\nExamples after preprocessing:")
for _, row in sample_df.iterrows():
    print("\nOriginal:", str(row["Text"])[:200])
    print("Cleaned :", str(row["Cleaned_Text"])[:200])

# Verify that sentiment words are preserved
test_review = "This product is NOT good. I will never buy it!"
print("\nPreprocessing test:")
print("Original:", test_review)
print("Cleaned :", preprocess_text(test_review))
