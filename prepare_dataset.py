
import pandas as pd
from pathlib import Path

# Project dataset paths
input_file = Path("dataset") / "Reviews.csv"
output_file = Path("dataset") / "processed_reviews.csv"

# Check whether the original dataset exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Original dataset not found: {input_file}"
    )

# Remove an old output file if the script is run again
if output_file.exists():
    output_file.unlink()

# Counters for the dataset summary
total_rows = 0
missing_text_rows = 0
empty_text_rows = 0
neutral_rows = 0
negative_rows = 0
positive_rows = 0
saved_rows = 0

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 6: PREPARING SENTIMENT LABELS")
print("=" * 60)

# Process the original dataset in manageable chunks
for chunk in pd.read_csv(
    input_file,
    usecols=["Text", "Score"],
    chunksize=50000
):
    total_rows += len(chunk)

    # Count rows with missing review text
    missing_text_rows += int(chunk["Text"].isna().sum())

    # Remove rows with missing review text or rating
    chunk = chunk.dropna(subset=["Text", "Score"]).copy()

    # Remove empty review text
    chunk["Text"] = chunk["Text"].astype(str).str.strip()

    empty_mask = chunk["Text"].eq("")
    empty_text_rows += int(empty_mask.sum())
    chunk = chunk.loc[~empty_mask].copy()

    # Convert ratings into numeric values
    chunk["Score"] = pd.to_numeric(
        chunk["Score"], errors="coerce"
    )

    chunk = chunk.dropna(subset=["Score"]).copy()

    # Count neutral three-star reviews
    neutral_rows += int(chunk["Score"].eq(3).sum())

    # Create three sentiment labels
    chunk["Sentiment"] = chunk["Score"].map(
        {
            1: "Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Positive"
        }
    )

    # Keep only the columns needed for this case study
    chunk = chunk.dropna(subset=["Sentiment"])
    chunk = chunk[["Text", "Score", "Sentiment"]]

    # Count sentiment labels
    negative_rows += int(
        (chunk["Sentiment"] == "Negative").sum()
    )
    neutral_count = int(
        (chunk["Sentiment"] == "Neutral").sum()
    )
    positive_rows += int(
        (chunk["Sentiment"] == "Positive").sum()
    )

    # Write each chunk to the output CSV
    chunk.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not output_file.exists()
        or output_file.stat().st_size == 0
    )

    saved_rows += len(chunk)

    print(f"Processed {total_rows:,} original rows...")

print("\nDATASET PREPARATION COMPLETED")
print("-" * 40)
print("Original reviews:", f"{total_rows:,}")
print("Missing review texts:", f"{missing_text_rows:,}")
print("Empty review texts:", f"{empty_text_rows:,}")
print("Neutral reviews included:", f"{neutral_rows:,}")
print("Negative reviews saved:", f"{negative_rows:,}")
print("Positive reviews saved:", f"{positive_rows:,}")
print("Total labelled reviews:", f"{saved_rows:,}")
print("Output file:", output_file.resolve())
print("Output file exists:", output_file.exists())
