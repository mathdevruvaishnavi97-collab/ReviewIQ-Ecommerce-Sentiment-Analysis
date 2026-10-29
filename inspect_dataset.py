import pandas as pd
from pathlib import Path

# Step 1: Define the dataset path
file_path = Path("dataset") / "Reviews.csv"

# Step 2: Check whether the file exists
if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 5: DATASET INSPECTION")
print("=" * 60)

# Step 3: Display the column names
columns = pd.read_csv(file_path, nrows=0).columns.tolist()

print("\n1. Dataset column names:")
print(columns)

# Step 4: Load the first five rows
sample = pd.read_csv(file_path, nrows=5)

print("\n2. First five rows:")
print(sample.to_string(index=False))

# Step 5: Display dataset structure
print("\n3. Dataset information:")
sample.info()

# Step 6: Inspect important columns
print("\n4. Review text and ratings:")
print(sample[["Text", "Score"]].to_string(index=False))

# Step 7: Inspect the rating values in the full dataset
print("\n5. Checking the complete dataset in chunks...")

total_rows = 0
missing_text = 0
missing_score = 0
rating_counts = {}

for chunk in pd.read_csv(
    file_path,
    usecols=["Text", "Score"],
    chunksize=50000
):
    total_rows += len(chunk)

    missing_text += chunk["Text"].isna().sum()
    missing_score += chunk["Score"].isna().sum()

    counts = chunk["Score"].value_counts(dropna=False)

    for rating, count in counts.items():
        rating_counts[rating] = (
            rating_counts.get(rating, 0) + int(count)
        )

print("\n6. Complete dataset summary:")
print("Total reviews:", total_rows)
print("Missing review texts:", missing_text)
print("Missing ratings:", missing_score)

print("\n7. Rating distribution:")
for rating, count in sorted(
    rating_counts.items(),
    key=lambda item: str(item[0])
):
    print(f"Rating {rating}: {count} reviews")

print("\nDataset inspection completed successfully.")