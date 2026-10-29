import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset paths
input_file = Path("dataset") / "processed_reviews.csv"
output_folder = Path("results")

# Check that the processed dataset exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {input_file}"
    )

# Create a folder for results
output_folder.mkdir(exist_ok=True)

print("=" * 60)
print("PRODUCT REVIEW ANALYSIS FOR E-COMMERCE")
print("STEP 7: SENTIMENT DISTRIBUTION ANALYSIS")
print("=" * 60)

# Load only the columns needed for this analysis
df = pd.read_csv(
    input_file,
    usecols=["Score", "Sentiment"]
)

# Count reviews in each sentiment class
sentiment_counts = df["Sentiment"].value_counts()

# Calculate percentages
sentiment_percentages = (
    df["Sentiment"].value_counts(normalize=True) * 100
)

print("\n1. Total labelled reviews:", len(df))

print("\n2. Sentiment counts:")
print(sentiment_counts.to_string())

print("\n3. Sentiment percentages:")
for sentiment in ["Negative", "Positive"]:
    count = int(sentiment_counts.get(sentiment, 0))
    percentage = float(
        sentiment_percentages.get(sentiment, 0)
    )

    print(
        f"{sentiment}: {count:,} reviews "
        f"({percentage:.2f}%)"
    )

# Save the distribution as a CSV report
distribution = pd.DataFrame({
    "Sentiment": sentiment_counts.index,
    "Review_Count": sentiment_counts.values,
    "Percentage": [
        sentiment_percentages[label]
        for label in sentiment_counts.index
    ]
})

distribution.to_csv(
    output_folder / "sentiment_distribution.csv",
    index=False
)

# Create a bar chart
plot_order = ["Negative", "Positive"]
plot_counts = [
    int(sentiment_counts.get(label, 0))
    for label in plot_order
]

plt.figure(figsize=(8, 5))
bars = plt.bar(plot_order, plot_counts)

plt.title("Sentiment Distribution of Product Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.bar_label(
    bars,
    labels=[f"{count:,}" for count in plot_counts],
    padding=3
)

plt.tight_layout()

# Save the chart
chart_path = output_folder / "sentiment_distribution.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("\n4. Results saved successfully:")
print(output_folder / "sentiment_distribution.csv")
print(chart_path.resolve())

print("\nSentiment distribution analysis completed.")