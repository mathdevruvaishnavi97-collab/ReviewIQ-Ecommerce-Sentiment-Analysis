# ReviewIQ: E-commerce Product Review Sentiment Analysis

## Project Overview

ReviewIQ is a machine learning project that analyzes e-commerce product reviews and classifies customer feedback into three sentiment categories: Positive, Negative, and Neutral.

The project uses Natural Language Processing (NLP), Term Frequency–Inverse Document Frequency (TF-IDF), and machine learning algorithms to identify sentiment in customer reviews. It also compares multiple classification models using standard evaluation metrics to determine their performance.

The objective is to demonstrate how text analytics and machine learning can help businesses understand customer opinions and evaluate product feedback more efficiently.

## Objectives

- Develop a sentiment classification system for e-commerce product reviews.
- Apply text preprocessing and NLP techniques to customer feedback.
- Convert review text into numerical features using TF-IDF.
- Train and compare multiple machine learning classification algorithms.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Provide a foundation for interactive sentiment analysis through a Streamlit application.

## Key Features

- Three-class sentiment classification: Positive, Negative, and Neutral.
- Text preprocessing and feature extraction for review analysis.
- Machine learning model training and comparison.
- Evaluation using classification metrics and confusion matrices.
- Saved model artifacts for sentiment prediction.
- Streamlit-based interface for interactive analysis, subject to the implemented application features.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| TF-IDF | Text feature extraction |
| Joblib | Saving and loading trained models |
| Streamlit | Web application interface |
| Matplotlib | Visualization of results, where implemented |

## Dataset

The project uses the Amazon Fine Food Reviews dataset, which contains customer reviews and associated product ratings.

**Dataset details**

- Total records: 568,454
- Review text: Customer-written product feedback
- Rating scale: 1 to 5
- Prediction task: Three-class sentiment classification

### Sentiment Mapping

| Product Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

The dataset contains substantially more positive reviews than negative or neutral reviews. Macro-averaged metrics are therefore included to provide a more balanced view of classification performance across sentiment classes.

The full dataset and certain intermediate training files are excluded from this repository because of their size. They remain available locally for the training workflow and must be obtained separately to reproduce the complete experiment.

## Methodology

The project follows an end-to-end machine learning workflow.

1. **Data collection:** Load the Amazon Fine Food Reviews dataset.
2. **Data preparation:** Inspect the data and map ratings to sentiment labels.
3. **Text preprocessing:** Clean and prepare review text using the project's preprocessing pipeline.
4. **Feature extraction:** Use TF-IDF with unigram and bigram features to represent review text numerically.
5. **Data splitting:** Split the dataset into stratified training and testing sets.
6. **Model training:** Train Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine classifiers.
7. **Model evaluation:** Compare the trained models using accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.
8. **Sentiment prediction:** Apply the trained classifier and TF-IDF vectorizer to review text.
9. **Application integration:** Make sentiment analysis accessible through the Streamlit interface, according to the implemented features.

## Model Performance

The following results were recorded during model evaluation.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.65% | 78.73% | 68.88% | 72.06% |
| Multinomial Naive Bayes | 85.05% | 77.34% | 52.33% | 55.97% |
| Linear SVM | 89.89% | 79.18% | 69.73% | 72.64% |

### Best Performing Model

Linear SVM achieved the highest recorded accuracy of **89.89%** and the highest macro F1-score of **72.64%** among the three evaluated models.

Macro-averaged metrics calculate performance across classes with equal weight. They are useful for this dataset because the sentiment categories are imbalanced.

These figures represent the recorded evaluation run. Results may differ if the dataset, preprocessing, feature extraction, or training configuration changes.

## Project Structure

```text
ReviewIQ-Ecommerce-Sentiment-Analysis/
|
|-- app.py
|-- predict_sentiment.py
|-- preprocess_text.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- dataset/
|   |-- Dataset files stored locally
|
|-- models/
|   |-- logistic_regression_model.joblib
|   |-- naive_bayes_model.joblib
|   |-- svm_model.joblib
|   |-- tfidf_vectorizer.joblib
|
|-- results/
|   |-- model_comparison.csv
|   |-- model_comparison.png
|   |-- Model metrics and confusion matrices
```

This structure describes the intended organization of the project. Large datasets and training-only artifacts are excluded from version control. The actual repository contents may vary depending on the files committed.

## Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- pip
- Required project dependencies
- Trained model and vectorizer files required for prediction

### 1. Clone the repository

```bash
git clone https://github.com/mathdevruvaishnavi97-collab/ReviewIQ-Ecommerce-Sentiment-Analysis.git
```

### 2. Navigate to the project directory

```bash
cd ReviewIQ-Ecommerce-Sentiment-Analysis
```

### 3. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If the repository contains a `requirements.txt` file, install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

Use package versions compatible with the saved model artifacts, particularly Scikit-learn.

### 5. Run the application

If `app.py` is present and its dependencies and model files are available, start the application with:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the Streamlit application.

The application must be configured to load model files from the correct paths. The training dataset is not required for ordinary prediction unless the application explicitly loads it.

## Example

**Input review**

> The product quality is excellent, and I am very satisfied with my purchase.

**Illustrative output**

```text
Predicted Sentiment: Positive
```

The actual sentiment is determined by the trained model and may vary depending on the input text.

## Applications

- Customer feedback analysis
- E-commerce product review monitoring
- Customer satisfaction assessment
- Identification of negative product experiences
- Product improvement and feedback analysis
- Exploratory analysis of customer opinions

## Limitations

- Model performance depends on the training data and text preprocessing pipeline.
- Class imbalance can affect predictions for minority sentiment categories.
- Sarcasm, mixed sentiment, and context-dependent expressions may be difficult to classify.
- A sentiment prediction does not establish whether a review is factually accurate.
- Recorded evaluation metrics should be considered alongside class-wise performance.

## Future Enhancements

- Deploy the application to a public hosting platform.
- Add interactive sentiment distribution dashboards.
- Support batch analysis through CSV uploads.
- Improve minority-class performance through class weighting and hyperparameter tuning.
- Evaluate transformer-based NLP models.
- Add downloadable reports and review filtering.

## Repository

GitHub: [ReviewIQ-Ecommerce-Sentiment-Analysis](https://github.com/mathdevruvaishnavi97-collab/ReviewIQ-Ecommerce-Sentiment-Analysis)

Live application: To be added after successful deployment.

## Author

**Vaishnavi Mathdevru**

B.Tech in Computer Science and Engineering, specializing in Artificial Intelligence and Machine Learning.

Project domain: Natural Language Processing, Machine Learning, and Sentiment Analysis.

