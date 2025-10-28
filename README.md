#### 🧠 K-Nearest Neighbors (KNN) Classifier — Iris Dataset

📄 Overview

This project implements a K-Nearest Neighbors (KNN) classifier to predict flower species using the Iris dataset.
It demonstrates end-to-end machine learning — including data preprocessing, model training, evaluation, and visualization — and automatically generates a PDF report with key insights and plots.

🚀 How It Works
1. Data Preprocessing (preprocess_data)

Loads the dataset using pandas and removes the Id column.

Encodes the categorical Species labels into numeric format using LabelEncoder.

Scales numerical features using StandardScaler to normalize feature ranges.

Splits the dataset into training (80%) and testing (20%) sets using train_test_split.

2. Model Training and Evaluation (train_evaluate)

Initializes a KNN classifier with 4 neighbors and default Minkowski distance.

Trains the model and predicts test data.

Calculates:

Accuracy

Confusion Matrix

Classification Report

Generates a multi-page PDF report with:

Model parameters and text summary

Confusion matrix heatmap

PCA-based 2D decision boundary visualization for multiclass data

3. Run the Pipeline

To execute the project:

python execute.py


The report will be automatically saved to:

reports/KNN_Classifier_report.pdf

📊 Example Output

Each generated PDF includes:

Model performance metrics

Confusion matrix visualization

PCA-based 2D decision boundary

Parameters and dataset insights

#### 🧠 KNN — Interview Questions & Answers

1. How does the KNN algorithm work?

K-Nearest Neighbors is a non-parametric, instance-based learning algorithm.
It classifies a data point by looking at the K nearest points in the feature space and assigns the majority class among them.
The distance between points is usually computed using Euclidean or Minkowski distance.

2. How do you choose the right K?

Small K (e.g., 1 or 3): Model becomes sensitive to noise and may overfit.

Large K (e.g., 10+): Model becomes smoother but may underfit.

The optimal value of K is usually determined using cross-validation or error-rate vs K plotting.

3. Why is normalization important in KNN?

KNN relies on distance calculations.
If features have different scales (e.g., centimeters vs kilograms), large-valued features dominate the distance metric.
Normalization or standardization ensures all features contribute equally to the distance measure, improving accuracy.

4. What is the time complexity of KNN?

Training: O(1) — KNN is a lazy learner, meaning it stores the data without explicit model fitting.

Prediction: O(n × d) — where n is the number of training samples and d is the number of features.
For every prediction, KNN computes the distance from the test point to all training samples.

5. What are the pros and cons of KNN?

| Pros                              | Cons                                         |
| --------------------------------- | -------------------------------------------- |
| Simple and easy to implement      | Computationally expensive for large datasets |
| No training phase required        | Sensitive to noise and irrelevant features   |
| Works well with small datasets    | High memory usage                            |
| Naturally handles multiclass data | Requires feature scaling                     |

6. Is KNN sensitive to noise?

Yes. Since KNN uses local neighbor votes, any noisy or mislabeled samples in the training data can negatively influence predictions.
Using larger K values or applying data cleaning techniques can reduce this effect.

7. How does KNN handle multi-class problems?

KNN supports multi-class classification naturally.
When predicting a new sample, it simply takes the majority vote among neighbors, regardless of how many classes exist in the dataset.

8. What’s the role of distance metrics in KNN?

Distance metrics define how “closeness” between samples is measured.
Common ones include:

Euclidean distance — default for continuous data

Manhattan distance — better for grid-like data

Minkowski distance — generalization of both

Cosine similarity — used for text or sparse data
The choice of metric can significantly affect model performance.

#### 🧰 Dependencies

Install all dependencies with:

pip install pandas numpy matplotlib seaborn scikit-learn

#### 📘 Summary

This project:

=> Demonstrates a full ML pipeline using KNN

=> Automates preprocessing, scaling, and visualization

=> Generates professional PDF reports

=> Includes conceptual understanding suitable for interview preparation