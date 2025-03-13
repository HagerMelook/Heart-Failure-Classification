# Pattern Recognition Assignment 1

## Heart Failure Classification Using Machine Learning

## Table of Contents

- [Dataset and Preparation](#1-dataset-and-preparation)
  - [Data Description](#11-data-description)
  - [Preprocessing Steps](#12-preprocessing-steps)
- [Implementation Details](#2-implementation-details)
  - [Decision Tree](#21-decision-tree)
  - [Bagging Ensemble](#22-bagging-ensemble)
  - [AdaBoost Ensemble](#23-adaboost-ensemble)
- [Evaluation and Results](#3-evaluation-and-results)
  - [Performance Metrics](#31-performance-metrics)
  - [Results and Visualizations](#32-results-and-visualizations)
- [Analysis and Discussion](#4-analysis-and-discussion)
- [Bonus Implementations](#5-bonus-implementations)
  - [K-Nearest Neighbors (KNN)](#51-k-nearest-neighbors-knn-classifier)
  - [Logistic Regression](#52-logistic-regression)
  - [Feedforward Neural Network (FNN)](#53-feedforward-neural-network-fnn)
- [Conclusion](#6-conclusion)

## Overview

Heart failure is a critical condition often resulting from cardiovascular diseases. The early prediction of heart failure can help improve patient outcomes through timely medical intervention. This report explores the application of machine learning classifiers to predict heart failure based on clinical attributes.

---

## 1. Dataset and Preparation

### 1.1 Data Description

- **Clinical data:** Includes age, cholesterol levels, resting blood pressure, and maximum heart rate.
- **Categorical variables:** Sex, chest pain type, resting ECG results, exercise-induced angina, and ST slope type.
- **Binary targets:** 0 = No heart disease, 1 = Heart disease.

### 1.2 Preprocessing Steps

1. Load the heart disease dataset from CSV file into IDE using `read_csv()`.
2. Split the dataset into features and target using `drop()`.
3. Encode the categorical features using one-hot encoding with `get_dummies()`.
4. Split the dataset into training (70%), validation (10%), and test sets (20%) using `train_test_split()`.
5. Normalize the dataset using `StandardScaler.fit_transform()` (if needed).

---

## 2. Implementation Details

### 2.1 Decision Tree

- **Decision tree node:** Stores feature index, threshold, entropy, sample count, class distribution, and left/right children.
- **Decision tree structure:** Implements a tree-based model where each node represents a decision.
- **Entropy and information gain:** Used to evaluate splits and maximize information gain.
- **Recursive tree building:** Constructs the tree until max depth or minimum samples are met.
- **Prediction mechanism:** Traverses the tree to make predictions.
- **Hyperparameter tuning:** Adjusts max depth and min samples split to balance complexity and generalization.

### 2.2 Bagging Ensemble

#### 2.2.1 Without Decision Tree Hyperparameter Tuning

1. **Bagging training:** Resamples the dataset and trains multiple decision trees.
2. **Hyperparameter tuning:** Evaluates model accuracy on validation data and selects the best model.
3. **Evaluation:** Tests the final model on the test dataset.

#### 2.2.2 With Decision Tree Hyperparameter Tuning

- Same as above, but tunes max depth for each resample.

### 2.3 AdaBoost Ensemble

- **Decision Stump:** Selects the best feature and threshold to minimize weighted classification error.
- **AdaBoost training:** Iteratively trains decision stumps and updates sample weights.
- **Prediction:** Combines weak learners' weighted predictions.
- **Hyperparameter tuning:** Optimizes the number of weak learners.

---

## 3. Evaluation and Results

### 3.1 Performance Metrics

- **Accuracy:** Measures overall correctness of predictions.
- **F1-Score:** Balances precision and recall.
- **Confusion Matrix:** Summarizes model performance with counts of true/false positives and negatives.

### 3.2 Results and Visualizations

| Model                 | Accuracy | F1-Score |
|----------------------|----------|----------|
| Decision Tree        | 0.8423   | 0.85     |
| Bagging             | 0.8695   | 0.88     |
| AdaBoost            | 0.7826   | 0.8      |
| Logistic Regression | 0.89     | 0.9      |
| KNN                 | 0.8858   | 0.89     |
| FNN                 | 0.86     | 0.88     |

---

## 4. Analysis and Discussion

- **Best models:** KNN and Logistic Regression with accuracy ~88-89% and lowest false negatives.
- **FNN and Bagging:** Performed well; FNN captures complex patterns, Bagging reduces overfitting.
- **Decision Tree:** Prone to overfitting with higher false negatives.
- **AdaBoost:** Lowest accuracy (78.26%), highest false negatives (22), sensitive to noise.

---

## 5. Bonus Implementations

### 5.1 K-Nearest Neighbors (KNN) Classifier

- **Model training:** Uses Minkowski metric (p=2 for Euclidean distance).
- **Hyperparameter tuning:** Optimizes `k` for best accuracy.
- **Evaluation:** Final model tested on the test dataset.

### 5.2 Logistic Regression

- **Training:** Normalizes features, applies logistic regression.
- **Hyperparameter tuning:** Adjusts regularization strength and penalty type.

### 5.3 Feedforward Neural Network (FNN)

- **Architecture:** Input layer, one hidden layer (ReLU), output layer (Sigmoid).
- **Compilation:** Uses Adam optimizer and binary cross-entropy loss.
- **Training process:** Batch gradient descent with validation and callbacks.
- **Early stopping:** Stops training when validation accuracy stagnates.
- **Learning rate reduction:** Adjusts learning rate when validation accuracy plateaus.

---

## 6. Conclusion

- **Bagging:** Reduced overfitting and improved decision tree performance.
- **AdaBoost:** Performed the worst due to high sensitivity to noise.
- **KNN and Logistic Regression:** Achieved the highest accuracy and lowest false negatives.

---
