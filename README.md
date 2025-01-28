
# Cardio Disease Predictor

## Overview

This project aims to predict the likelihood of cardiovascular diseases in individuals using various machine learning models. The models used for prediction include:

- **Gaussian Naive Bayes**
- **Logistic Regression**
- **XGBoost**
- **Decision Tree**
- **Random Forest**
- **Support Vector Classifier (SVC)**
- **Multi-layer Perceptron (MLP)**
- **LightGBM**

At the end of the project, the performance of all models is compared using the **ROC Curve** to assess their effectiveness in predicting cardiovascular diseases.

## Features

- **Preprocessing:** The dataset undergoes preprocessing, including feature selection, data cleaning, and normalization.
- **Models:** A variety of popular models are used to predict the probability of cardiovascular diseases.
- **Comparison:** All models are compared using ROC curve, which helps to evaluate the classification performance.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cardio-disease-predictor.git
   cd cardio-disease-predictor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Dataset**: Download the dataset and place it in the `data/` directory. (Ensure the dataset is in a format compatible with the models.)
   
2. **Training the Model**:
   ```bash
   python train.py
   ```

   This will train all the models on the dataset and display the ROC curves for comparison.

3. **Evaluate Performance**:
   The ROC curves will be plotted for each model to allow for easy comparison of their performances.

## Models Implemented

### 1. Gaussian Naive Bayes (GaussianNB)
A probabilistic classifier based on Bayes' theorem, assuming feature independence.

### 2. Logistic Regression
A linear classifier that models the relationship between the features and the probability of an outcome.

### 3. XGBoost
An optimized gradient boosting algorithm that is highly effective for classification problems.

### 4. Decision Tree
A non-linear classifier that splits the data into regions based on feature values.

### 5. Random Forest
An ensemble method that uses multiple decision trees for improved prediction accuracy.

### 6. Support Vector Classifier (SVC)
A powerful classifier that separates the data into classes using hyperplanes.

### 7. Multi-layer Perceptron (MLP)
A deep learning model composed of multiple layers of neurons, capable of learning non-linear relationships.

### 8. LightGBM
A gradient boosting model optimized for efficiency and performance on large datasets.

## Results

After training, the performance of each model is compared using the ROC curve. The model with the highest area under the curve (AUC) is considered the best for predicting cardiovascular diseases.

## Conclusion

This project demonstrates the power of various machine learning models in solving the problem of cardiovascular disease prediction. Through comparison using ROC curves, we can effectively identify the best model for the task at hand.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

