# ML Classification — Linear Models & Neural Networks

## Overview
Comparative analysis of classification algorithms on two 
real-world datasets (Glass and Diabetes) using weighted F1 
as the evaluation metric.

## Models
- Perceptron
- Logistic Regression
- MLP (Multi-Layer Perceptron)
- RBF Kernel + Linear Models

## Results

**Diabetes Dataset**
- MLP (tuned): 0.776
- Logistic Regression: 0.736
- Perceptron (after RBF kernel): 0.712
- Perceptron (before kernel): 0.632

**Glass Dataset**
- MLP (tuned): 0.625
- Logistic Regression: 0.506
- Perceptron (after RBF kernel): 0.502
- Perceptron (before kernel): 0.403

MLP outperformed all linear models on both datasets.

## Techniques
- Missing value imputation
- MinMax normalization
- RBF kernel transformation
- GridSearchCV hyperparameter tuning
- 5-fold repeated stratified cross validation
- t-SNE visualization
- Feature importance via model coefficients

## Stack
Python · Scikit-learn · NumPy · Pandas · Matplotlib
