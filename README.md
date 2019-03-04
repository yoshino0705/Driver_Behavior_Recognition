# Driver Behavior Recognition

This project mainly focuses on evaluating the performance of some driving dataset 
using several traditional machine learning models.

In addition, this project also experiments with restricting the number of features to be included for training,
because often times not all features within a dataset contribute to the overall performance.

The default models are:
- Decision Tree
- Random Forest
- K Nearest Neighbors
- Multi-Layer Perceptrons
- Logistic Regression
- Gradient Boosting
- Linear Support Vector Machine
- AdaBoost
- Naive Bayes

# Dataset
The dataset used is from this research paper: https://arxiv.org/abs/1704.05223

Data are collected using On Board Diagnostics 2 (OBD-II) and processed from In-vehicle Controller Area Network protocol (CAN bus)


# Results

[Test Accuracies on Nine Models](https://plot.ly/~yoshino0705/15)

[Train and Test Accuracies on Nine Models](https://plot.ly/~yoshino0705/13)

# Acknowledgements
The models are provided by the Python Sci-kit Learn Packages

This project evaluates and ranks features within the dataset using https://github.com/WillKoehrsen/feature-selector
