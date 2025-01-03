# Breast Cancer Diagnosis Using Logistic Regression

Overview

This project implements logistic regression to diagnose breast cancer based on tumor measurements. The dataset used contains features computed from digitized images of breast masses. The goal is to classify tumors as either malignant (M) or benign (B).

Project Structure

# 1. Dataset

Source: The dataset contains 569 samples and 33 columns.

Key Columns:

diagnosis: Target column (M = malignant, B = benign).

30 numerical features: Measurements of the tumor (e.g., radius_mean, texture_mean).

Irrelevant Columns:

id: Identifier column.

Unnamed: 32: Contains no useful data.



# 2. Implementation Steps

* a. Loading Libraries

We use the following libraries:

numpy for mathematical operations.

pandas for handling data.

matplotlib.pyplot for visualizations.

* b. Data Preprocessing

Load Dataset: Read the CSV file using pandas.

Inspect Data: Understand the structure and identify irrelevant columns.

* Clean Data:

Drop id and Unnamed: 32 columns.

Convert diagnosis into numerical values:

Malignant (M) = 1

Benign (B) = 0

Split Data:

Separate the features (x_data) and the target (y).

Normalize the features to a range of [0, 1].

Split data into training (85%) and testing (15%) subsets.

* c. Logistic Regression Implementation

We implemented logistic regression from scratch using Python:

Initialize Weights and Bias: Start with small initial values.

Sigmoid Function: Converts linear outputs into probabilities.

* Forward and Backward Propagation:

Forward: Calculate predictions and loss.

Backward: Compute gradients to update weights and bias.

Parameter Updates: Iterate over multiple steps to reduce the error (cost function).

Prediction Function: Classify samples based on a probability threshold of 0.5.

* d. Model Training and Evaluation

Train the logistic regression model using the training data.

Evaluate its performance using metrics:

Training accuracy.

Testing accuracy.

* e. Verification with Scikit-Learn

To validate our implementation, we use Scikit-Learn’s LogisticRegression model to compare results.

# 3. Results

Custom Logistic Regression

Train accuracy: ~80.75%

Test accuracy: ~81.39%

Scikit-Learn Logistic Regression

Train accuracy: ~86.33%

Test accuracy: ~89.53%

# 4. Files in the Project

data.csv: The dataset containing tumor measurements and diagnoses.

breast_cancer_diagnosis.py: Python script implementing logistic regression.

README.md: Documentation for the project.

# 5. How to Run the Project

Install required libraries:

pip install numpy pandas matplotlib scikit-learn

Run the Python script:

python breast_cancer_diagnosis.py

# 6. Acknowledgments

The dataset is publicly available and often used for machine learning projects to benchmark algorithms for binary classification tasks.

# 7. Future Improvements

Hyperparameter tuning to improve model performance.

Explore more advanced algorithms like Support Vector Machines or Neural Networks.

Perform additional feature engineering and analysis.

