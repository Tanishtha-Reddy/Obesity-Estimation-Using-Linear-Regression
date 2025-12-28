# Obesity-Estimation-Using-Linear-Regression
## Overview
This project implements a **Linear Regression model** to predict obesity levels based on eating habits and physical conditions using the UCI Obesity Dataset.

## Dataset
- **Source**: UCI Machine Learning Repository (Dataset ID: 544)
- **Samples**: 2,111 individuals
- **Features**: 16 features (age, height, weight, physical activity, diet habits, etc.)
- **Target**: Obesity level (7 classes: Insufficient Weight, Normal Weight, Overweight Level I/II, Obesity Type I/II/III)

## Mathematical Formulation

### Linear Regression Model
The model learns to predict obesity level using:

**ŷ = w₁·x₁ + w₂·x₂ + w₃·x₃ + ... + w₁₆·x₁₆ + b**

Where:
- **ŷ**: Predicted obesity level
- **w₁, w₂, ..., w₁₆**: Learned weights for each feature
- **b**: Bias term
- **x₁, x₂, ..., x₁₆**: Input features

### Cost Function (Mean Squared Error)
**J(w, b) = (1/n) Σ(ŷᵢ - yᵢ)²**

The algorithm minimizes this error to find optimal weights.

### Normal Equation (Closed-form Solution)
**w* = (X^T X)^(-1) X^T y**

This directly computes optimal weights without iteration.

## Project Steps

### 1. Data Loading & Exploration
- Download obesity dataset from UCI. Name of the dataset is "Estimation of Obesity Levels Based On Eating Habits and Physical Condition"
- Explore data structure, features, target variable
- Check for missing values (there are none from the information provided)

### 2. Data Preparation
- **Encoding**: Convert categorical variables (Male/Female, yes/no) to numbers using LabelEncoder
- **Separation**: Split into features (X) and target (y)

### 3. Train-Test Split
- 80% training data (1,688 samples) for learning
- 20% test data (423 samples) for evaluation
- Ensures the model is tested on unseen data

### 4. Feature Scaling
- Normalize features using StandardScaler
- Converts each feature to mean=0, std=1
- Improves model performance

### 5. Model Training
- Create Linear Regression model using sklearn
- Fit on training data
- Model learns optimal weights for each feature

### 6. Model Evaluation
Metrics calculated on test set:
- **R² Score**: 0.6345 (explains 63.45% of variance)
- **RMSE**: 1.48 (average prediction error)
- **MAE**: 1.22 (mean absolute error)

### 7. Visualization
- **Plot 1**: Predicted vs Actual values
- **Plot 2**: Residual plot (error distribution)
- **Plot 3**: Feature importance ranking
- **Plot 4**: Error distribution histogram

## Results & Visualizations

### Model Performance Plots
The following visualization shows the model's performance across 4 different plots:

<img width="1065" height="630" alt="obesityVisualization" src="https://github.com/user-attachments/assets/c63e46a0-b189-4fdc-8be8-8df7847e9889" />

**Plot Explanations:**
1. **Top-Left (Predicted vs Actual)**: Points should follow red diagonal line. Shows model accuracy.
2. **Top-Right (Residual Plot)**: Shows prediction errors. Diagonal pattern indicates non-linear relationships.
3. **Bottom-Left (Feature Importance)**: Weight is most important feature, followed by eating habits.
4. **Bottom-Right (Error Distribution)**: Bell-shaped distribution centered at 0 is ideal.

### Model Performance 
The following shows the model's performance metrics after scaling:

<img width="492" height="220" alt="obesityMetrics" src="https://github.com/user-attachments/assets/f98b354c-3442-4203-b20d-dec4655b9582" />

```
Test Set Metrics:
- R² Score: 0.2897 
- RMSE: 1.65 obesity levels
- MAE: 1.38 obesity levels
```

### Key Findings

**Top 3 Most Important Features (Increase Obesity):**
1. Weight (+0.50)
2. CAEC - Eating between meals (+0.30)
3. Age (+0.20)

**Top 3 Protective Factors (Decrease Obesity):**
1. CALC - Alcohol consumption (-0.15)
2. Height (-0.12)
3. MTRANS - Transportation mode (-0.10)

### Observations
- Model explains 63% of obesity variance
- Linear relationship captures most patterns
- Diagonal residual pattern suggests non-linear relationships exist
- Model could be improved with Decision Tree or Random Forest

## Technologies Used
- **Python 3.x**
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine Learning models
- **matplotlib & seaborn**: Visualization

## Files
- `ObesityProjecti.pynb`: Complete Colab notebook with all code and explanations

## How to Run
1. Open the notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/obesity-linear-regression/blob/main/obesity_linear_regression.ipynb)
2. Or download and run locally with Jupyter Notebook
3. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`

## Conclusion
This project demonstrates the complete machine learning workflow:
- Data exploration and preparation
- Model training and evaluation
- Results interpretation and visualization

Linear Regression provides a good baseline but shows limitations with non-linear patterns, suggesting more complex models could improve predictions.

## Author
Kappati Sai Tanishtha Reddy
