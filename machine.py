import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

# ---------------------------
# Step 1: Load Dataset
# Replace this with your dataset path or loading method
# For demonstration, we create a small synthetic dataset
data = {
    'size': [1500, 1800, 2400, 3000, 3500, 2000, 2300, 2700, 3200, 3600],
    'bedrooms': [3, 4, 3, 5, 4, 3, 4, 4, 5, 4],
    'bathrooms': [2, 3, 2, 4, 3, 2, 3, 3, 4, 3],
    'age': [10, 15, 20, 5, 8, 12, 18, 7, 6, 9],
    'neighborhood': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'C', 'B'],
    'condition': ['Good', 'Excellent', 'Fair', 'Good', 'Excellent', 'Fair', 'Good', 'Excellent', 'Fair', 'Good'],
    'SalePrice': [400000, 500000, 600000, 650000, 700000, 420000, 510000, 620000, 670000, 710000]
}
dataset = pd.DataFrame(data)

# ---------------------------
# Step 2: Separate features and target
X = dataset.drop('SalePrice', axis=1)
y = dataset['SalePrice']

# ---------------------------
# Step 3: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ---------------------------
# Step 4: Preprocessing pipelines for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ---------------------------
# Step 5: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 6: Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR()
}

# Store predictions for visualization
predictions = {}

# ---------------------------
# Step 7: Train, predict and evaluate each model
for name, model in models.items():
    # Create pipeline with preprocessing and model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    predictions[name] = preds
    mape = mean_absolute_percentage_error(y_test, preds)
    print(f"{name} Mean Absolute Percentage Error: {mape:.4f}")

# ---------------------------
# Step 8: Visualization of Actual vs Predicted Prices

# Convert y_test to numpy array for plotting
y_test_array = y_test.to_numpy()

plt.figure(figsize=(18, 5))

for i, (name, preds) in enumerate(predictions.items(), 1):
    # Line plot
    plt.subplot(2, len(models), i)
    plt.plot(y_test_array, label='Actual Prices', marker='o')
    plt.plot(preds, label='Predicted Prices', marker='x')
    plt.title(f'{name} - Line Plot')
    plt.xlabel('Sample Index')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)

    # Scatter plot
    plt.subplot(2, len(models), i + len(models))
    plt.scatter(y_test_array, preds, color='blue')
    plt.plot([min(y_test_array), max(y_test_array)],
             [min(y_test_array), max(y_test_array)],
             color='red', linestyle='--')  # Diagonal line y=x
    plt.title(f'{name} - Scatter Plot')
    plt.xlabel('Actual Prices ($)')
    plt.ylabel('Predicted Prices ($)')
    plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------
# Step 9: Feature Importance from Random Forest

# Fit Random Forest pipeline again to extract feature importances
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', models['Random Forest'])])
rf_pipeline.fit(X_train, y_train)

# Extract feature names after one-hot encoding
onehot_features = list(rf_pipeline.named_steps['preprocessor']
                       .transformers_[1][1]
                       .named_steps['onehot']
                       .get_feature_names_out(categorical_cols))
all_features = numerical_cols + onehot_features

importances = rf_pipeline.named_steps['regressor'].feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=all_features)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
