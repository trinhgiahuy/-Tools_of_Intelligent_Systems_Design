import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load PCA-applied data
data_dir = '../data'
train_pca_file = os.path.join(data_dir, 'train_pca_1433.csv')
test_pca_file = os.path.join(data_dir, 'test_pca_1433.csv')

train_pca_df = pd.read_csv(train_pca_file)
test_pca_df = pd.read_csv(test_pca_file)

# Split features and labels
features = train_pca_df.drop(columns=['Label']).values
labels = train_pca_df['Label'].values
test_features = test_pca_df.values

# Standardize the data
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
test_features_standardized = scaler.transform(test_features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_standardized, labels, test_size=0.2, random_state=seed_value)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=seed_value)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [200, 300],
    'colsample_bytree': [0.3, 0.7, 1],
    'subsample': [0.5, 0.7, 1],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3]
}

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softprob', class_weight='balanced')

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=3)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_xgb_model = grid_search.best_estimator_

# Save the best model configuration
best_params = grid_search.best_params_
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv("best_xgb_model_params.csv", index=False)

# Evaluate the best model
y_val_pred = best_xgb_model.predict(X_val)
val_accuracy = np.mean(y_val_pred == y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Predict on the test set
test_predictions = best_xgb_model.predict(test_features_standardized)
test_proba_predictions = best_xgb_model.predict_proba(test_features_standardized)

# Save predictions for potential ensemble learning
np.savetxt("test_proba_predictions.csv", test_proba_predictions, delimiter=',')

# Save submission file
submission_df = pd.DataFrame({'ID': test_pca_df['ID'], 'Label': test_predictions})
submission_df.to_csv("submission.csv", index=False)

# Save the best model using joblib
joblib.dump(best_xgb_model, "best_xgb_model.pkl")

print(f"Run with seed {seed_value} completed.")
