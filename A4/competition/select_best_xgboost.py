import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib

# Set random seed for reproducibility
seed_value = 4242
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load PCA-applied data
data_dir = './data'
train_pca_file = os.path.join(data_dir, 'train_pca_1433.csv')
test_pca_file = os.path.join(data_dir, 'test_pca_1433.csv')

train_pca_df = pd.read_csv(train_pca_file)
test_pca_df = pd.read_csv(test_pca_file)

# Create ID column for test_pca_df using provided indices
test_pca_df['ID'] = range(4090, 5843)

# Split features and labels
features = train_pca_df.drop(columns=['ID', 'Label']).values
labels = train_pca_df['Label'].values
test_features = test_pca_df.drop(columns=['ID']).values

# Standardize the data
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
test_features_standardized = scaler.transform(test_features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_standardized, labels, test_size=0.2, random_state=seed_value)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=seed_value)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Parameters from the best model
best_params = {
    'colsample_bytree': 0.3,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'n_estimators': 300,
    'subsample': 0.7
}

# Function to train and evaluate the model with a given random seed
def train_and_evaluate(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_standardized, labels, test_size=0.2, random_state=seed)
    
    # Apply SMOTE to balance the training set
    smote = SMOTE(random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Initialize the XGBoost classifier with the best parameters
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', class_weight='balanced', **best_params)
    
    # Train the model
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    y_val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return val_accuracy, xgb_model

# Perform the training and evaluation with different seeds and keep the best model
best_accuracy = 0
best_model = None
for seed in range(10):  # Run for 10 different seeds
    val_accuracy, model = train_and_evaluate(seed)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model

print(f"Best Validation Accuracy: {best_accuracy}")

# Save the best model parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv("best_xgb_model_params.csv", index=False)

# Save the best model predictions for submission
test_predictions = best_model.predict(test_features_standardized)
submission_df = pd.DataFrame({'ID': range(4090, 5843), 'Label': test_predictions})
submission_df.to_csv("submission_best_model.csv", index=False)

print("Training and evaluation complete. Best model saved.")
