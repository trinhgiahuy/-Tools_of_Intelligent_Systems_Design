import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Set random seed for reproducibility
seed_value = 25
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load data
data_dir = '../data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
features = train_df.drop(columns=['ID', 'Label']).values
labels = train_df['Label'].values
test_features = test_df.drop(columns=['ID']).values

print(f"Original shape of features: {features.shape}")
print(f"Original shape of test_features: {test_features.shape}")

# Apply PCA
pca = PCA(n_components=2500)  # Adjust n_components or variance ratio as needed
features_pca = pca.fit_transform(features)
test_features_pca = pca.transform(test_features)

print(f"Shape of features after PCA: {features_pca.shape}")
print(f"Shape of test_features after PCA: {test_features_pca.shape}")

# Reshape the PCA-transformed features back to image-like format if possible
number_of_train_data_points = features_pca.shape[0]
number_of_test_data_points = test_features_pca.shape[0]

reshaped_features = features_pca.reshape(number_of_train_data_points, 50, 50)  # Adjust dimensions as needed
reshaped_test = test_features_pca.reshape(number_of_test_data_points, 50, 50)  # Adjust dimensions as needed

print(f"New shape of features: {reshaped_features.shape}")
print(f"New shape of test_features: {reshaped_test.shape}")

reshaped_features_df = pd.DataFrame(reshaped_features.reshape(number_of_train_data_points, -1))
reshaped_features_df['ID'] = train_df['ID'].values
reshaped_features_df['Label'] = train_df['Label'].values

reshaped_test_df = pd.DataFrame(reshaped_test.reshape(number_of_test_data_points, -1))
reshaped_test_df['ID'] = test_df['ID'].values
print(f"Reshaped features DataFrame shape: {reshaped_features_df.shape}")
print(f"Reshaped test features DataFrame shape: {reshaped_test_df.shape}")

X_train, X_val, y_train, y_val = train_test_split(
        reshaped_features_df.drop(columns=['ID', 'Label']).values.reshape(-1, 50, 50, 1),  # Adjust dimensions as needed
        reshaped_features_df['Label'].values,
        test_size=0.2, random_state=42

)

scaler = StandardScaler()
# Flatten the 2D data for scaling
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = reshaped_test_df.drop(columns=['ID']).values.reshape(reshaped_test_df.shape[0], -1)

X_train_scaled_flat = scaler.fit_transform(X_train_flat)
X_val_scaled_flat = scaler.transform(X_val_flat)
X_test_scaled_flat = scaler.transform(X_test_flat)

X_train_scaled = X_train_scaled_flat.reshape(X_train.shape[0], 50, 50, 1)  # Adjust dimensions as needed
X_val_scaled = X_val_scaled_flat.reshape(X_val.shape[0], 50, 50, 1)  # Adjust dimensions as needed
X_test_scaled = X_test_scaled_flat.reshape(X_test_flat.shape[0], 50, 50, 1)  # Adjust dimensions as needed

# Verify the shapes
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_val shape: {X_val_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

# Define the class weights for handling imbalanced classes
class_weights = {i: max(np.bincount(labels)) / count for i, count in enumerate(np.bincount(labels))}

# Callbacks
checkpoint_cb = ModelCheckpoint("best_model_v2.h5", save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build the CNN model
CNN_model = Sequential([
        Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=(50, 50, 1)),  # Adjust dimensions as needed
#        MaxPooling2D(pool_size=(2, 2)),
#        BatchNormalization(),
#        Conv2D(128, kernel_size=(5, 5), activation='relu'),
#        MaxPooling2D(pool_size=(2, 2)),
#        BatchNormalization(),
        Flatten(),
#        Dense(128, activation='relu'),
#        BatchNormalization(),
        Dense(64, activation='relu'),
        # Dropout(0.5),
        Dense(3, activation='softmax')

])

CNN_model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = CNN_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb]

)

# Evaluate the model on validation set
val_loss, val_accuracy = CNN_model.evaluate(X_val_scaled, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Save the final model
CNN_model.save("improved_cnn_model_v2.h5")

# Predict on the test set
predictions = CNN_model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)

# Prepare the submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': predicted_labels})
submission_df.to_csv('cnn_submission.csv', index=False)

