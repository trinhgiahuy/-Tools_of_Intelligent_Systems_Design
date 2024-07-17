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

# Split features and labels
features = train_df.drop(columns=['ID', 'Label']).values
labels = train_df['Label'].values
test_features = test_df.drop(columns=['ID']).values

# Reshape features to 150x200
number_of_train_data_points = features.shape[0]
number_of_test_data_points = test_features.shape[0]
reshaped_features = features.reshape(number_of_train_data_points, 150, 200, 1)
reshaped_test = test_features.reshape(number_of_test_data_points, 150, 200, 1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(reshaped_features, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = reshaped_test.reshape(reshaped_test.shape[0], -1)

X_train_scaled_flat = scaler.fit_transform(X_train_flat)
X_val_scaled_flat = scaler.transform(X_val_flat)
X_test_scaled_flat = scaler.transform(X_test_flat)

X_train_scaled = X_train_scaled_flat.reshape(X_train.shape[0], 150, 200, 1)
X_val_scaled = X_val_scaled_flat.reshape(X_val.shape[0], 150, 200, 1)
X_test_scaled = X_test_scaled_flat.reshape(X_test_flat.shape[0], 150, 200, 1)

# Define the class weights for handling imbalanced classes
class_weights = {i: max(np.bincount(labels)) / count for i, count in enumerate(np.bincount(labels))}

# Callbacks
checkpoint_cb = ModelCheckpoint("best_model_v2.h5", save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Improved CNN model architecture
CNN_model = Sequential([
        Conv2D(64, kernel_size=(11, 11), activation='relu', input_shape=(150, 200, 1)),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(3, activation='softmax')

])

CNN_model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = CNN_model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_val_scaled, y_val), class_weight=class_weights, callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb])

# Evaluate the model on validation set
val_loss, val_accuracy = CNN_model.evaluate(X_val_scaled, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Save the final model
# CNN_model.save("improved_cnn_model_v2.h5")

# Load the best model weights
CNN_model.load_weights("best_model_v2.h5")

# Combine training and validation sets for full dataset training
X_full_train = np.concatenate((X_train_scaled, X_val_scaled), axis=0)
y_full_train = np.concatenate((y_train, y_val), axis=0)

# Adjust callbacks for full dataset training
# reduce_lr_cb_full = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
# early_stopping_cb_full = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Continue training on full dataset
history_full = CNN_model.fit(X_full_train, y_full_train, epochs=10, class_weight=class_weights)

# Predict on the validation set after transfer learning
val_predictions = CNN_model.predict(X_val_scaled)
val_predicted_labels = np.argmax(val_predictions, axis=1)
val_accuracy_full = np.mean(val_predicted_labels == y_val)
print(f"Validation Accuracy after transfer learning: {val_accuracy_full}")

# Predict on the test set
predictions = CNN_model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)

# Prepare the submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': predicted_labels})
submission_df.to_csv('cnn_submission_full_train.csv', index=False)

