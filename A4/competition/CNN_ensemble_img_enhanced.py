import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Global variables for image dimensions
IMG_SHAPE1 = 150
IMG_SHAPE2 = 200

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

# Reshape features to IMG_SHAPE1 x IMG_SHAPE2
number_of_train_data_points = features.shape[0]
number_of_test_data_points = test_features.shape[0]
reshaped_features = features.reshape(number_of_train_data_points, IMG_SHAPE1, IMG_SHAPE2, 1)
reshaped_test = test_features.reshape(number_of_test_data_points, IMG_SHAPE1, IMG_SHAPE2, 1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(reshaped_features, labels, test_size=0.2, random_state=42)

# Normalize the pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = reshaped_test / 255.0

# Flatten the data for scaling
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train_flat)
X_val_standardized = scaler.transform(X_val_flat)
X_test_standardized = scaler.transform(X_test_flat)

# Reshape back to original shape
X_train_standardized = X_train_standardized.reshape(X_train.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
X_val_standardized = X_val_standardized.reshape(X_val.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
X_test_standardized = X_test_standardized.reshape(X_test.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fit the generator on the training data
datagen.fit(X_train_standardized)

# Compute class weights
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}

# Function to create a new CNN model with a custom seed
def create_model(seed):
    # Set random seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = Sequential([
        Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=(IMG_SHAPE1, IMG_SHAPE2, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training ensemble models
num_models = 5
seeds = [43, 44, 45, 46, 47]
val_accuracies = []
ensemble_predictions = np.zeros((X_test_standardized.shape[0], 3))

for i in range(num_models):
    print(f"Training model {i + 1} with seed {seeds[i]}")
    model = create_model(seeds[i])
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(datagen.flow(X_train_standardized, y_train, batch_size=32),
                        epochs=100, validation_data=(X_val_standardized, y_val), 
                        class_weight=class_weights, callbacks=[reduce_lr_cb, early_stopping_cb])
    
    val_loss, val_accuracy = model.evaluate(X_val_standardized, y_val)
    print(f"Validation Accuracy for model {i + 1}: {val_accuracy}")
    val_accuracies.append(val_accuracy)
    
    # Predict on the test set
    predictions = model.predict(X_test_standardized)
    ensemble_predictions += predictions

# Average ensemble predictions
ensemble_predictions /= num_models

# Get the final predicted labels
predicted_labels = np.argmax(ensemble_predictions, axis=1)

# Prepare the submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': predicted_labels})
submission_df.to_csv('cnn_submission_ensemble.csv', index=False)

# Print final ensemble validation accuracies
print(f"Ensemble validation accuracies: {val_accuracies}")
print(f"Average validation accuracy: {np.mean(val_accuracies)}")
