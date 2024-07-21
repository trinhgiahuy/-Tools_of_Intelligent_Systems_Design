import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Global variables for image dimensions
IMG_SHAPE1 = 150
IMG_SHAPE2 = 200

# Set random seed from command-line argument
seed = int(sys.argv[1])
np.random.seed(seed)
tf.random.set_seed(seed)

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
X_train, X_val, y_train, y_val = train_test_split(reshaped_features, labels, test_size=0.2, random_state=seed)

# Standardize the data
scaler = StandardScaler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = reshaped_test.reshape(reshaped_test.shape[0], -1)

X_train_standardized = scaler.fit_transform(X_train_flat)
X_val_standardized = scaler.transform(X_val_flat)
X_test_standardized = scaler.transform(X_test_flat)

# Reshape back to original shape
X_train_standardized = X_train_standardized.reshape(X_train.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
X_val_standardized = X_val_standardized.reshape(X_val.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
X_test_standardized = X_test_standardized.reshape(X_test_flat.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)

# Compute class weights
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}

# Function to create a new CNN model with a custom seed
def create_model(seed):
    model = Sequential([
        #Conv2D(32, kernel_size=(11, 11), activation='relu', input_shape=(IMG_SHAPE1, IMG_SHAPE2, 1)),
        Conv2D(64, kernel_size=(11, 11), activation='relu'),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
model = create_model(seed)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_standardized, y_train, batch_size=64,
                    epochs=100, validation_data=(X_val_standardized, y_val),
                    class_weight=class_weights, callbacks=[reduce_lr_cb, early_stopping_cb])

val_loss, val_accuracy = model.evaluate(X_val_standardized, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Predict on the test set and save predictions
predictions = model.predict(X_test_standardized)
predicted_labels = np.argmax(predictions, axis=1)

if val_accuracy > 0.62:
    predictions_file = f"./62/predictions_{val_accuracy:.3f}.csv"
    submission_file = f"./62/submission_{val_accuracy:.3f}.csv"

    np.savetxt(predictions_file, predictions, delimiter=',', fmt='%f')

    submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': predicted_labels})
    submission_df.to_csv(submission_file, index=False)

print(f"Run with seed {seed} completed.")
