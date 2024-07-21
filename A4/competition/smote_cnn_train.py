import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# Set random seed from command-line argument
seed = int(sys.argv[1])
np.random.seed(seed)
tf.random.set_seed(seed)

# Load PCA-applied data
data_dir = '../data'
train_pca_file = os.path.join(data_dir, 'train_pca_1433.csv')
test_pca_file = os.path.join(data_dir, 'test_pca_1433.csv')

train_pca_df = pd.read_csv(train_pca_file)
test_pca_df = pd.read_csv(test_pca_file)

# Split features and labels
features_pca = train_pca_df.drop(columns=['Label']).values
labels = train_pca_df['Label'].values
test_features_pca = test_pca_df.values

# Reshape data to 2D for CNN input
IMG_SHAPE1 = 47  # Adjust these values as per the PCA component count
IMG_SHAPE2 = 31  # 1433 = 47 * 31 (approximation)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features_pca, labels, test_size=0.2, random_state=seed)

# Apply SMOTE to balance the training set
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Reshape for CNN input
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
X_val = X_val.reshape(X_val.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)
test_features_pca = test_features_pca.reshape(test_features_pca.shape[0], IMG_SHAPE1, IMG_SHAPE2, 1)

# Compute class weights
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}

# Function to create a new CNN model with a custom seed
def create_model(seed):
    model = Sequential([
        Conv2D(64, kernel_size=(11, 11), activation='relu', input_shape=(IMG_SHAPE1, IMG_SHAPE2, 1)),
        Dropout(0.3),
        Conv2D(128, kernel_size=(5, 5), activation='relu'),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=L2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
model = create_model(seed)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_resampled, y_train_resampled, batch_size=64,
                    epochs=100, validation_data=(X_val, y_val),
                    class_weight=class_weights, callbacks=[reduce_lr_cb, early_stopping_cb])

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Predict on the test set and save predictions
predictions = model.predict(test_features_pca)
predicted_labels = np.argmax(predictions, axis=1)

if val_accuracy > 0.62:
    predictions_file = f"./62/predictions_{val_accuracy:.3f}.csv"
    submission_file = f"./62/submission_{val_accuracy:.3f}.csv"

    np.savetxt(predictions_file, predictions, delimiter=',', fmt='%f')

    submission_df = pd.DataFrame({'ID': test_pca_df['ID'], 'Label': predicted_labels})
    submission_df.to_csv(submission_file, index=False)

print(f"Run with seed {seed} completed.")
