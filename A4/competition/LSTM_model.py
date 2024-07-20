import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Set random seed from command-line argument
seed = int(sys.argv[1])
np.random.seed(seed)
tf.random.set_seed(seed)

# Load PCA-applied data
train_pca_file = 'train_pca.csv'
test_pca_file = 'test_pca.csv'

train_pca_df = pd.read_csv(train_pca_file)
test_pca_df = pd.read_csv(test_pca_file)

# Split features and labels
features_pca = train_pca_df.drop(columns=['Label']).values
labels = train_pca_df['Label'].values
test_features_pca = test_pca_df.values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_pca, labels, test_size=0.2, random_state=seed)

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = test_features_pca.reshape((test_features_pca.shape[0], test_features_pca.shape[1], 1))

# Compute class weights
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}

# Function to create a new LSTM model with a custom seed
def create_model(seed):
    model = Sequential([
        LSTM(128, input_shape=(200, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
model = create_model(seed)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size=64,
                    epochs=100, validation_data=(X_val, y_val),
                    class_weight=class_weights, callbacks=[reduce_lr_cb, early_stopping_cb])

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Predict on the test set and save predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

if val_accuracy > 0.62:
    predictions_file = f"predictions_{val_accuracy:.3f}.csv"
    submission_file = f"submission_{val_accuracy:.3f}.csv"
    
    np.savetxt(predictions_file, predictions, delimiter=',', fmt='%f')
    
    submission_df = pd.DataFrame({'ID': test_pca_df.index, 'Label': predicted_labels})
    submission_df.to_csv(submission_file, index=False)

print(f"Run with seed {seed} completed.")
