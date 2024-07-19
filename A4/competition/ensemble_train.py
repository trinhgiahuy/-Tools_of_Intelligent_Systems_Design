import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
X_test_flat = reshaped_test.reshape(number_of_test_data_points, -1)

X_train_scaled_flat = scaler.fit_transform(X_train_flat)
X_val_scaled_flat = scaler.transform(X_val_flat)
X_test_scaled_flat = scaler.transform(X_test_flat)

X_train_scaled = X_train_scaled_flat.reshape(X_train.shape[0], 150, 200, 1)
X_val_scaled = X_val_scaled_flat.reshape(X_val.shape[0], 150, 200, 1)
X_test_scaled = X_test_scaled_flat.reshape(number_of_test_data_points, 150, 200, 1)

# Define the class weights for handling imbalanced classes
class_weights = {i: max(np.bincount(labels)) / count for i, count in enumerate(np.bincount(labels))}

# Function to create a new CNN model
def create_model(seed):
    # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_seed(seed)
    model = Sequential([
        Conv2D(64, kernel_size=(11, 11), activation='relu', input_shape=(150, 200, 1)),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training ensemble models
num_models = 5
seeds = [43, 44, 45, 46, 47, 48, 49]
models = []
val_accuracies = []

ensemble_predictions = np.zeros((X_test_scaled.shape[0], 3))
for i in range(num_models):
    print(f"Training model {i + 1} with seed {seeds[i]}")
    model = create_model(seeds[i])
    #checkpoint_cb = ModelCheckpoint(f"best_model_v2_{i}.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    # reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-6, mode='max')
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_val_scaled, y_val),
                        class_weight=class_weights, callbacks=[early_stopping_cb])

    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
    print(f"Validation Accuracy for model {i + 1}: {val_accuracy}")
    val_accuracies.append(val_accuracy)

    # Load best model weights and append to models list
    # model.load_weights(f"best_model_v2_{i}.h5")
    # models.append(model)

    predictions = model.predict(X_test_scaled)
    print(predictions)
    ensemble_predictions += predictions


ensemble_predictions /= num_models
# Get the final predicted labels
predicted_labels = np.argmax(ensemble_predictions, axis=1)

# Prepare the submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': predicted_labels})
submission_df.to_csv('cnn_submission_ensemble.csv', index=False)
print(f"Ensemble validation accuracies: {val_accuracies}")
print(f"Average validation accuracy: {np.mean(val_accuracies)}")
