import numpy as np
import pandas as pd

# List of CSV files containing predictions
csv_files = [
    'partial_predictions_seed_0.csv',
    'partial_predictions_seed_1.csv',
    'partial_predictions_seed_2.csv',
    'partial_predictions_seed_3.csv',
    'partial_predictions_seed_4.csv'
]

# Initialize an array to store the sum of predictions
predictions_sum = None

# Read and accumulate predictions from each file
for file in csv_files:
    predictions = np.loadtxt(file, delimiter=',')
    if predictions_sum is None:
        predictions_sum = predictions
    else:
        predictions_sum += predictions

# Compute the average of the predictions
average_predictions = predictions_sum / len(csv_files)

# Determine the final class labels using argmax
final_labels = np.argmax(average_predictions, axis=1)

# Load the test DataFrame to get the IDs
test_df = pd.read_csv('../data/test.csv')

# Prepare the submission DataFrame
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': final_labels})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('final_submission.csv', index=False)

print("Final submission file 'final_submission.csv' has been created.")
