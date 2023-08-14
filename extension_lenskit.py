import time
from lenskit import batch
from lenskit.algorithms import als
import pandas as pd


def calculate_map(predictions, truth):
    # Initialize sum of average precisions
    ap_sum = 0
    # Initialize count of users
    user_count = 0
    
    # For each user
    for user in predictions['user'].unique():
        # Get predicted items for user
        pred_items = predictions[predictions['user'] == user]['item'].tolist()
        # Get true items for user
        truth_items = truth[truth['user'] == user]['item'].tolist()
        
        # Initialize count of hits and sum of precisions
        hits = 0
        sum_precisions = 0
        
        # For each item in the predicted items
        for i, item in enumerate(pred_items):
            # If the item is in the true items
            if item in truth_items:
                # Increase the count of hits
                hits += 1
                # Add the precision at this rank to the sum of precisions
                sum_precisions += hits / (i + 1)
                
        # If there were hits, add the average precision to the sum of average precisions
        if hits > 0:
            ap_sum += sum_precisions / len(truth_items)
        
        # Increase the count of users
        user_count += 1

    # Calculate and return MAP
    return ap_sum / user_count

start_time = time.time()

indexed_data = pd.read_csv("indexed_data.csv")

# Randomly select 10% of the training data
train_df = indexed_data.sample(frac=0.1, random_state=42)

# Create ALS model
algo = als.BiasedMF(0)

# Rename columns
train_df = train_df.rename(columns={'user_id': 'user', 'recording_mbid_index': 'item', 'rating': 'rating'})

# Drop unnecessary columns
train_df = train_df[['user', 'item', 'rating']]

# Fit model
algo.fit(train_df)

# Read the held-out interactions from the Parquet file
test_df = pd.read_csv("indexed_data_val.csv")

# Rename columns
test_df = test_df.rename(columns={'user_id': 'user', 'recording_mbid_index': 'item', 'rating': 'rating'})

# Drop unnecessary columns
test_df = test_df[['user', 'item', 'rating']]

# Predict ratings for the test set
predictions = batch.predict(algo, test_df)

end_time = time.time()

print(f"Time taken to implement the algorithm: {end_time - start_time} seconds")



# Generate recommendations for all users
users = test_df.user.unique()
predictions = predictions.sort_values('prediction', ascending=False).groupby('user').head(100)

# Sort test_df by rating and group by user to get the top rated items for each user
test_df = test_df.sort_values('rating', ascending=False).groupby('user').head(100)



MAP = calculate_map(predictions, test_df)
print(f"Mean Average Precision score: {MAP:.4f}")

