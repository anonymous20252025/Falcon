import numpy as np

# Function to calculate recall at k (retrieved images at k vs ground truth images)
def recall_at_k(retrieved_images, ground_truth_images, k):
    # Retrieve the top k images from the retrieved list that are also in the ground truth
    relevant_retrieved = [img for img in retrieved_images[:k] if img in ground_truth_images]
    print(relevant_retrieved)  # Print the relevant images that were retrieved within the top k
    # Return the recall as the number of relevant images retrieved divided by the total ground truth images
    return len(relevant_retrieved) / len(ground_truth_images)

# Function to calculate recall at k using a similarity matrix (based on similarity scores)
def recall_at_k_sim(similarity_matrix, k):
    recalls = []  # List to store recall values for each query
    num_queries = similarity_matrix.shape[0]  # Get the number of queries (rows in the similarity matrix)
    
    # Loop through each query in the similarity matrix
    for i in range(num_queries):
        # Sort the similarity scores in descending order and get the indices of the top k images
        sorted_indices = np.argsort(-similarity_matrix[i])
        top_k = sorted_indices[:k]
        
        # If the query image is in the top k retrieved images, recall is 1, else 0
        if i in top_k:
            recalls.append(1)
        else:
            recalls.append(0)
    
    # Return the mean recall across all queries
    return np.mean(recalls)

# Function to calculate average precision (AP) for a specific query
def average_precision(similarity_matrix, query_idx):
    # Sort the similarity scores for the given query and get the indices of sorted images
    sorted_indices = np.argsort(-similarity_matrix[query_idx])
    num_relevant = 1  # In this case, we assume only the query image itself is considered relevant
    ap = 0.0  # Initialize average precision
    hits = 0  # Count of relevant images retrieved
    
    # Loop through the sorted indices to calculate precision at each rank position
    for i, idx in enumerate(sorted_indices):
        if idx == query_idx:  # If the index corresponds to the query image (relevant)
            hits += 1
            precision = hits / (i + 1)  # Precision at this rank position
            ap += precision  # Add precision to average precision calculation
    
    # Return the average precision (AP) for the given query
    return ap / num_relevant

# Function to calculate mean average precision (mAP) across all queries
def mean_average_precision(similarity_matrix):
    aps = []  # List to store the average precision for each query
    num_queries = similarity_matrix.shape[0]  # Get the number of queries (rows in the similarity matrix)
    
    # Loop through each query and calculate average precision
    for i in range(num_queries):
        ap = average_precision(similarity_matrix, i)
        aps.append(ap)  # Append the average precision for the current query
    
    # Return the mean average precision (mAP) across all queries
    return np.mean(aps)
