import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_precision_recall_f1(ground_truth, predictions):
    """
    Calculate precision, recall, F1 score, and accuracy for the given ground truth and predictions.
    
    Parameters:
    - ground_truth: List of ground truth labels (true answers).
    - predictions: List of predicted labels.
    
    Returns:
    - accuracy: Accuracy score.
    - precision: Weighted precision score.
    - recall: Weighted recall score.
    - f1: Weighted F1 score.
    """
    # Flatten ground truth for single answer case
    gt_flat = [gt for gt in ground_truth]
    
    ## For multi-answer ground truth, uncomment the following line and comment the above line
    # gt_flat = [gt[0] for gt in ground_truth]
    
    # Calculate accuracy
    accuracy = accuracy_score(gt_flat, predictions)
    
    # Calculate weighted precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(gt_flat, predictions, average='weighted')
    
    return accuracy, precision, recall, f1

def evaluate_vqa(ground_truth, predictions):
    """
    Wrapper function to evaluate the Visual Question Answering (VQA) task.
    
    Parameters:
    - ground_truth: List of ground truth labels (true answers).
    - predictions: List of predicted labels.
    
    Returns:
    - accuracy, precision, recall, f1: Evaluation metrics for VQA.
    """
    return calculate_precision_recall_f1(ground_truth, predictions)
