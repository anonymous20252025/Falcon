import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def single_label(y_true_bin, y_pred_bin, class_names):
    # Calculate accuracy, precision, recall, and F1 score
    accuracy_avg = accuracy_score(y_true_bin, y_pred_bin)
    precision_macro = precision_score(y_true_bin, y_pred_bin, average='macro')
    recall_macro = recall_score(y_true_bin, y_pred_bin, average='macro')
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')  

    precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
    recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
    f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

    micro_metrics = {}
    for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
        micro_metrics[class_name] = {'Precision':p, 'Recall':r, 'F1-score':f}
        
    # Print or return the results
    # print(f'Accuracy: {accuracy_avg:.4f}')
    # print(f'Precision Macro: {precision_macro:.4f}')
    # print(f'Recall Macro: {recall_macro:.4f}')
    # print(f'F1 Score Macro: {f1_macro:.4f}')

    return accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics

def multi_label(y_true_bin, y_pred_bin, class_names):
    accuracy = sample_acc(y_true_bin, y_pred_bin)#Accuracy(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, average='samples')
    recall = recall_score(y_true_bin, y_pred_bin, average='samples')
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')

    precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
    recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
    f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

    micro_metrics = {}
    for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
        if not ',' in class_name:
            micro_metrics[class_name] = {'Precision':p, 'Recall':r, 'F1-score':f}
    # Print or return the results
    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1, micro_metrics

def sample_acc(y_true, y_pred):
    sample_accuracies = []
    for true, pred in zip(y_true, y_pred):
        # Convert binary vectors to sets of indices where value is 1
        true_set = set(idx for idx, val in enumerate(true) if val == 1)
        pred_set = set(idx for idx, val in enumerate(pred) if val == 1)
        
        # Calculate intersection and union
        if len(true_set) > 0:  # Avoid division by zero
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            sample_accuracies.append(intersection / union)
        else:
            sample_accuracies.append(0)  # No ground truth labels
    
    # Calculate average accuracy
    average_accuracy = sum(sample_accuracies) / len(sample_accuracies)
    print(f"Sample-Level Accuracy: {average_accuracy:.2f}")
    return average_accuracy


def evaluate_classification(y_true, y_pred, class_names=''):
    # Check if the task is multi-label based on presence of commas
    is_multi_label = any(',' in yt for yt in y_true)# or any(',' in yp for yp in y_pred)

    if is_multi_label:

        # 转换为多标签格式
        y_true_sets = [set(labels.split(', ')) for labels in y_true]
        y_pred_sets = [set(labels.split(', ')) for labels in y_pred]
    
        
        # 统计所有可能的标签
        all_labels = list(set.union(*y_true_sets, *y_pred_sets))
    
        # 将标签集转为多标签二进制矩阵
        y_true_bin = [[1 if label in true else 0 for label in all_labels] for true in y_true_sets]
        y_pred_bin = [[1 if label in pred else 0 for label in all_labels] for pred in y_pred_sets]

        print("Multi-label task detected.")
        return multi_label(y_true_bin, y_pred_bin,class_names)

    else:
        
        # Convert each string of classes into a list of classes, removing any extra spaces
        y_true = [set(yt.strip().split(', ')) for yt in y_true]
        y_pred = [set(yp.strip().split(', ')) for yp in y_pred]
        
        # Find all unique classes in y_pred that are not in class_names
        unique_pred_classes = set().union(*y_pred)
        extra_classes = unique_pred_classes - set(class_names)
        
        # Rename any extra classes to 'dummy'
        if extra_classes:
            print(f"Found classes in y_pred not in class_names: {extra_classes}. Renaming to 'dummy'.")
            y_pred = [{('dummy' if cls in extra_classes else cls) for cls in yp} for yp in y_pred]
            class_names = class_names + ['dummy']  # Add 'dummy' class to class_names
    
        # Convert to one-hot encoded format using MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=class_names)
        
        # Transform y_true and y_pred to a binary indicator matrix
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        print("Single-label task detected.")
        return single_label(y_true_bin, y_pred_bin,class_names)
    
