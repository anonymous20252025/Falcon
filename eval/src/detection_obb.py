import os
import re
import json
import string
import math
import argparse
import numpy as np
from shapely.geometry import Polygon

def process_flat_obb(a):
    if len(a)!=8:
        print('Error obb points number.')
        return False
    else:
        obb = [(a[0], a[1]), (a[2], a[3]),
           (a[4], a[5]), (a[6], a[7])]
        poly = Polygon(obb)
        if poly.is_valid:
            return poly
        else:
            return False


def format_preprocess(data, is_box=False, is_obb=False, isGT = False):
    multiplyer = 1
    if is_box:
        if '<quad>' in data or '<box>' in data:
            if is_obb:
                boxes = re.findall(r'<quad>(.*?)</quad>', data)
                boxes = [[ float(num)*multiplyer for num in re.findall(r'\d+', box)] for box in boxes]
            else:
                boxes = re.findall(r'<box>(.*?)</box>', data)
                boxes = [[ float(num)*multiplyer for num in re.findall(r'\d+', box)] for box in boxes]
        else:
            return []
    else:
        if isinstance(data, list):  
            boxes = [[float(a) for a in x] for x in data]
        else:
            return []
        #print (boxes)
    # if isGT:
    #     print (data)
    #     print (boxes)
        
    return boxes


def calculate_iou_obb(pred_box, gt_box):
    """
    计算两个OBB（用四个顶点表示）的IoU。
    :param points1: 第一个OBB的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param points2: 第二个OBB的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: IoU值
    """
    poly1 = process_flat_obb(pred_box)
    poly2 = process_flat_obb(gt_box)

    if not poly1:
        #print('error pred box:', pred_box)
        return 0.0

    if not poly2:
        print('error gt box:', gt_box)
        return 0.0
    
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    
    iou = intersection_area / union_area
    return iou


def calculate_iou_hbb(box1, box2):
    """计算两个HBB之间的IoU"""
    try:
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
    
        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area+0.1
    
        iou = inter_area / union_area
    except:
        iou = 1
    return iou

def calculate_ap(precision, recall):
    # Insert (0, 0) at the beginning of recall and precision (optional, depends on implementation)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Calculate AP by summing the precision-recall curve
    indices = np.where(recall[1:] != recall[:-1])[0]  # Points where recall changes
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    print ("precision", precision)
    print ("indices",indices)
    return ap


def evaluate_one_OLD(gt_boxes, pred_boxes, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    pred_boxes = format_preprocess(pred_boxes, pred_format, is_obb)
    gt_boxes = format_preprocess(gt_boxes, gt_format, is_obb, isGT=True)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    ious = np.zeros(len(pred_boxes))
    refune_ious = np.zeros(len(pred_boxes))
    match_ious = np.zeros(len(pred_boxes))
    detected_gt = []

    for i, pred_box in enumerate(pred_boxes):
        match_found = False
        min_ov=-1
        for j, gt_box in enumerate(gt_boxes):
            if is_obb:
                iou = calculate_iou_obb(pred_box, gt_box)
            else:
                iou = calculate_iou_hbb(pred_box, gt_box)
            if iou >= min_ov:
                match_ious[i] = j
                ious[i] = iou
                min_ov = iouwww


        if not match_found:
            fp[i] = 1

    fn = len(gt_boxes) - len(detected_gt)
    return tp, fp, ious, fn, len(pred_boxes), len(gt_boxes)


def evaluate_one(gt_boxes, pred_boxes, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    pred_boxes = format_preprocess(pred_boxes, pred_format, is_obb)
    gt_boxes = format_preprocess(gt_boxes, gt_format, is_obb, isGT=True)
    
    tp = np.zeros(len(pred_boxes))  # True positives
    fp = np.zeros(len(pred_boxes))  # False positives
    ious = np.zeros(len(pred_boxes))  # Original IOUs
    refune_ious = np.zeros(len(pred_boxes))  # Refined IOUs after duplicate removal
    match_ious = -1 * np.ones(len(pred_boxes))  # Matched GT index for each prediction
    detected_gt = set()  # Track matched GT indices

    # Compute IOUs and assign GT indices
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if is_obb:
                iou = calculate_iou_obb(pred_box, gt_box)
            else:
                iou = calculate_iou_hbb(pred_box, gt_box)

            # Update if better IOU is found
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        ious[i] = best_iou
        if best_iou >= iou_threshold:
            match_ious[i] = best_gt_idx  # Assign the GT index
        else:
            match_ious[i] = -1  # No match found

    # Refine IOUs and resolve duplicate matches
    for j in range(len(gt_boxes)):
        # Find all predictions matching the same GT
        matching_preds = [i for i, match_idx in enumerate(match_ious) if match_idx == j]

        if matching_preds:
            # Select the prediction with the highest IOU
            best_pred_idx = max(matching_preds, key=lambda idx: ious[idx])
            detected_gt.add(j)

            # Set refined IOU for the best match
            refune_ious[best_pred_idx] = ious[best_pred_idx]

            # Mark all other matches for this GT as zero
            for pred_idx in matching_preds:
                if pred_idx != best_pred_idx:
                    refune_ious[pred_idx] = 0

    # Mark TPs and FPs
    for i, iou in enumerate(refune_ious):
        if iou > 0:
            tp[i] = 1  # True positive
        else:
            fp[i] = 1  # False positive

    # Calculate false negatives (GTs not detected)
    fn = len(gt_boxes) - len(detected_gt)

    #return tp, fp, ious, fn, len(pred_boxes), len(gt_boxes)
    return tp, fp, ious, refune_ious, fn, len(pred_boxes), len(gt_boxes)



def calculate_eval_matrix(data, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    total_tp, total_fp, total_ious, total_refune_ious = np.array([]), np.array([]), np.array([]), np.array([])
    total_fn, total_pres, total_gts = 0,0,0

    #print (data)
    for a in data:
        tp, fp, iou, refune_ious, fn, pres, gts = evaluate_one(a['gt'], a['answer'], iou_threshold, pred_format, gt_format, is_obb)
        total_tp = np.concatenate((total_tp, tp), axis=0)
        total_fp = np.concatenate((total_fp, fp), axis=0)
        total_ious = np.concatenate((total_ious, iou), axis=0)
        total_refune_ious = np.concatenate((total_refune_ious, refune_ious), axis=0)
        total_fn += fn
        total_pres += pres
        total_gts += gts

    #print ('TP', total_tp)
    #print ('iou',total_ious)
    #print ('refune_ious',total_refune_ious)
    sorted_indices = np.argsort(-total_ious)
    total_tp = np.array(total_tp)[sorted_indices]
    total_fp = np.array(total_fp)[sorted_indices]
    
   # print ('tp',total_tp)
    #print ('fp',total_fp)
    
    all_tp = np.cumsum(total_tp)
    all_fp = np.cumsum(total_fp)
    #print (total_tp)
    recall = all_tp / float(total_gts) if total_gts > 0 else np.zero(len(all_tp))
    precision = all_tp / list(range(1,total_pres+1)) if total_pres >0 else np.zeros(len(all_tp)) 
    # print (total_gts)
    # print (len(precision))
    # print ('total_gts:', total_gts)
    # print ('all_tp:', all_tp)
    mAP = calculate_ap(precision, recall)
    
    #print (precision)
    #accuracy = all_tp[-1] / float(total_fn + total_pres) if (total_fn + total_pres) > 0 else 0
    #far = all_fp[-1] / float(all_fp[-1] + all_tp[-1] + total_fn) if (all_fp[-1] + all_tp[-1] + total_fn) >0 else 0

    #return precision[-1], recall[-1], mAP, accuracy, far
    accuracy = all_tp[-1] / float(total_fn + total_pres) if (total_fn + total_pres)>0 and len(all_tp)> 0 else 0
    far = all_fp[-1] / float(all_fp[-1] + all_tp[-1] + total_fn) if len(all_fp)> 0 and len(all_tp)> 0 and (all_fp[-1] + all_tp[-1] + total_fn) >0 else 0
    
    if len(recall)>0 and len(precision)>0:
        return precision[-1], recall[-1], mAP, accuracy, far
    else:
        return precision, recall, mAP, accuracy, far

def evaluate_detection2(data, pre_box=False, is_box=False, is_obb=False):
    re_dict = {}
    ious = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for iou in ious:
        precision, recall, mAP, accuracy, far = calculate_eval_matrix(data, iou, pre_box, is_box, is_obb)

        re_dict['FAR@'+str(iou)] = far*100.00
        if not isinstance(recall, np.ndarray):  
            re_dict['Recall@'+str(iou)] = recall*100.00
        else:
            re_dict['Recall@'+str(iou)] = 0.00
        if not isinstance(precision, np.ndarray):  
            re_dict['Precision@'+str(iou)] = precision*100.00
        else:
            re_dict['Precision@'+str(iou)] = 0.00
        re_dict['Accuracy@'+str(iou)] = accuracy*100.00
        re_dict['mAP@'+str(iou)] = mAP*100.00

    return re_dict
