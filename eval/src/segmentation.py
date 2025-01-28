import os
import numpy as np
from pycocotools import mask as maskUtils
from skimage import measure
import json


def format_polygon(polygon):
    """ 
    Convert a polygon from a flat list of coordinates [x1, y1, x2, y2, ...] 
    to a list of coordinate pairs [(x1, y1), (x2, y2), ...].
    """
    return [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

def polygon_to_mask(polygon):
    """ 
    Convert a polygon into a binary mask.
    This function takes in a polygon (list of x, y coordinates) and converts it to a binary mask (0s and 1s).
    The function also calculates the bounding box of the polygon to help in creating the mask.
    """
    x_coords = polygon[::2]  # Extract x coordinates (every other element)
    y_coords = polygon[1::2]  # Extract y coordinates (the rest)
    
    # Calculate bounding box (min and max x/y coordinates)
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Calculate the mask dimensions based on the bounding box
    mask_width = max_x - min_x + 1
    mask_height = max_y - min_y + 1
    
    # Initialize an empty mask with the size of the bounding box
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # Adjust the polygon coordinates to fit within the bounding box
    adjusted_polygon = [(x - min_x, y - min_y) for x, y in format_polygon(polygon)]
    flat_polygon = np.array(adjusted_polygon).flatten().tolist()  # Flatten the list
    
    # Create a COCO-style RLE (Run-Length Encoding) mask from the polygon
    rle = maskUtils.frPyObjects([flat_polygon], mask_height, mask_width)
    
    # Decode the RLE mask into a binary mask
    mask = maskUtils.decode(rle)
    
    return mask, (min_x, min_y, mask_width, mask_height)


def polygon_to_mask(polygon, image_height, image_width):
    """
    Convert a polygon to a binary mask, given the image height and width.
    This function is an alternative version of the above `polygon_to_mask` function.
    """
    mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    
    # Create a COCO-style RLE mask from the polygon
    rle = maskUtils.frPyObjects([polygon], image_height, image_width)
    
    # Decode the RLE mask into a binary mask
    mask = maskUtils.decode(rle)
    
    return mask

def resize_mask(mask, target_size):
    """ 
    Resize a binary mask to the target size using nearest-neighbor interpolation.
    This is useful for scaling down or up the mask for matching the target image dimensions.
    """
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

def find_best_matches(masks1, masks2, threshold=0.1):
    """ 
    Find the best matching mask pairs between two sets of masks based on the Dice coefficient. 
    If no match exceeds the threshold, set the Dice score to 0 (indicating over-segmentation).
    """
    n = len(masks1)
    m = len(masks2)
    
    # Create a matrix to store the Dice coefficients between all mask pairs
    dice_matrix = np.zeros((n, m))
    
    for i, mask1 in enumerate(masks1):
        for j, mask2 in enumerate(masks2):
            dice_matrix[i, j] = dice_coefficient(mask1, mask2)
    
    matched_pairs = []
    unmatched_masks1 = set(range(n))  # Set of mask indices in masks1
    unmatched_masks2 = set(range(m))  # Set of mask indices in masks2
    dice_scores = []

    # For each mask in masks1, find the best match in masks2
    for i in range(n):
        best_j = np.argmax(dice_matrix[i, :])  # Find best match in masks2 for mask i
        best_score = dice_matrix[i, best_j]

        if best_score > threshold:
            # If a good match is found, record it and remove from unmatched sets
            matched_pairs.append((i, best_j))
            dice_scores.append(best_score)
            unmatched_masks1.discard(i)
            unmatched_masks2.discard(best_j)
        else:
            # No good match found, assign Dice score of 0 (indicating over-segmentation)
            dice_scores.append(0.0)

    # Handle unmatched masks from both sets (indicating over/under-segmentation)
    for i in unmatched_masks1:
        dice_scores.append(0.0)  # Masks in masks1 with no match in masks2
    for j in unmatched_masks2:
        dice_scores.append(0.0)  # Masks in masks2 with no match in masks1

    return dice_scores, matched_pairs


def dice_coefficient(mask1, mask2):
    """ 
    Calculate the Dice coefficient between two binary masks.
    The Dice coefficient is a measure of overlap between two sets, with values between 0 (no overlap) and 1 (perfect overlap).
    """
    intersection = np.sum(mask1 * mask2)
    total_area = np.sum(mask1) + np.sum(mask2)
    
    if total_area == 0:
        return 1.0  # If both masks are empty, return perfect overlap
    return 2 * intersection / total_area

def iou_score(mask1, mask2):
    """ 
    Calculate the Intersection over Union (IoU) between two binary masks.
    The IoU is the ratio of the intersection area to the union area between two masks.
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    if union == 0:
        return 1.0  # If both masks are empty, return perfect IoU
    return intersection / union

def vc_score(mask1, mask2, volume_diff_t=0.08):
    """ 
    Calculate the Volume Consistency (VC_8) between two binary masks.
    The VC_8 score checks if the volume difference is within a specified threshold (8% by default).
    """
    volume1 = np.sum(mask1)
    volume2 = np.sum(mask2)
    
    if volume1 == 0 and volume2 == 0:
        return 1.0  # Both volumes are empty
    
    volume_diff = np.abs(volume1 - volume2)
    avg_volume = (volume1 + volume2) / 2.0
    
    # Return 1.0 if the volume difference is within 8% of the average volume
    return 1.0 if volume_diff <= volume_diff_t * avg_volume else 0.0


def merge_masks(masks):
    """ 
    Merge multiple binary masks into one by using logical OR operation.
    This creates a single binary mask that combines all the input masks.
    """
    merged_mask = np.zeros_like(masks[0], dtype=np.uint8)  # Initialize an empty mask with the same size
    
    for mask in masks:
        merged_mask = np.logical_or(merged_mask, mask)  # Use OR to combine masks
    
    return merged_mask.astype(np.uint8)

def calculate_merged(masks1, masks2):
    """ 
    Merge all ground truth and predicted masks, then calculate the Dice coefficient, IoU, and Volume Consistency.
    """
    # Merge all ground truth masks
    merged_gt = merge_masks(masks1)
    
    # Merge all predicted masks
    merged_result = merge_masks(masks2)
    
    # Calculate Dice coefficient, IoU, and Volume Consistency for the merged masks
    dice = dice_coefficient(merged_gt, merged_result)
    iou = iou_score(merged_gt, merged_result)
    vc8 = vc_score(merged_gt, merged_result, volume_diff_t=0.08)
    vc16 = vc_score(merged_gt, merged_result, volume_diff_t=0.16)
    
    return dice, iou, vc8, vc16

def evaluate_segmentation(y_true, y_pred, crops):
    """ 
    Evaluate the segmentation performance by calculating Dice coefficient, IoU, and Volume Consistency for each pair of ground truth and predicted masks.
    """
    index = 0
    sum_dice = 0
    sum_iou = 0
    sum_vc8 = 0
    sum_vc16 = 0
    
    # Loop over ground truth, predicted polygons, and crop coordinates
    for poly_true, poly_pred, crop in zip(y_true[0:], y_pred[0:], crops[0:]):
        if len(poly_true) != 0:
            if len(poly_pred) != 0:
                try:
                    # Calculate image width and height based on crop coordinates
                    image_w = crop[2] - crop[0]
                    image_h = crop[3] - crop[1]
                except:
                    # If crop coordinates are invalid, fall back to maximum value in the polygons
                    image_w = 0
                    image_h = 0
                    max_value_t = max(max(max(sublist) for sublist in poly_true))
                    max_value_p = max(max(max(sublist) for sublist in poly_pred))
                    image_w = int(max(max_value_t, max_value_p) + 5)
                    image_h = image_w

                # Convert polygons to masks
                mask_true = polygon_to_mask(poly_true, image_h, image_w)
                mask_pred = polygon_to_mask(poly_pred, image_h, image_w)

                # Compute Dice coefficient, IoU, and Volume Consistency for this pair of masks
                dice, iou, vc8, vc16 = calculate_merged([mask_true], [mask_pred])
                
                sum_dice += dice
                sum_iou += iou
                sum_vc8 += vc8
                sum_vc16 += vc16
                index += 1

    # Calculate average performance across all pairs of masks
    avg_dice = sum_dice / index if index > 0 else 0
    avg_iou = sum_iou / index if index > 0 else 0
    avg_vc8 = sum_vc8 / index if index > 0 else 0
    avg_vc16 = sum_vc16 / index if index > 0 else 0
    
    return avg_dice, avg_iou, avg_vc8, avg_vc16
