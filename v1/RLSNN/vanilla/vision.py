import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import RLSNN.vanilla.parameters as parameters

def crop_image(image, new_mask):
    height, width, channel = np.shape(image)
    new_mask = np.asarray(new_mask).astype("int")
    new_mask[0] = max(new_mask[0], 0)
    new_mask[1] = max(new_mask[1], 0)
    new_mask[2] = min(new_mask[2], width)
    new_mask[3] = min(new_mask[3], height)
    cropped_image = image[new_mask[1]:new_mask[3], new_mask[0]:new_mask[2]]
    new_height, new_width, new_channel = np.shape(cropped_image)

    if new_height == 0 or new_width == 0:
        cropped_image = np.zeros((224, 224, 3))
    else:
        cropped_image = cv2.resize(cropped_image, (224, 224))

    return cropped_image

def feature_extract(img, history, vgg16):
    feat_hist = np.zeros(parameters.action_option * parameters.history_size)
    for i in range(parameters.history_size):
        if history[i] != -1:
            feat_hist[i * parameters.action_option + history[i]] = 1
            
    img_resized = cv2.resize(img, (224, 224))
    img_transposed = np.transpose(img_resized, (2, 0, 1)) 
    
    image_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float() 
    
    device = next(vgg16.parameters()).device
    image_tensor = image_tensor.to(device)

    vgg16.eval()
    with torch.no_grad():
        x = vgg16.features(image_tensor)
        x = vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        extracted_tensor = vgg16.classifier[:2](x)[0]
    
    image_feature = extracted_tensor.cpu().numpy().flatten()
    feature = np.concatenate((image_feature, feat_hist))

    return np.array([feature])

def compute_mask(action, current_mask):
    image_rate = 0.1
    delta_width = image_rate * (current_mask[2] - current_mask[0])
    delta_height = image_rate * (current_mask[3] - current_mask[1])
    dx1 = 0
    dy1 = 0
    dx2 = 0
    dy2 = 0

    if action == 0:
        dx1 = delta_width
        dx2 = delta_width
    elif action == 1:
        dx1 = -delta_width
        dx2 = -delta_width
    elif action == 2:
        dy1 = delta_height
        dy2 = delta_height
    elif action == 3:
        dy1 = -delta_height
        dy2 = -delta_height
    elif action == 4:
        dx1 = -delta_width
        dx2 = delta_width
        dy1 = -delta_height
        dy2 = delta_height
    elif action == 5:
        dx1 = delta_width
        dx2 = -delta_width
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 6:
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 7:
        dx1 = delta_width
        dx2 = -delta_width

    new_mask_tmp = np.array([current_mask[0] + dx1, current_mask[1] + dy1,
                         current_mask[2] + dx2, current_mask[3] + dy2])
    new_mask = np.array([
        min(new_mask_tmp[0], new_mask_tmp[2]),
        min(new_mask_tmp[1], new_mask_tmp[3]),
        max(new_mask_tmp[0], new_mask_tmp[2]),
        max(new_mask_tmp[1], new_mask_tmp[3])
    ])

    return new_mask

def compute_iou(mask, ground_truth):
    dx = min(mask[2], ground_truth[2]) - max(mask[0], ground_truth[0])
    dy = min(mask[3], ground_truth[3]) - max(mask[1], ground_truth[1])

    if (dx >= 0) and (dy >= 0):
        inter_area = dx*dy
    else:
        inter_area = 0

    mask_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
    ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])

    return inter_area / (mask_area + ground_truth_area - inter_area)