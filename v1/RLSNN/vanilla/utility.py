import cv2
import xmltodict
from typing import Dict
import numpy as np
import pandas as pd
import os

def load_data(obj_num, test=False):
    """
    Loading dataset images for a specific class by calling corresponding functions
    and saving images and their annotations into arrays
    """
    def get_obj_name(obj_num: int) -> str:
        """
        Converting object's id to object's name
        """
        obj_dict: Dict[int, str] = { 1:'aeroplane' , 2:'bicycle', 3:'bird', 4:'boat', 
                     5:'bottle' , 6:'bus', 7:'car', 8:'cat', 9:'chair', 
                     10:'cow', 11:'diningtable', 12:'dog' , 13:'horse', 
                     14:'motorbike', 15:'person', 16:'pottedplant',
                     17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor' }

        return obj_dict[obj_num]

    def read_img_idx(obj_name, dataset_path, test):
        """
        Reading the name of images from the txt file of target object
        """
        idx_list = []
        if test:
          idx_file_path = dataset_path + "ImageSets/Main/" + obj_name + "_trainval.txt"
        else:
          idx_file_path = dataset_path + "ImageSets/Main/" + obj_name + "_train.txt"
        with open(idx_file_path, 'r') as f:
            for line in f:
                # only consider those images that consist of the class (aeroplane) we are interested in.
                if "-1" not in line.split(" ")[1]:
                    idx_list.append(line.split(" ")[0])

        return idx_list

    def read_img(img_idx, dataset_path):
        """
        Loading images using their name from JPEGImages folder
        """
        img_list = []
        img_folder_path = dataset_path + "JPEGImages/"
        for each_image in img_idx:
            img = cv2.imread(img_folder_path + each_image + ".jpg")
            img_list.append(img)

        return img_list

    def load_annotation(img_idx, obj_name, dataset_path):

        """
        Loading bounding boxes around objects in images
        """
        bbox_list = []
        annotation_path = dataset_path + "Annotations/"
        for each_img in img_idx:
            path = annotation_path + each_img + ".xml"
            xml = xmltodict.parse(open(path, 'rb'))
            xml_objs = xml['annotation']['object']
            if isinstance(xml_objs, list):
                for each_obj in xml_objs:

                    if each_obj["name"] == obj_name:
                        xmin = each_obj["bndbox"]["xmin"]
                        ymin = each_obj["bndbox"]["ymin"]
                        xmax = each_obj["bndbox"]["xmax"]
                        ymax = each_obj["bndbox"]["ymax"]
                        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
                        bbox_list.append(bbox)
                        break
            else:
                if xml_objs["name"] == obj_name:
                    xmin = xml_objs["bndbox"]["xmin"]
                    ymin = xml_objs["bndbox"]["ymin"]
                    xmax = xml_objs["bndbox"]["xmax"]
                    ymax = xml_objs["bndbox"]["ymax"]
                    bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
                    bbox_list.append(bbox)

        return bbox_list

    dataset_path = "./VOC2012/"
    obj_name = get_obj_name(obj_num)  
    img_idx = read_img_idx(obj_name, dataset_path, test)
    img_list = np.asarray(read_img(img_idx, dataset_path), dtype=object)
    bbox_list = np.asarray(load_annotation(img_idx, obj_name, dataset_path), dtype=object)
    
    return img_list, bbox_list

