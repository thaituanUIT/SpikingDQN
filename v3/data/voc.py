import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from torch.utils.data import Dataset
import random

class VOCDataset(Dataset):
    """
    Dataset loader for VOC2012 for Active Object Localization.
    """
    def __init__(self, root_dir, target_class="mixing", num_samples=None, split="train"):
        """
        Args:
            root_dir (str): Path to VOC2012 directory.
            target_class (str): Target class to detect (e.g., 'aeroplane') or 'mixing' for any object.
            num_samples (int): Max number of samples to load (useful for testing/limiting size).
            split (str): 'train' or 'val'. By default we just load all images if no specific ImageSets are used, 
                         or we can parse ImageSets/Main/train.txt if needed. For simplicity we will read the dir.
        """
        self.root_dir = root_dir
        self.target_class = target_class
        
        self.annotations_dir = os.path.join(root_dir, 'Annotations')
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        
        self.samples = []
        
        self._load_data(num_samples)

    def _load_data(self, num_samples):
        print(f"Loading VOC dataset (Target: {self.target_class})...")
        
        # Determine which images to load. This can be refined to use ImageSets.
        all_xmls = sorted(os.listdir(self.annotations_dir))
        
        # Only use a deterministic subset for randomness testing if not using exact splits
        random.seed(42)
        random.shuffle(all_xmls)

        for xml_file in all_xmls:
            if num_samples is not None and len(self.samples) >= num_samples:
                break
                
            xml_path = os.path.join(self.annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            filename = root.find('filename').text
            img_path = os.path.join(self.images_dir, filename)
            
            # Extract bounding boxes
            boxes = []
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                
                if self.target_class == 'mixing' or obj_name == self.target_class:
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    
            if len(boxes) > 0:
                # If there are multiple target objects, we might take the largest one for active object localization,
                # or arbitrarily the first one as the primary target for a single localization episode.
                # Let's take the first one or the one with the largest area.
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                largest_box = boxes[np.argmax(areas)]
                
                self.samples.append({
                    'image_path': img_path,
                    'box': largest_box
                })
        
        print(f"Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['image_path'])
        
        # We ensure it's loaded properly. cv2 loads BGR, convert to RGB for standard processing
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        box = np.array(sample['box'])
        
        return {
            'image': img, 
            'box': box,
            'image_path': sample['image_path']
        }
