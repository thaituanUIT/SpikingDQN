import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision.datasets import VOCDetection

class VOCDataset(Dataset):
    """
    Dataset loader for VOC2012 for Active Object Localization using torchvision.
    """
    def __init__(self, root_dir, target_class="mixing", num_samples=None, split="train", use_random=False):
        """
        Args:
            root_dir (str): Path to directory where torchvision will download/store VOC.
            target_class (str): Target class to detect (e.g., 'aeroplane') or 'mixing' for any object.
            num_samples (int): Max number of samples to load.
            split (str): 'train', 'trainval', or 'val'.
        """
        self.root_dir = root_dir
        self.target_class = target_class
        self.use_random = use_random
        
        image_set = 'train'
        if split in ['train', 'trainval', 'val']:
            image_set = split
            
        print(f"Downloading/Loading VOC dataset (Target: {self.target_class}) via torchvision...")
        self.voc = VOCDetection(root=self.root_dir, image_set=image_set, download=True)
        
        self.samples = []
        self._load_data(num_samples)

    def _load_data(self, num_samples):
        print(f"Loading VOC dataset (Target: {self.target_class})...")
        
        # Determine which images to load. This can be refined to use ImageSets.
        indices = list(range(len(self.voc)))
        
        # Only use a deterministic subset for randomness testing if not using exact splits
        if self.use_random:
            random.seed(42)
            random.shuffle(indices)

        for i in indices:
            if num_samples is not None and len(self.samples) >= num_samples:
                break
                
            _, target = self.voc[i]
            annotation = target['annotation']
            
            objects = annotation.get('object', [])
            if not isinstance(objects, list):
                objects = [objects]
                
            boxes = []
            box_names = []
            for obj in objects:
                obj_name = obj['name']
                
                if self.target_class == 'mixing' or obj_name == self.target_class:
                    bndbox = obj['bndbox']
                    xmin = int(float(bndbox['xmin']))
                    ymin = int(float(bndbox['ymin']))
                    xmax = int(float(bndbox['xmax']))
                    ymax = int(float(bndbox['ymax']))
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    box_names.append(obj_name)
                    
            if len(boxes) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                max_idx = np.argmax(areas)
                largest_box = boxes[max_idx]
                largest_name = box_names[max_idx]
                
                filename = annotation['filename']
                
                self.samples.append({
                    'index': i,
                    'box': largest_box,
                    'filename': filename,
                    'class_name': largest_name
                })
        
        print(f"Loaded {len(self.samples)} valid samples.")
        
        self.class_counts = {}
        for s in self.samples:
            cls = s['class_name']
            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
            
        print("\n--- Class Distribution ---")
        for cls, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls}: {count}")
        print("--------------------------\n")

    def get_sample_weights(self):
        """Returns sample weights inversely proportional to their class frequency."""
        weights = [1.0 / self.class_counts[s['class_name']] for s in self.samples]
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        voc_idx = sample['index']
        
        img_pil, _ = self.voc[voc_idx]
        img = np.array(img_pil)
        
        box = np.array(sample['box'])
        
        return {
            'image': img, 
            'box': box,
            'image_path': sample['filename']
        }
