import numpy as np

class TFDSVOC2007TestDataset:
    def __init__(self, target_class="mixing", num_samples=None):
        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
            import os
            
            # Suppress verbose TF and TFDS output
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            tfds.disable_progress_bar()
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow_datasets to use TFDSVOC2007TestDataset")

        print(f"Loading TFDS VOC2007 test dataset (Target: {target_class})...")
        self.target_class = target_class
        
        # Download and load the dataset
        ds, info = tfds.load('voc/2007', split='test', with_info=True, data_dir="./dataset")
        self.class_names = info.features['objects'].feature['label'].names
        
        self.samples = []
        
        for i, example in enumerate(tfds.as_numpy(ds)):
            if num_samples is not None and len(self.samples) >= num_samples:
                break
                
            image = example['image']
            objects = example['objects']
            labels = objects['label']
            bboxes = objects['bbox'] # [ymin, xmin, ymax, xmax] normalized
            
            height, width, _ = image.shape
            
            valid_boxes = []
            valid_areas = []
            valid_names = []
            
            for j in range(len(labels)):
                label_name = self.class_names[labels[j]]
                if self.target_class == 'mixing' or label_name == self.target_class:
                    bbox = bboxes[j]
                    ymin = int(bbox[0] * height)
                    xmin = int(bbox[1] * width)
                    ymax = int(bbox[2] * height)
                    xmax = int(bbox[3] * width)
                    
                    area = (xmax - xmin) * (ymax - ymin)
                    
                    valid_boxes.append([xmin, ymin, xmax, ymax])
                    valid_areas.append(area)
                    valid_names.append(label_name)
                    
            if len(valid_boxes) > 0:
                max_idx = np.argmax(valid_areas)
                largest_box = valid_boxes[max_idx]
                largest_name = valid_names[max_idx]
                
                self.samples.append({
                    'image': image,
                    'box': largest_box,
                    'class_name': largest_name,
                    'image_path': example['image/filename'].decode('utf-8')
                })
                
        print(f"Loaded {len(self.samples)} valid samples.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Return exactly what agent expects
        return {
            'image': sample['image'], 
            'box': np.array(sample['box']),
            'image_path': sample['image_path']
        }
