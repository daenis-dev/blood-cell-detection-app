# Overview

- Follow the guide below in order to create a system that identifies the presence of red blood cells, white blood cells or platelets within images



# Download and Prepare the Data

- Download the Blood Cell Count and Detection dataset

  - Forked from https://github.com/Shenggan/BCCD_Dataset

- Convert the XML labels to YOLO format

  ```java
  import os
  import shutil
  import random
  from sklearn.model_selection import train_test_split
  import xml.etree.ElementTree as ET
  
  dataset_path = os.path.expanduser('~/projects/blood-cell-detection-app/BCCD_Dataset/BCCD')
  image_folder = os.path.join(dataset_path, 'JPEGImages')
  label_folder = os.path.join(dataset_path, 'Annotations')
  
  images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
  labels = [f for f in os.listdir(label_folder) if f.endswith('.xml')]
  
  
  def convert_voc_to_yolo(annotations_dir, output_dir, classes):
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
      
      for filename in os.listdir(annotations_dir):
          if filename.endswith('.xml'):
              tree = ET.parse(os.path.join(annotations_dir, filename))
              root = tree.getroot()
  
              img_width = int(root.find('size').find('width').text)
              img_height = int(root.find('size').find('height').text)
  
              with open(os.path.join(output_dir, filename.replace('.xml', '.txt')), 'w') as yolo_file:
                  for obj in root.findall('object'):
                      class_name = obj.find('name').text
                      class_id = classes.index(class_name)
  
                      bbox = obj.find('bndbox')
                      xmin = int(bbox.find('xmin').text)
                      xmax = int(bbox.find('xmax').text)
                      ymin = int(bbox.find('ymin').text)
                      ymax = int(bbox.find('ymax').text)
  
                      x_center = (xmin + xmax) / 2 / img_width
                      y_center = (ymin + ymax) / 2 / img_height
                      width = (xmax - xmin) / img_width
                      height = (ymax - ymin) / img_height
  
                      yolo_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
  
  classes = ['RBC', 'WBC', 'Platelets']  # Define your blood cell classes
  convert_voc_to_yolo('C:/Users/your-username/projects/blood-cell-detection-app/BCCD_Dataset/BCCD/Annotations', 'C:/Users/your-username/projects/blood-cell-detection-app/BCCD_Dataset/BCCD/labels', classes)
  ```

- Perform stratified sampling and split the data between `train`, `test` and `val` directories

  ```python
  import pandas as pd
  from sklearn.model_selection import train_test_split
  import shutil
  
  image_folder = 'C:/Users/your-username/projects/blood-cell-detection-app/BCCD_Dataset/BCCD/JPEGImages'
  label_folder = 'C:/Users/your-username/projects/blood-cell-detection-app/BCCD_Dataset/BCCD/labels'
  data = []
  
  for label_file in os.listdir(label_folder):
      if label_file.endswith('.txt'):
          with open(os.path.join(label_folder, label_file), 'r') as f:
              lines = f.readlines()
              for line in lines:
                  class_id = line.split()[0]
                  class_name = 'RBC' if class_id == '0' else 'WBC' if class_id == '1' else 'Platelets'
                  image_file = label_file.replace('.txt', '.jpg')  # Assuming image file has same name
                  data.append((image_file, class_name))
  
  df = pd.DataFrame(data, columns=['image', 'class'])
  
  train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
  val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
  
  print("Training set class distribution:\n", train_df['class'].value_counts())
  print("Validation set class distribution:\n", val_df['class'].value_counts())
  print("Testing set class distribution:\n", test_df['class'].value_counts())
  
  output_folder = 'C:/Users/your-username/projects/blood-cell-detection-app/data'
  os.makedirs(os.path.join(output_folder, 'train/images'), exist_ok=True)
  os.makedirs(os.path.join(output_folder, 'train/labels'), exist_ok=True)
  os.makedirs(os.path.join(output_folder, 'val/images'), exist_ok=True)
  os.makedirs(os.path.join(output_folder, 'val/labels'), exist_ok=True)
  os.makedirs(os.path.join(output_folder, 'test/images'), exist_ok=True)
  os.makedirs(os.path.join(output_folder, 'test/labels'), exist_ok=True)
  
  for _, row in train_df.iterrows():
      shutil.copy(os.path.join(image_folder, row['image']), os.path.join(output_folder, 'train/images', row['image']))
      shutil.copy(os.path.join(label_folder, row['image'].replace('.jpg', '.txt')), os.path.join(output_folder, 'train/labels', row['image'].replace('.jpg', '.txt')))
  
  for _, row in val_df.iterrows():
      shutil.copy(os.path.join(image_folder, row['image']), os.path.join(output_folder, 'val/images', row['image']))
      shutil.copy(os.path.join(label_folder, row['image'].replace('.jpg', '.txt')), os.path.join(output_folder, 'val/labels', row['image'].replace('.jpg', '.txt')))
  
  for _, row in test_df.iterrows():
      shutil.copy(os.path.join(image_folder, row['image']), os.path.join(output_folder, 'test/images', row['image']))
      shutil.copy(os.path.join(label_folder, row['image'].replace('.jpg', '.txt')), os.path.join(output_folder, 'test/labels', row['image'].replace('.jpg', '.txt')))
  ```

  

# Download and Configure the Model

- Use Git to download the yolov5 library

  ```
  >> git clone https://github.com/ultralytics/yolov5
  ```

- Update yolov5 to apply weights for classes

  - Update the `__init__` method of the `ComputeLoss` class in the `loss.py` script

    ```python
    def __init__(self, model, autobalance=False, applyWeights=False, classOneWeight=1.0, classTwoWeight=1.0, classThreeWeight=1.0):
    	# ...
    	self.device = device
        self.applyWeights = applyWeights
            
        self.class_weights = torch.tensor([1.0] * self.nc, device=device)  # Default weights
        if applyWeights:
        	self.class_weights[:3] = torch.tensor([classOneWeight, classTwoWeight, classThreeWeight], device=device)
    ```

    - Add weight controls to constructor parameters
    - Assign weights if applied

  - Update the `__call__` method of the `ComputeLoss` class in the `loss.py` script

    ```python
    def __call__(self, p, targets):
        for i, pi in enumerate(p):
      	# ...
        	if n:
            	# ...
        		if self.nc > 1:
    				t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    if self.applyWeights:
                    	weight = self.class_weights[tcls[i]]
                        lcls += (self.BCEcls(pcls, t) * weight).mean()
    				else:
                    	lcls += self.BCEcls(pcls, t).mean()
               
    ```

  - Update the `parse_opt(known=False)` function in `train.py` to handle the new parameters:

    ```python
    parser.add_argument('--applyWeights', action='store_true', help='Apply custom class weights')
    parser.add_argument('--classOneWeight', type=float, default=1.0, help='Weight for class one')
    parser.add_argument('--classTwoWeight', type=float, default=1.0, help='Weight for class two')
    parser.add_argument('--classThreeWeight', type=float, default=1.0, help='Weight for class three')
    ```

  - Update the `train` function within yolov5's `train.py` script with:

    ```python
    def train(hyp, opt, device, callbacks):
        # ...
        compute_loss = ComputeLoss(model, autobalance=True, applyWeights=opt.applyWeights, classOneWeight=opt.classOneWeight, classTwoWeight=opt.classTwoWeight, classThreeWeight=opt.classThreeWeight)
    	# ...
        for epoch in range(start_epoch, epochs):
            # ...
            for i, (imgs, targets, paths, _) in pbar:
                # ...
                class_weights = None
                if opt.applyWeights:
                    class_weights = {
                        0: opt.classOneWeight,
                        1: opt.classTwoWeight,
                        2: opt.classThreeWeight,
                    }
    
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)
                    loss, loss_items = compute_loss(pred, targets.to(device))
                    if RANK != -1:
                        loss *= WORLD_SIZE
                    if opt.quad:
                        loss *= 4.0
    			# ...
    ```



# Apply Weights to Classes

- Get the weights to apply to each class (makes up for classes with less instances)

  ```python
  import numpy as np
  
  # Number of instances for each class
  n_rbc = 4155 # total number of RBC images
  n_wbc = 372 # total number of WBC images
  n_platelets = 361 # total number of Platelets images
  
  total = n_rbc + n_wbc + n_platelets
  
  # Inverse frequency weights
  weight_rbc = total / n_rbc
  weight_wbc = total / n_wbc
  weight_platelets = total / n_platelets
  
  # Normalize so the sum of weights is 1
  weights = np.array([weight_rbc, weight_wbc, weight_platelets])
  weights = weights / weights.sum()
  
  print(f'Class weights: {weights}')
  ```

- Make sure that the desired weights are present in the *train_yolo_model.py* script

  ```python
  models = [
      {'name': 'yolov5s', 'cfg': 'yolov5s.yaml', 'epochs': 10, 'img_size': 320, 'applyWeights': True, 'classOneWeight': 0.04223142, 'classTwoWeight': 0.47169776, 'classThreeWeight': 0.48607082}
  ]
  ```

  

# Test the Model

- Run the script

  ```
  >> py train_yolo_model.py
  ```

- Evaluate the results

  - Logged metrics

    | **Class** | Images | Instances | Precision | Recall | Mean Avg Precision (Intersection over Union threshold of 0.5) | mAP (IoU thresholds from 0.5 to 0.95) |
    | --------- | ------ | --------- | --------- | ------ | ------------------------------------------------------------ | ------------------------------------- |
    | all       | 314    | 4,403     | .653      | .736   | .717                                                         | .433                                  |
    | RBC       | 314    | 3,765     | .593      | .87    | .772                                                         | .477                                  |
    | WBC       | 314    | 320       | .909      | .905   | .96                                                          | .65                                   |
    | Platelets | 314    | 318       | .457      | .434   | .419                                                         | .172                                  |

    - 59.3% of the time the model predicts RBC, it is correct; it is often mistaking instances of other classes for instances of RBC
    - 87% of the actual RBC instances in the dataset were successfully detected; it rarely misses RBC detection
    - 77.2% is generated for the mAP@.5, indicating balance between the precision / recall tradeoff

  - Generated confusion matrix within `~/projects/blood-detection-app/yolov5/runs/train/yolov5s17/confusion_matrix.png`

    - Indicates that most instances of other classes (WBC and Platelets) are mistakenly classified as RBC

  

  # Conclusion

  - This system's biggest strength is that it doesn't often miss images of Red Blood Cells
    - Best used when identification of red blood cells is crucial and false positives are acceptable
  - The system's imbalance of data is a contributing factor to the poor classification ability around WBCs and Platelets
    - Can be improved by supplementing missing data, implementing ensemble models, updating hyperparameters, applying data augmentation, etc.

