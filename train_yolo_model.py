import os
import subprocess

yolov5_repo = 'C:/Users/your-username/projects/blood-cell-detection-app/yolov5'  
dataset_yaml = 'C:/Users/your-username/projects/blood-cell-detection-app/data/bccd.yaml'  

models = [
    {'name': 'yolov5s', 'cfg': 'yolov5s.yaml', 'epochs': 10, 'img_size': 320, 'applyWeights': True, 'classOneWeight': 0.04223142, 'classTwoWeight': 0.47169776, 'classThreeWeight': 0.48607082}
]

def train_model(model_cfg):
    model_name = model_cfg['name']
    cfg_file = model_cfg['cfg']
    epochs = model_cfg['epochs']
    img_size = model_cfg['img_size']
    hyp_flag = f"--hyp {model_cfg['hyp']}" if 'hyp' in model_cfg else ""

    apply_weights_flag = "--applyWeights" if model_cfg['applyWeights'] else ""
    class_one_weight = f"--classOneWeight {model_cfg['classOneWeight']}" if 'classOneWeight' in model_cfg else ""
    class_two_weight = f"--classTwoWeight {model_cfg['classTwoWeight']}" if 'classTwoWeight' in model_cfg else ""
    class_three_weight = f"--classThreeWeight {model_cfg['classThreeWeight']}" if 'classThreeWeight' in model_cfg else ""
    
    command = (
        f"python {os.path.join(yolov5_repo, 'train.py')} "
        f"--img {img_size} "
        f"--batch 8 "
        f"--epochs {epochs} "
        f"--data {dataset_yaml} "
        f"--cfg {os.path.join(yolov5_repo, 'models', cfg_file)} "
        f"--weights {model_name}.pt "
        f"--name {model_name} "
        f"{hyp_flag} "
        f"{apply_weights_flag} "
        f"{class_one_weight} "
        f"{class_two_weight} "
        f"{class_three_weight} "
        f"--cache "
    )
    
    print(f"Training {model_name}...")
    subprocess.run(command, shell=True)
    print(f"Finished training {model_name}!\n")

for model in models:
    train_model(model)
