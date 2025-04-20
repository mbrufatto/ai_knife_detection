# -----------------------------------------------------------------------------
# Passo 1: Instalação dependências
# -----------------------------------------------------------------------------
# pip install ultralytics
#
# Bibliotecas necessárias para baixar dataset COCO
# pip install fiftyone
# 
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# -----------------------------------------------------------------------------
# Passo 2: Importação bibliotecas
# -----------------------------------------------------------------------------
import os
import cv2
import time
import shutil
import ultralytics

from ultralytics import YOLO

# Importar as bibliotecas para formação dataset COCO
import fiftyone as fo
import fiftyone.zoo as foz

from pathlib import Path

# Diretório base para o dataset
# Substitua pelo caminho desejado no seu sistema
base_dir = "\\Dev\\techchallenge\\fase05_gpu\\datasets\\coco_cutlery_gpu"

os.makedirs(base_dir, exist_ok=True)
classes = ["fork", "knife", "scissors"]

ultralytics.checks()

# -----------------------------------------------------------------------------
# Passo 3: Download datasets COCO:
# cup, fork, knife, spoon, scissors, hair drier, toothbrush
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Passo 3.1: Baixar e exportar dataset COCO filtrado
# -----------------------------------------------------------------------------
def download_and_export_dataset(split, max_samples, base_dir, classes):
    """
    Downloads and exports COCO dataset for a specific split
    """
    print(f"Baixando conjunto de {split}...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples
        #overwrite=True
    )

    export_dir = os.path.join(base_dir, "train" if split == "train" else "val")
    print(f"Exportando {split}...")
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth"
    )

# -----------------------------------------------------------------------------
# Passo 3.2: Criar YAML para treino do YOLO
# -----------------------------------------------------------------------------
def create_yaml_config(base_dir, class_names):
    """
    Creates YAML configuration file for YOLO training
    Args:
        base_dir: Base directory path
        class_names: List of class names
    """
    yaml_path = os.path.join(base_dir, "coco_cutlery_gpu.yaml")
    
    # Create class name dictionary
    names_dict = {i: name for i, name in enumerate(class_names)}
    
    # Format YAML content
    yaml_content = {
        'path': base_dir,
        'train': 'train/data',
        'val': 'val/data',
        'names': names_dict
    }
    
    try:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)
        print(f"YAML configuration saved to: {yaml_path}")
    except Exception as e:
        print(f"Error creating YAML file: {e}")
        raise


import json
import os
from pathlib import Path
from tqdm import tqdm
import yaml

# ----------------------------
# Passo 4: Configurações para converter COCO JSON para YOLO format com filtro de classes
# ----------------------------

# ----------------------------
# Carregar JSON do COCO
# ----------------------------
def load_coco_json(json_path):
    """Load and parse COCO JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_category_mappings(coco_data, wanted_classes):
    """Create category ID to name mappings"""
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    name_to_new_id = {name: idx for idx, name in enumerate(wanted_classes)}
    return category_id_to_name, name_to_new_id

def group_annotations_by_image(coco_data, category_id_to_name, wanted_classes):
    """Group annotations by image ID, filtering for wanted classes"""
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        category_name = category_id_to_name[ann['category_id']]
        if category_name in wanted_classes:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
    return annotations_by_image

def convert_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO bbox to YOLO format"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def convert_coco_to_yolo(json_path, images_dir, labels_dir, wanted_classes):
    
    """Convert a COCO-format JSON file to YOLO-format .txt files"""
    os.makedirs(labels_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    name_to_new_id = {name: idx for idx, name in enumerate(wanted_classes)}

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        cat_name = category_id_to_name[ann['category_id']]
        if cat_name not in wanted_classes:
            continue
        image_id = ann['image_id']
        annotations_by_image.setdefault(image_id, []).append(ann)

    images_by_id = {img['id']: img for img in coco_data['images']}

    for image_id, anns in tqdm(annotations_by_image.items(), desc=f"Convertendo {os.path.basename(labels_dir)}"):
        img = images_by_id[image_id]
        file_stem = Path(img['file_name']).stem
        label_path = os.path.join(labels_dir, f"{file_stem}.txt")
        w, h = img['width'], img['height']

        lines = []
        for ann in anns:
            cat_name = category_id_to_name[ann['category_id']]
            class_id = name_to_new_id[cat_name]
            x, y, bw, bh = ann['bbox']
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw/w:.6f} {bh/h:.6f}")

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

    print(f"✅ Labels YOLO salvos em: {labels_dir}")



if __name__ == "__main__":
    # Download and export training data
    download_and_export_dataset("train", max_samples=5000, base_dir=base_dir, classes=classes)
    # Download and export validation data  
    download_and_export_dataset("validation", max_samples=2000, base_dir=base_dir, classes=classes)
    
    # Convert annotations for train
    train_json = os.path.join(base_dir, "annotations", "instances_train2017.json")
    train_images = os.path.join(base_dir, "train", "data")
    train_labels = os.path.join(base_dir, "train", "data")
    convert_coco_to_yolo(train_json, train_images, train_labels, classes)

    # Convert annotations for val
    val_json = os.path.join(base_dir, "annotations", "instances_val2017.json")
    val_images = os.path.join(base_dir, "val", "data")
    val_labels = os.path.join(base_dir, "val", "data")
    convert_coco_to_yolo(val_json, val_images, val_labels, classes)

    create_yaml_config(base_dir, classes)

    yaml_path = os.path.join(base_dir, "coco_cutlery_gpu.yaml")

    # Treinamento
    model = YOLO("yolo11s.pt")

    model.train(
        data=yaml_path,                # arquivo YAML com paths e classes
        epochs=40,                     # mais épocas para maior convergência
        imgsz=640,                     # bom equilíbrio entre qualidade e tempo
        batch=8,                       # ideal para GPU de 4GB (evita OOM)
        device="cuda",                 # garante uso da GPU
        workers=2,                     # menor número para estabilidade em máquinas com limitação de RAM
        optimizer="AdamW",             # otimizador rápido e estável
        amp=True,                      # mixed precision: acelera e economiza memória
        augment=True,                  # aumenta variedade de dados no treinamento
        lr0=0.002,                     # taxa de aprendizado inicial ligeiramente menor para mais estabilidade
        weight_decay=0.0005,           # regularização leve (evita overfitting)
        patience=15,                   # early stopping com mais tolerância
        warmup_epochs=3,               # ajuda a estabilizar o início do treino
        name="yolo11_cutlery_gpu",     # nome para a execução
        project="coco_cutlery_gpu",    # pasta onde salvar os resultados
        exist_ok=True                  # permite sobrescrever diretórios existentes
    )

    print("Treinamento finalizado!")