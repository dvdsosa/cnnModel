import torch
import faiss
import sqlite3
import numpy as np
from torchvision import transforms, datasets
from SupContrast.networks.resnet_big import SupConResNet
from tqdm import tqdm
import time
import os
import torch.backends.cudnn as cudnn
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
)

class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomDataset, self).__getitem__(index)
        path, _ = self.samples[index]
        class_name = self.classes[original_tuple[1]]
        return original_tuple[0], class_name

def set_loader():
    mean = (0.0425, 0.0436, 0.0432)
    std = (0.1466, 0.1488, 0.1476)
    normalize = transforms.Normalize(mean=mean, std=std)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    custom_val_dataset = CustomDataset(root='/home/dsosatr/ofa/segment-anything/output/sam2_test_masks_yes/', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        custom_val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)
    return val_loader

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_model(ckpt_path, model_name='resnet50timm'):
    model = SupConResNet(name=model_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    # Remove 'module.' if present
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    cudnn.benchmark = True
    return model

def get_label_from_db(cursor, faiss_id):
    cursor.execute('SELECT label FROM feature_mappings WHERE faiss_id=?', (int(faiss_id),))
    row = cursor.fetchone()
    return row[0] if row else None

def main():
    FAISS_INDEX_PATH = 'faiss_index_stage2_pruned.bin'
    SQLITE_DB_PATH = 'plankton_db_stage2_pruned.sqlite'
    CKPT_PATH = '/home/dsosatr/tesis/cnnmodel/SupContrast/pesos/resnet50_stage2.pth'
    MODEL_NAME = 'resnet50timm'

    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    model = load_model(CKPT_PATH, MODEL_NAME)
    val_loader = set_loader()

    preds = []
    targets = []
    class_to_idx = val_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model.eval()
    start_time = time.time()  # Start timing
    with torch.no_grad():
        for images, real_labels in tqdm(val_loader, desc="Processing"):
            images = images.float().cuda()
            features = model.encoder(images)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            feature_np = features.cpu().numpy().astype(np.float32)
            D, I = faiss_index.search(feature_np, 1)
            faiss_id = int(I[0][0])
            result_queried = get_label_from_db(cursor, faiss_id)
            real_result = real_labels[0]
            #print(f"Inferenced image class: {result_queried} - Real image class {real_result}")

            # Store predictions and targets as integer indices
            if result_queried in class_to_idx:
                preds.append(class_to_idx[result_queried])
            else:
            # If label not found, append a dummy class or -1
                preds.append(-1)
            targets.append(class_to_idx[real_result])

    conn.close()
    end_time = time.time()  # End timing

    # Compute metrics using torcheval
    preds_tensor = torch.tensor(preds)
    targets_tensor = torch.tensor(targets)
    num_classes = len(class_to_idx)

    acc = multiclass_accuracy(preds_tensor, targets_tensor, num_classes=num_classes, average="micro", k=1)
    prec = multiclass_precision(preds_tensor, targets_tensor, num_classes=num_classes, average="macro")
    rec = multiclass_recall(preds_tensor, targets_tensor, num_classes=num_classes, average="macro")
    f1 = multiclass_f1_score(preds_tensor, targets_tensor, num_classes=num_classes, average="macro")

    print(f"Multiclass Accuracy: {acc.item() * 100:.2f}")
    print(f"Multiclass Precision (macro): {prec.item() * 100:.2f}")
    print(f"Multiclass Recall (macro): {rec.item() * 100:.2f}")
    print(f"Multiclass F1 Score (macro): {f1.item() * 100:.2f}")
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    
    # Calculate and print the mean processing time per image
    total_images = len(val_loader.dataset)
    mean_time_per_image = (end_time - start_time) / total_images
    print(f"Mean processing time per image: {mean_time_per_image:.4f} seconds")

    # Print the size of the FAISS index file
    faiss_index_size = os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024)  # Convert bytes to MB
    print(f"FAISS index file size: {faiss_index_size:.2f} MB")

if __name__ == '__main__':
    main()
