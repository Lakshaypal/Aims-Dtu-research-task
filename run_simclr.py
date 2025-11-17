# run_simclr.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import copy
from PIL import Image
from tqdm import tqdm
import timm
from sklearn.metrics import f1_score

# --- Global Configuration Variables ---
device = None
BASE_DATA_PATH = '/Users/laksh/Documents/aims_dtu_ssl_project/data/ssl_dataset' # Update if necessary
TRAIN_FOLDER_PREFIX = "train.X"
VAL_FOLDER_NAME = "val.X"
LABELS_JSON_NAME = "Labels.json"
LABELS_JSON_PATH = ""
train_sub_folders = []
VAL_DIR_ACTUAL = ""

NUM_CLASSES = 0
class_to_idx = {}
idx_to_class_name = {}
sorted_class_ids = []

IMG_SIZE = 224
PRETRAIN_BATCH_SIZE_SIMCLR = 32
LINEAR_PROBE_BATCH_SIZE = 64
# CRITICAL: Increase PRETRAIN_EPOCHS_SIMCLR for good results (e.g., 100, 200, 400+)
PRETRAIN_EPOCHS_SIMCLR = 5  # Placeholder, increase for actual training
LINEAR_PROBE_EPOCHS = 10
LR_PRETRAIN_SIMCLR = 1e-4
LR_LINEAR_PROBE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE_SIMCLR = 0.07
PROJECTION_DIM_SIMCLR = 128
NUM_WORKERS = 2 # Adjust based on your system

# Path for the SimCLR pretrained backbone
PRETRAINED_SIMCLR_BACKBONE_PATH = 'simclr_resnet18_backbone_pretrained.pth'

# History dictionary for SimCLR
history_simclr = {
    'simclr_pretrain_loss': [],
    'simclr_linear_probe_train_loss': [], 'simclr_linear_probe_val_loss': [],
    'simclr_linear_probe_train_acc': [], 'simclr_linear_probe_val_acc': [],
    'simclr_linear_probe_val_f1': [],
}

# --- Class Definitions ---
class SimCLRTransform:
    def __init__(self, size):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size), T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x): return self.transform(x), self.transform(x)

class SimCLRModel(nn.Module):
    def __init__(self, backbone_name='resnet18', projection_dim=PROJECTION_DIM_SIMCLR):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        backbone_out_features = self.backbone.feature_info.channels()[-1] # More robust way to get features

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_out_features, backbone_out_features), nn.ReLU(),
            nn.Linear(backbone_out_features, projection_dim)
        )
    def forward(self, x):
        features = self.backbone(x)
        pooled_features = self.adaptive_pool(features)
        flattened_features = torch.flatten(pooled_features, 1)
        z = self.projection_head(flattened_features)
        return flattened_features, z # Return flattened_features for potential other uses, z for contrastive loss

class NTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE_SIMCLR, device_beschikbaar=None): # Pass device
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.device = device_beschikbaar # Store device

    def forward(self, z_i, z_j):
        batch_size_actual = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size_actual)
        sim_j_i = torch.diag(sim, -batch_size_actual)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size_actual, 1)
        
        # Use self.device for mask creation
        mask_device = z.device if self.device is None else self.device
        mask = torch.ones((2 * batch_size_actual, 2 * batch_size_actual), dtype=torch.bool, device=mask_device)
        mask.fill_diagonal_(False)
        arange_device = torch.arange(batch_size_actual, device=mask_device)
        mask[arange_device, arange_device + batch_size_actual] = False
        mask[arange_device + batch_size_actual, arange_device] = False
        
        negative_samples = sim[mask].reshape(2 * batch_size_actual, -1)
        labels = torch.zeros(2 * batch_size_actual, device=mask_device).long() # Use mask_device
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= (2 * batch_size_actual)
        return loss

class LinearClassifier(nn.Module):
    def __init__(self, backbone_state_dict_path, backbone_name='resnet18', num_classes_local=100, img_size_local=IMG_SIZE):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='')
        try:
            state_dict = torch.load(backbone_state_dict_path, map_location=device) # device is global
            self.backbone.load_state_dict(state_dict)
            print(f"Successfully loaded pretrained backbone weights from {backbone_state_dict_path}")
        except Exception as e:
            print(f"ERROR loading backbone state_dict for Linear Classifier: {e}. Using a randomly initialized backbone.")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        # Get output features dynamically from the loaded backbone
        try:
            dummy_input = torch.randn(1, 3, img_size_local, img_size_local).to(device)
            with torch.no_grad():
                dummy_output = self.adaptive_pool(self.backbone(dummy_input))
            backbone_out_features = dummy_output.shape[1]
            print(f"Dynamically determined backbone_out_features for LP: {backbone_out_features}")
        except Exception as e:
            print(f"Could not dynamically get features for LP, using default 512. Error: {e}")
            backbone_out_features = 512 # Fallback for resnet18

        self.classifier = nn.Linear(backbone_out_features, num_classes_local)

    def forward(self, x):
        with torch.no_grad(): # Ensure backbone is frozen
            features = self.backbone(x)
            features = self.adaptive_pool(features)
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

# ==============================================================================
# MAIN SIMCLR PIPELINE FUNCTION
# ==============================================================================
def main_simclr_pipeline():
    global device, LABELS_JSON_PATH, train_sub_folders, VAL_DIR_ACTUAL, NUM_CLASSES, class_to_idx, idx_to_class_name, sorted_class_ids, history_simclr, PRETRAINED_SIMCLR_BACKBONE_PATH

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    LABELS_JSON_PATH = os.path.join(BASE_DATA_PATH, LABELS_JSON_NAME)
    train_sub_folders = sorted([os.path.join(BASE_DATA_PATH, d) for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d)) and d.startswith(TRAIN_FOLDER_PREFIX)])
    VAL_DIR_ACTUAL = os.path.join(BASE_DATA_PATH, VAL_FOLDER_NAME)
    print(f"Data source: {BASE_DATA_PATH}")

    try:
        with open(LABELS_JSON_PATH, 'r') as f: labels_map_raw = json.load(f)
        if not isinstance(labels_map_raw, dict): raise ValueError("Labels.json content is not a dictionary.")
        if len(labels_map_raw) != 100: print(f"Warning: Expected 100 classes based on ImageNet-100, but found {len(labels_map_raw)} in Labels.json.")
        
        sorted_class_ids_from_json = sorted(labels_map_raw.keys()) # Avoid conflict with global sorted_class_ids
        class_to_idx.update({cid: idx for idx, cid in enumerate(sorted_class_ids_from_json)})
        idx_to_class_name.update({idx: labels_map_raw[cid].split(',')[0].strip() for idx, cid in enumerate(sorted_class_ids_from_json)})
        NUM_CLASSES = len(sorted_class_ids_from_json)
        sorted_class_ids.extend(sorted_class_ids_from_json) # Now update global
        
        print(f"Successfully loaded Labels.json. Number of classes: {NUM_CLASSES}")
        if NUM_CLASSES == 0: raise ValueError("Number of classes is 0 after loading Labels.json. Check the file.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Labels.json not found at {LABELS_JSON_PATH}. Exiting.")
        exit()
    except Exception as e:
        print(f"CRITICAL ERROR loading or processing Labels.json: {e}. Exiting.")
        exit()

    SKIP_SIMCLR_PRETRAINING = False # Set to True to load existing backbone and skip pretraining
                                  # For a fresh run, set to False

    if not SKIP_SIMCLR_PRETRAINING:
        print("\n--- Starting SimCLR Pretraining ---")
        simclr_pretrain_datasets_list = []
        for train_part_path in train_sub_folders:
            try:
                dataset_part = ImageFolder(train_part_path, transform=SimCLRTransform(IMG_SIZE))
                if len(dataset_part) > 0: simclr_pretrain_datasets_list.append(dataset_part)
                else: print(f"Warning: No images found in {train_part_path}")
            except Exception as e: print(f"Error loading SimCLR pretrain data from {train_part_path}: {e}")
        
        if not simclr_pretrain_datasets_list: 
            print("CRITICAL ERROR: No data loaded for SimCLR pretraining. Check dataset paths and structure. Exiting.")
            exit()
        simclr_pretrain_full_dataset = ConcatDataset(simclr_pretrain_datasets_list)
        simclr_pretrain_loader = DataLoader(simclr_pretrain_full_dataset, batch_size=PRETRAIN_BATCH_SIZE_SIMCLR, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False, drop_last=True) # pin_memory true for cuda
        print(f"SimCLR pretrain dataset size: {len(simclr_pretrain_full_dataset)} images.")
        
        simclr_model_instance = SimCLRModel(backbone_name='resnet18', projection_dim=PROJECTION_DIM_SIMCLR).to(device)
        simclr_loss_fn_instance = NTXentLoss(temperature=TEMPERATURE_SIMCLR, device_beschikbaar=device).to(device) # Pass device
        simclr_optimizer = optim.AdamW(simclr_model_instance.parameters(), lr=LR_PRETRAIN_SIMCLR, weight_decay=WEIGHT_DECAY)
        
        # Optional: Learning Rate Scheduler for long pretraining runs
        # simclr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(simclr_optimizer, T_max=len(simclr_pretrain_loader) * PRETRAIN_EPOCHS_SIMCLR)

        start_time_simclr_pretrain = time.time()
        print(f"Starting SimCLR pretraining for {PRETRAIN_EPOCHS_SIMCLR} epochs...")
        for epoch in range(PRETRAIN_EPOCHS_SIMCLR):
            simclr_model_instance.train()
            epoch_loss = 0.0
            progress_bar = tqdm(simclr_pretrain_loader, desc=f"SimCLR Epoch {epoch+1}/{PRETRAIN_EPOCHS_SIMCLR}")
            for (images_tuple, _) in progress_bar: # Labels from ImageFolder are ignored
                images_i, images_j = images_tuple
                images_i = images_i.to(device, non_blocking=True if device.type != 'mps' else False)
                images_j = images_j.to(device, non_blocking=True if device.type != 'mps' else False)
                
                simclr_optimizer.zero_grad()
                _, z_i = simclr_model_instance(images_i)
                _, z_j = simclr_model_instance(images_j)
                loss = simclr_loss_fn_instance(z_i, z_j)
                loss.backward()
                simclr_optimizer.step()
                # if simclr_scheduler: simclr_scheduler.step() # If using scheduler per iteration
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_epoch_loss = epoch_loss / len(simclr_pretrain_loader)
            history_simclr['simclr_pretrain_loss'].append(avg_epoch_loss)
            print(f"SimCLR Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
            # if simclr_scheduler: # If using scheduler per epoch
            #    current_lr = simclr_scheduler.get_last_lr()[0]
            #    print(f"Current LR: {current_lr:.6f}")
            #    simclr_scheduler.step()


        print(f"SimCLR Pretraining Finished. Total time: {(time.time()-start_time_simclr_pretrain)/60:.2f} minutes.")
        torch.save(simclr_model_instance.backbone.state_dict(), PRETRAINED_SIMCLR_BACKBONE_PATH)
        print(f"SimCLR pretrained backbone saved to {PRETRAINED_SIMCLR_BACKBONE_PATH}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_simclr['simclr_pretrain_loss'], label='SimCLR Pretraining Loss')
        plt.title('SimCLR Pretraining Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('simclr_pretrain_loss_curve.png')
        print("SimCLR pretraining loss curve saved to simclr_pretrain_loss_curve.png")
        plt.close()
    else:
        print(f"\n--- SKIPPING SimCLR Pretraining ---")
        if not os.path.exists(PRETRAINED_SIMCLR_BACKBONE_PATH):
            print(f"CRITICAL ERROR: Pretrained SimCLR backbone {PRETRAINED_SIMCLR_BACKBONE_PATH} not found, but skipping pretraining was enabled. Exiting.")
            exit()
        print(f"Using existing SimCLR backbone from: {PRETRAINED_SIMCLR_BACKBONE_PATH}")

    # ==============================================================================
    # SECTION 2: LINEAR PROBING FOR SIMCLR BACKBONE
    # ==============================================================================
    print("\n--- Starting Linear Probing for SimCLR Pretrained Backbone ---")
    linear_probe_train_transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    linear_probe_val_transform = T.Compose([
        T.Resize(IMG_SIZE + 32), # Standard practice: slightly larger resize then center crop
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    
    lp_train_datasets_list = []
    for train_part_path in train_sub_folders:
        try:
            dataset_part = ImageFolder(train_part_path, transform=linear_probe_train_transform)
            if len(dataset_part) > 0: lp_train_datasets_list.append(dataset_part)
        except Exception as e: print(f"Error loading LP train data from {train_part_path}: {e}")
    
    if not lp_train_datasets_list: 
        print("CRITICAL ERROR: No data loaded for SimCLR Linear Probing (train). Exiting.")
        exit()
    lp_train_full_dataset = ConcatDataset(lp_train_datasets_list)
    lp_train_loader = DataLoader(lp_train_full_dataset, batch_size=LINEAR_PROBE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"SimCLR LP train dataset size: {len(lp_train_full_dataset)} images.")
    
    try:
        lp_val_dataset = ImageFolder(VAL_DIR_ACTUAL, transform=linear_probe_val_transform)
        if len(lp_val_dataset) == 0: raise ValueError("Validation dataset is empty.")
    except Exception as e: 
        print(f"CRITICAL ERROR: Could not load or val data empty from {VAL_DIR_ACTUAL}: {e}. Exiting.")
        exit()
    lp_val_loader = DataLoader(lp_val_dataset, batch_size=LINEAR_PROBE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"SimCLR LP val dataset size: {len(lp_val_dataset)} images.")

    simclr_lp_model = LinearClassifier(
        backbone_state_dict_path=PRETRAINED_SIMCLR_BACKBONE_PATH,
        backbone_name='resnet18', # Ensure this matches the pretrained backbone
        num_classes_local=NUM_CLASSES,
        img_size_local=IMG_SIZE
    ).to(device)
    
    lp_criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the classifier head
    lp_optimizer = optim.AdamW(simclr_lp_model.classifier.parameters(), lr=LR_LINEAR_PROBE, weight_decay=WEIGHT_DECAY)
    
    print(f"\n--- Starting SimCLR Linear Probe Training for {LINEAR_PROBE_EPOCHS} epochs ---")
    start_time_simclr_lp = time.time()
    
    best_val_f1 = 0.0 # For saving best model if desired

    for epoch in range(LINEAR_PROBE_EPOCHS):
        simclr_lp_model.train() # Set classifier to train mode, backbone is already in eval and frozen
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(lp_train_loader, desc=f"SimCLR LP Train Epoch {epoch+1}/{LINEAR_PROBE_EPOCHS}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device, non_blocking=True if device.type != 'mps' else False), labels.to(device, non_blocking=True if device.type != 'mps' else False)
            lp_optimizer.zero_grad()
            outputs = simclr_lp_model(inputs)
            loss = lp_criterion(outputs, labels)
            loss.backward()
            lp_optimizer.step()
            
            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss_train / total_train
        epoch_train_acc = correct_train / total_train
        history_simclr['simclr_linear_probe_train_loss'].append(epoch_train_loss)
        history_simclr['simclr_linear_probe_train_acc'].append(epoch_train_acc)
        
        simclr_lp_model.eval() # Set to eval mode for validation
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        val_bar = tqdm(lp_val_loader, desc=f"SimCLR LP Val Epoch {epoch+1}/{LINEAR_PROBE_EPOCHS}")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device, non_blocking=True if device.type != 'mps' else False), labels.to(device, non_blocking=True if device.type != 'mps' else False)
                outputs = simclr_lp_model(inputs)
                loss = lp_criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                val_bar.set_postfix({'loss': loss.item()})
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        
        history_simclr['simclr_linear_probe_val_loss'].append(epoch_val_loss)
        history_simclr['simclr_linear_probe_val_acc'].append(epoch_val_acc)
        history_simclr['simclr_linear_probe_val_f1'].append(epoch_val_f1)
        
        print(f"SimCLR LP Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}")

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            # torch.save(simclr_lp_model.state_dict(), 'simclr_lp_best_model.pth') # Optional: save best LP model
            # print(f"New best F1: {best_val_f1:.4f}. LP model saved.")

    print(f"SimCLR Linear Probing Finished. Total time: {(time.time()-start_time_simclr_lp)/60:.2f} minutes.")
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history_simclr['simclr_linear_probe_train_loss'], label='Train Loss')
    plt.plot(history_simclr['simclr_linear_probe_val_loss'], label='Val Loss')
    plt.title('SimCLR LP Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_simclr['simclr_linear_probe_train_acc'], label='Train Acc')
    plt.plot(history_simclr['simclr_linear_probe_val_acc'], label='Val Acc')
    plt.title('SimCLR LP Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history_simclr['simclr_linear_probe_val_f1'], label='Val F1 Score')
    plt.title('SimCLR LP F1 Score')
    plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simclr_linear_probe_curves.png')
    print("SimCLR linear probing curves saved to simclr_linear_probe_curves.png")
    plt.close()

    print("\nSimCLR Script Finished.")

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    main_simclr_pipeline()