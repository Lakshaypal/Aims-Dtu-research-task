# run_moco.py

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
# class_to_idx = {} # Will be populated locally in main function
# idx_to_class_name = {} # If needed for detailed label names
# sorted_class_ids = [] # If needed

IMG_SIZE = 224 # Standard for ImageNet
NUM_WORKERS = 2 # Adjust based on your system

# --- MoCo v2 Specific Configurations ---
MOCO_BACKBONE_NAME = 'resnet18' # 'resnet50' is common too
MOCO_DIM = 128  # Output dimension of projection head
MOCO_K = 16384  # Size of the dictionary queue (official uses 65536, adjust if needed)
MOCO_M = 0.999  # Momentum for updating key encoder
MOCO_T = 0.07   # Temperature for softmax

# !!! CRITICAL FOR TODAY'S DEADLINE: USE EXTREMELY LOW EPOCH COUNT IF RUNNING !!!
MOCO_PRETRAIN_EPOCHS = 5 # Changed based on your last request, still high risk for today
MOCO_PRETRAIN_BATCH_SIZE = 32 # Adjust based on memory for ResNet18
MOCO_LR_PRETRAIN = 0.03 # SGD LR, common for MoCo
MOCO_WEIGHT_DECAY_PRETRAIN = 1e-4
MOCO_SGD_MOMENTUM = 0.9

# Linear Probing settings for MoCo
LINEAR_PROBE_BATCH_SIZE_MOCO = 64
LINEAR_PROBE_EPOCHS_MOCO = 5 # Changed based on your last request
LR_LINEAR_PROBE_MOCO = 1e-3 # Or adjust based on other LP LRs
WEIGHT_DECAY_LP_MOCO = 1e-4

# Path for the MoCo pretrained backbone (encoder_q)
PRETRAINED_MOCO_BACKBONE_PATH = 'moco_resnet18_encoder_q_pretrained.pth'

# History dictionary for MoCo
history_moco = {
    'moco_pretrain_loss': [],
    'moco_linear_probe_train_loss': [], 'moco_linear_probe_val_loss': [],
    'moco_linear_probe_train_acc': [], 'moco_linear_probe_val_acc': [],
    'moco_linear_probe_val_f1': [],
}

# --- MoCo v2 Data Augmentation ---
class MoCoTransform:
    def __init__(self, size):
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=int(0.1 * size)//2*2+1, sigma=(0.1, 2.0))], p=0.5), # Ensure kernel is odd
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return q, k

# --- MoCo_V2 Model Definition ---
class MoCo_V2(nn.Module):
    def __init__(self, backbone_name=MOCO_BACKBONE_NAME, dim=MOCO_DIM, K=MOCO_K, m=MOCO_M, T=MOCO_T, device_for_moco=None):
        super(MoCo_V2, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device_for_moco = device_for_moco

        self.encoder_q = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='')
        self.encoder_k = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='')

        # --- CORRECTED backbone_out_features ---
        try:
            # This is common for CNNs like ResNet from timm
            backbone_out_features = self.encoder_q.feature_info[-1]['num_chs']
        except Exception as e_info:
            print(f"MoCo Model: Could not get feature_info[-1]['num_chs'] for {backbone_name}. Error: {e_info}")
            print(f"MoCo Model: Attempting self.encoder_q.num_features...")
            try:
                 # This is common for ViTs, but sometimes available for CNNs too (often after global_pool if not '')
                 backbone_out_features = self.encoder_q.num_features
                 if backbone_out_features == 0 and backbone_name.startswith('resnet'): # num_features might be 0 if global_pool=''
                     print("MoCo Model: num_features is 0, likely due to global_pool=''. Falling back for ResNet.")
                     raise AttributeError # Force fallback for ResNet types if num_features is 0
            except AttributeError:
                print(f"MoCo Model: self.encoder_q.num_features not found or unsuitable for {backbone_name}.")
                if 'resnet18' in backbone_name: default_dim = 512
                elif 'resnet50' in backbone_name: default_dim = 2048
                else: default_dim = 512 # A general fallback
                print(f"MoCo Model: Defaulting backbone_out_features to {default_dim} for {backbone_name}.")
                backbone_out_features = default_dim
        print(f"MoCo Model: Determined backbone_out_features: {backbone_out_features} for {backbone_name}")
        # --- END OF CORRECTION ---


        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector_q = nn.Sequential(nn.Linear(backbone_out_features, backbone_out_features), nn.ReLU(), nn.Linear(backbone_out_features, dim))
        self.projector_k = nn.Sequential(nn.Linear(backbone_out_features, backbone_out_features), nn.ReLU(), nn.Linear(backbone_out_features, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys): # keys: (batch_size, dim)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        assert self.K % batch_size == 0, f"Queue size K ({self.K}) must be divisible by batch_size ({batch_size}) for this simplified implementation."
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q_feat = self.adaptive_pool(self.encoder_q(im_q))
        q = self.projector_q(torch.flatten(q_feat, 1))
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_feat = self.adaptive_pool(self.encoder_k(im_k))
            k = self.projector_k(torch.flatten(k_feat, 1))
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        current_device = self.device_for_moco if self.device_for_moco else logits.device
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=current_device)
        
        self._dequeue_and_enqueue(k)
        return logits, targets

# --- Linear Classifier (can be reused from SimCLR/MAE if backbone matches) ---
class LinearClassifier(nn.Module):
    def __init__(self, backbone_state_dict_path, backbone_name=MOCO_BACKBONE_NAME, num_classes_local=100, img_size_local=IMG_SIZE):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='')
        try:
            # Ensure device is defined globally or passed appropriately if map_location is used here
            state_dict_device = device if 'device' in globals() and device is not None else 'cpu'
            state_dict = torch.load(backbone_state_dict_path, map_location=state_dict_device)
            self.backbone.load_state_dict(state_dict)
            print(f"LP: Loaded pretrained backbone from {backbone_state_dict_path}")
        except Exception as e:
            print(f"LP ERROR: loading backbone {backbone_state_dict_path}: {e}. Using random init.")
        
        for param in self.backbone.parameters(): param.requires_grad = False
        self.backbone.eval()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        
        try:
            # Ensure device is defined globally or passed appropriately
            current_eval_device = device if 'device' in globals() and device is not None else 'cpu'
            dummy_input = torch.randn(1, 3, img_size_local, img_size_local).to(current_eval_device)
            # If backbone is on a different device, move dummy_input there or move backbone to current_eval_device for this check
            if next(self.backbone.parameters()).device != current_eval_device:
                 self.backbone.to(current_eval_device) # Temporarily move backbone for dummy pass

            with torch.no_grad(): dummy_output = self.adaptive_pool(self.backbone(dummy_input))
            backbone_out_features = dummy_output.shape[1]
        except Exception as e:
            print(f"LP ERROR: dynamic feature dim for {backbone_name}: {e}.")
            if 'resnet18' in backbone_name: default_dim_lp = 512
            elif 'resnet50' in backbone_name: default_dim_lp = 2048
            else: default_dim_lp = 512
            print(f"LP: Defaulting backbone_out_features to {default_dim_lp} for {backbone_name}.")
            backbone_out_features = default_dim_lp
        self.classifier = nn.Linear(backbone_out_features, num_classes_local)

    def forward(self, x):
        with torch.no_grad():
            features = self.adaptive_pool(self.backbone(x))
            features = torch.flatten(features, 1)
        return self.classifier(features)

# ==============================================================================
# MAIN MOCO PIPELINE FUNCTION
# ==============================================================================
def main_moco_pipeline():
    global device, LABELS_JSON_PATH, train_sub_folders, VAL_DIR_ACTUAL, NUM_CLASSES, history_moco, PRETRAINED_MOCO_BACKBONE_PATH, MOCO_K, MOCO_PRETRAIN_BATCH_SIZE # Removed class_to_idx etc. from globals if only used locally


    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device} for MoCo")

    LABELS_JSON_PATH = os.path.join(BASE_DATA_PATH, LABELS_JSON_NAME)
    train_sub_folders = sorted([os.path.join(BASE_DATA_PATH, d) for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d)) and d.startswith(TRAIN_FOLDER_PREFIX)])
    VAL_DIR_ACTUAL = os.path.join(BASE_DATA_PATH, VAL_FOLDER_NAME)
    
    try:
        with open(LABELS_JSON_PATH, 'r') as f: labels_map_raw = json.load(f)
        NUM_CLASSES = len(labels_map_raw) 
        if NUM_CLASSES == 0: raise ValueError("NUM_CLASSES is 0 from Labels.json")
        print(f"Labels loaded for MoCo. Num classes: {NUM_CLASSES}")
    except Exception as e: print(f"CRITICAL Label loading error for MoCo: {e}"); exit()

    # Adjust K to be divisible by batch_size if needed by the simplified queue logic
    if MOCO_PRETRAIN_BATCH_SIZE > 0 and MOCO_K % MOCO_PRETRAIN_BATCH_SIZE != 0:
        print(f"Adjusting MOCO_K from {MOCO_K} to be divisible by MOCO_PRETRAIN_BATCH_SIZE {MOCO_PRETRAIN_BATCH_SIZE}")
        MOCO_K = (MOCO_K // MOCO_PRETRAIN_BATCH_SIZE) * MOCO_PRETRAIN_BATCH_SIZE
        if MOCO_K == 0 : MOCO_K = MOCO_PRETRAIN_BATCH_SIZE * 128 # Ensure K is not zero if original K was too small
        print(f"New MOCO_K: {MOCO_K}")


    SKIP_MOCO_PRETRAINING = False 

    if not SKIP_MOCO_PRETRAINING:
        print("\n--- Starting MoCo v2 Pretraining ---")
        moco_pretrain_transform = MoCoTransform(IMG_SIZE)
        moco_pretrain_datasets_list = []
        for train_part_path in train_sub_folders:
            try:
                dataset_part = ImageFolder(train_part_path, transform=moco_pretrain_transform)
                if len(dataset_part) > 0: moco_pretrain_datasets_list.append(dataset_part)
            except Exception as e: print(f"Error loading MoCo pretrain data from {train_part_path}: {e}")
        
        if not moco_pretrain_datasets_list: print("CRITICAL: No MoCo pretrain data. Exiting."); exit()
        moco_pretrain_full_dataset = ConcatDataset(moco_pretrain_datasets_list)
        
        moco_pretrain_loader = DataLoader(moco_pretrain_full_dataset, batch_size=MOCO_PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False, drop_last=True) 
        print(f"MoCo pretrain dataset: {len(moco_pretrain_full_dataset)} images. Batch size: {MOCO_PRETRAIN_BATCH_SIZE}. K: {MOCO_K}")

        model_moco = MoCo_V2(backbone_name=MOCO_BACKBONE_NAME, dim=MOCO_DIM, K=MOCO_K, m=MOCO_M, T=MOCO_T, device_for_moco=device).to(device)
        criterion_moco = nn.CrossEntropyLoss()
        optimizer_moco = optim.SGD(model_moco.parameters(), lr=MOCO_LR_PRETRAIN, momentum=MOCO_SGD_MOMENTUM, weight_decay=MOCO_WEIGHT_DECAY_PRETRAIN) 
        
        start_time_moco_pretrain = time.time()
        print(f"Starting MoCo pretraining for {MOCO_PRETRAIN_EPOCHS} epochs...")
        for epoch in range(MOCO_PRETRAIN_EPOCHS):
            model_moco.train() 
            epoch_loss = 0.0
            progress_bar = tqdm(moco_pretrain_loader, desc=f"MoCo Epoch {epoch+1}/{MOCO_PRETRAIN_EPOCHS}")
            for (images_pair, _) in progress_bar: 
                im_q, im_k = images_pair
                im_q, im_k = im_q.to(device, non_blocking=True if device.type != 'mps' else False), im_k.to(device, non_blocking=True if device.type != 'mps' else False)
                
                optimizer_moco.zero_grad()
                logits, targets = model_moco(im_q, im_k)
                loss = criterion_moco(logits, targets)
                loss.backward()
                optimizer_moco.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_epoch_loss = epoch_loss / len(moco_pretrain_loader)
            history_moco['moco_pretrain_loss'].append(avg_epoch_loss)
            print(f"MoCo Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

        print(f"MoCo Pretraining Finished: {(time.time()-start_time_moco_pretrain)/60:.2f} min")
        torch.save(model_moco.encoder_q.state_dict(), PRETRAINED_MOCO_BACKBONE_PATH) 
        print(f"MoCo encoder_q backbone saved to {PRETRAINED_MOCO_BACKBONE_PATH}")
        
        plt.figure(); plt.plot(history_moco['moco_pretrain_loss'], label='MoCo Pretrain Loss'); plt.title('MoCo Pretraining Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.savefig('moco_pretrain_loss_curve.png'); plt.close(); print("MoCo loss curve saved.")
    else:
        print("\n--- SKIPPING MoCo Pretraining ---")
        if not os.path.exists(PRETRAINED_MOCO_BACKBONE_PATH): print(f"ERROR: {PRETRAINED_MOCO_BACKBONE_PATH} not found!"); exit()
        print(f"Using existing MoCo backbone: {PRETRAINED_MOCO_BACKBONE_PATH}")

    # --- Linear Probing for MoCo ---
    print("\n--- Starting Linear Probing for MoCo Pretrained Backbone ---")
    lp_train_transform_moco = T.Compose([T.RandomResizedCrop(IMG_SIZE), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    lp_val_transform_moco = T.Compose([T.Resize(IMG_SIZE+32), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    
    lp_train_datasets_list_moco = [] 
    for train_part_path in train_sub_folders:
        try:
            dataset_part = ImageFolder(train_part_path, transform=lp_train_transform_moco)
            if len(dataset_part) > 0: lp_train_datasets_list_moco.append(dataset_part)
        except Exception as e: print(f"Error LP data MoCo: {e}")
    if not lp_train_datasets_list_moco: print("CRITICAL: No LP train data MoCo."); exit()

    lp_train_full_dataset_moco = ConcatDataset(lp_train_datasets_list_moco)
    lp_train_loader_moco = DataLoader(lp_train_full_dataset_moco, batch_size=LINEAR_PROBE_BATCH_SIZE_MOCO, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"MoCo LP train dataset: {len(lp_train_full_dataset_moco)} images.")
    
    try:
        lp_val_dataset_moco = ImageFolder(VAL_DIR_ACTUAL, transform=lp_val_transform_moco)
        if len(lp_val_dataset_moco) == 0: raise ValueError("MoCo LP Val dataset is empty.")
    except Exception as e: print(f"CRITICAL: No LP val data MoCo: {e}"); exit()
    lp_val_loader_moco = DataLoader(lp_val_dataset_moco, batch_size=LINEAR_PROBE_BATCH_SIZE_MOCO, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"MoCo LP val dataset: {len(lp_val_dataset_moco)} images.")

    moco_lp_model = LinearClassifier(PRETRAINED_MOCO_BACKBONE_PATH, backbone_name=MOCO_BACKBONE_NAME, num_classes_local=NUM_CLASSES, img_size_local=IMG_SIZE).to(device)
    lp_criterion_moco = nn.CrossEntropyLoss()
    lp_optimizer_moco = optim.AdamW(moco_lp_model.classifier.parameters(), lr=LR_LINEAR_PROBE_MOCO, weight_decay=WEIGHT_DECAY_LP_MOCO)
    
    print(f"\n--- Starting MoCo Linear Probe Training for {LINEAR_PROBE_EPOCHS_MOCO} epochs ---")
    start_time_moco_lp = time.time()
    for epoch in range(LINEAR_PROBE_EPOCHS_MOCO):
        moco_lp_model.train(); running_loss=0.0; correct_train=0; total_train=0
        train_bar = tqdm(lp_train_loader_moco, desc=f"MoCo LP Train Ep {epoch+1}/{LINEAR_PROBE_EPOCHS_MOCO}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device, non_blocking=True if device.type != 'mps' else False), labels.to(device, non_blocking=True if device.type != 'mps' else False)
            lp_optimizer_moco.zero_grad(); outputs = moco_lp_model(inputs); loss = lp_criterion_moco(outputs, labels)
            loss.backward(); lp_optimizer_moco.step(); running_loss += loss.item()*inputs.size(0)
            _, pred = torch.max(outputs.data,1); total_train += labels.size(0); correct_train += (pred==labels).sum().item()
            train_bar.set_postfix({'loss': loss.item()})
        epoch_train_loss=running_loss/total_train; epoch_train_acc=correct_train/total_train
        history_moco['moco_linear_probe_train_loss'].append(epoch_train_loss); history_moco['moco_linear_probe_train_acc'].append(epoch_train_acc)
        
        moco_lp_model.eval(); running_val_loss=0.0; correct_val=0; total_val=0; all_preds, all_labels = [], []
        val_bar = tqdm(lp_val_loader_moco, desc=f"MoCo LP Val Ep {epoch+1}/{LINEAR_PROBE_EPOCHS_MOCO}")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device,non_blocking=True if device.type != 'mps' else False), labels.to(device,non_blocking=True if device.type != 'mps' else False)
                outputs = moco_lp_model(inputs); loss = lp_criterion_moco(outputs, labels)
                running_val_loss += loss.item()*inputs.size(0); _, pred = torch.max(outputs.data,1)
                total_val += labels.size(0); correct_val += (pred==labels).sum().item()
                all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        epoch_val_loss=running_val_loss/total_val; epoch_val_acc=correct_val/total_val
        epoch_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        history_moco['moco_linear_probe_val_loss'].append(epoch_val_loss); history_moco['moco_linear_probe_val_acc'].append(epoch_val_acc); history_moco['moco_linear_probe_val_f1'].append(epoch_val_f1)
        print(f"MoCo LP Ep {epoch+1} - Train L: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val L: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f}")
    
    print(f"MoCo LP Finished: {(time.time()-start_time_moco_lp)/60:.2f} min")
    plt.figure(figsize=(18,5)); 
    plt.subplot(1,3,1); plt.plot(history_moco['moco_linear_probe_train_loss'],label='Tr L'); plt.plot(history_moco['moco_linear_probe_val_loss'],label='Val L'); plt.title('MoCo LP Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,2); plt.plot(history_moco['moco_linear_probe_train_acc'],label='Tr Acc'); plt.plot(history_moco['moco_linear_probe_val_acc'],label='Val Acc'); plt.title('MoCo LP Acc'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,3); plt.plot(history_moco['moco_linear_probe_val_f1'],label='Val F1'); plt.title('MoCo LP F1'); plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('moco_linear_probe_curves.png'); plt.close(); print("MoCo LP curves saved.")

    print("\nMoCo Script Finished.")

if __name__ == '__main__':
    main_moco_pipeline()