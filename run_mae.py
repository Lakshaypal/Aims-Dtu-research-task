# run_mae.py

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
import timm # Ensure timm is installed (pip install timm)
from timm.optim import optim_factory # For MAE optimizer if using specific decay settings
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

# Common Image and LP settings (can be overridden by MAE specific if needed)
IMG_SIZE_GENERAL = 224
LINEAR_PROBE_BATCH_SIZE_GENERAL = 64
LINEAR_PROBE_EPOCHS_GENERAL = 5
LR_LINEAR_PROBE_GENERAL = 1e-3
WEIGHT_DECAY_GENERAL = 1e-4
NUM_WORKERS = 2 # Adjust based on your system

# MAE Specific Configurations
MAE_IMG_SIZE = IMG_SIZE_GENERAL # Usually same as encoder input size
# CRITICAL: Increase MAE_PRETRAIN_EPOCHS for good results (e.g., 200, 400, 800+)
MAE_PRETRAIN_EPOCHS = 5  # Placeholder, increase for actual training
MAE_PRETRAIN_BATCH_SIZE = 32 # Adjust based on GPU memory
MAE_LR_PRETRAIN = 1.5e-4    # Typical MAE learning rate
MAE_WEIGHT_DECAY_PRETRAIN = 0.05 # Typical MAE weight decay
MAE_MASKING_RATIO = 0.75
MAE_ENCODER_MODEL_NAME = 'vit_base_patch16_224' # timm model name for ViT encoder

MAE_DECODER_EMBED_DIM = 512
MAE_DECODER_DEPTH = 8
MAE_DECODER_NUM_HEADS = 16
MAE_MLP_RATIO_DECODER = 4.0
MAE_NORM_PIX_LOSS = True # Normalize pixel values for loss calculation

PRETRAINED_MAE_MODEL_PATH = 'mae_vit_full_model_pretrained.pth' # Path to save/load the full MAE model

# History dictionary for MAE
history_mae = {
    'mae_pretrain_loss': [],
    'mae_linear_probe_train_loss': [], 'mae_linear_probe_val_loss': [],
    'mae_linear_probe_train_acc': [], 'mae_linear_probe_val_acc': [],
    'mae_linear_probe_val_f1': [],
}


# --- MAE Model Components ---
def patchify(imgs, patch_size):
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, f"Image dimensions ({imgs.shape[2]}x{imgs.shape[3]}) not divisible by patch size ({p})"
    h = w = imgs.shape[2] // p
    num_patches = h * w
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], num_patches, p**2 * imgs.shape[1]))
    return x

def unpatchify(x, patch_size, channels=3):
    p = patch_size
    h = w = int(x.shape[1]**0.5)
    assert h * w == x.shape[1], "Cannot infer H, W from number of patches for unpatchify"
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * p, w * p))
    return imgs

class MAETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm): # qkv_bias typically True
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out # Dropping DropPath for simplicity
        x = x + self.mlp(self.norm2(x)) # Dropping DropPath
        return x

class MAE_ViT(nn.Module):
    def __init__(self, encoder_model_name=MAE_ENCODER_MODEL_NAME, img_size=MAE_IMG_SIZE,
                 mask_ratio=MAE_MASKING_RATIO,
                 decoder_embed_dim=MAE_DECODER_EMBED_DIM, decoder_depth=MAE_DECODER_DEPTH,
                 decoder_num_heads=MAE_DECODER_NUM_HEADS, mlp_ratio_decoder=MAE_MLP_RATIO_DECODER,
                 norm_pix_loss=MAE_NORM_PIX_LOSS):
        super().__init__()
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.encoder = timm.create_model(encoder_model_name, pretrained=False, num_classes=0, global_pool='')
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.num_patches = self.encoder.patch_embed.num_patches
        self.encoder_embed_dim = self.encoder.embed_dim # e.g., 768 for ViT-Base
        self.decoder_embed_dim = decoder_embed_dim # Store decoder_embed_dim as an instance attribute
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embedding for the decoder (fixed, not learned, for all patches)
        # It includes position for CLS token if encoder uses one, but CLS token is not processed by decoder blocks.
        # So, effective num_patches for decoder positional embedding should be self.num_patches.
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            MAETransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio_decoder)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * 3, bias=True) # 3 for RGB

        self.initialize_weights()

    def initialize_weights(self):
        # Sinusoidal positional encoding for decoder_pos_embed
        decoder_pos_embed_val = self.get_sinusoid_encoding_table(self.num_patches, self.decoder_embed_dim)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed_val.float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_custom) # Apply custom init for linear/layernorm

    def _init_weights_custom(self, m): # Renamed to avoid conflict if timm model also has _init_weights
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.as_tensor(sinusoid_table, dtype=torch.float32)


    def random_masking(self, x_patches, mask_ratio): # x_patches is (N, num_patches, D_encoder)
        N, L, D = x_patches.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x_patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x_patches, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([N, L], device=x_patches.device) # 1 for masked, 0 for visible
        mask[:, :len_keep] = 0 
        mask = torch.gather(mask, dim=1, index=ids_restore) # binary mask in original order
        return x_visible, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio_val): # imgs: (N, C, H, W)
        x_patches = self.encoder.patch_embed(imgs) # (N, num_patches, D_encoder)
        
        # Add positional encoding to patches
        # Timm's ViT adds pos_embed after adding CLS token. We operate on patches first.
        # self.encoder.pos_embed is (1, num_patches+1, D_encoder) if CLS token exists
        if hasattr(self.encoder, 'pos_embed') and self.encoder.pos_embed is not None:
             # Assuming pos_embed includes CLS token pos, so take [:, 1:, :] for patches
            x_patches = x_patches + self.encoder.pos_embed[:, 1:x_patches.size(1)+1, :]

        x_visible, mask, ids_restore = self.random_masking(x_patches, mask_ratio_val)
        
        # Prepend CLS token to visible patches before passing to encoder blocks
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            cls_token = self.encoder.cls_token.expand(x_visible.shape[0], -1, -1)
            # Add CLS pos_embed to CLS token
            cls_token = cls_token + self.encoder.pos_embed[:, :1, :] 
            x_visible = torch.cat((cls_token, x_visible), dim=1)
        
        # Pass through encoder blocks
        for blk in self.encoder.blocks:
            x_visible = blk(x_visible)
        if self.encoder.norm is not None: # Final norm layer of encoder
            x_visible = self.encoder.norm(x_visible)
        
        # x_visible now contains features for CLS token (if used) and visible patches
        return x_visible, mask, ids_restore


    def forward_decoder(self, x_visible_features, ids_restore): # x_visible_features from encoder (N, 1(CLS)+len_keep, D_encoder)
        # Separate CLS token if it was part of encoder output
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            x_patch_features = x_visible_features[:, 1:, :] # (N, len_keep, D_encoder)
        else:
            x_patch_features = x_visible_features # (N, len_keep, D_encoder)

        x_patch_features = self.decoder_embed(x_patch_features) # (N, len_keep, D_decoder)
        
        # Create full sequence with mask tokens
        num_masked = ids_restore.shape[1] - x_patch_features.shape[1]
        mask_tokens = self.mask_token.repeat(x_patch_features.shape[0], num_masked, 1) # (N, num_masked, D_decoder)
        
        # Concatenate visible patch features and mask tokens
        x_full_seq_shuffled = torch.cat([x_patch_features, mask_tokens], dim=1) # (N, num_patches, D_decoder), shuffled order
        
        # Unshuffle to original patch order
        x_full_seq = torch.gather(x_full_seq_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim))
        
        # Add decoder positional embeddings
        x_full_seq = x_full_seq + self.decoder_pos_embed # (N, num_patches, D_decoder)
        
        for blk in self.decoder_blocks:
            x_full_seq = blk(x_full_seq)
        x_full_seq = self.decoder_norm(x_full_seq)
        pred_pixel_values = self.decoder_pred(x_full_seq) # (N, num_patches, patch_size^2 * 3)
        return pred_pixel_values

    def forward_loss(self, imgs, pred_pixel_values, mask): # mask is (N, num_patches), 1 indicates masked
        target_patches = patchify(imgs, self.patch_size) # (N, num_patches, p*p*C)
        
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var.add(1.e-6)).sqrt()

        loss_per_patch = (pred_pixel_values - target_patches) ** 2
        loss_per_patch = loss_per_patch.mean(dim=-1) # MSE per patch (N, num_patches)
        
        # Calculate loss only on masked patches
        masked_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1) # Avoid division by zero if no patches are masked
        return masked_loss

    def forward(self, imgs): # imgs: (N, C, H, W)
        # mask_ratio here is self.mask_ratio defined in init
        latent_visible_features, mask_binary, ids_restore_order = self.forward_encoder(imgs, self.mask_ratio)
        predicted_pixel_values = self.forward_decoder(latent_visible_features, ids_restore_order)
        reconstruction_loss = self.forward_loss(imgs, predicted_pixel_values, mask_binary)
        return reconstruction_loss, predicted_pixel_values, mask_binary

    # For linear probing: extract features from the encoder
    def extract_encoder_features(self, imgs):
        # This needs to replicate the encoder pass without masking and return the desired feature representation
        # (typically the CLS token output or global average pooled patch features)
        
        x_patches = self.encoder.patch_embed(imgs) # (N, num_patches, D_encoder)
        
        if hasattr(self.encoder, 'pos_embed') and self.encoder.pos_embed is not None:
            x_patches = x_patches + self.encoder.pos_embed[:, 1:x_patches.size(1)+1, :] # Add patch pos_embed

        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            cls_token = self.encoder.cls_token.expand(x_patches.shape[0], -1, -1)
            cls_token = cls_token + self.encoder.pos_embed[:, :1, :] # Add CLS pos_embed
            x_full_encoder_input = torch.cat((cls_token, x_patches), dim=1)
        else:
            x_full_encoder_input = x_patches
            
        for blk in self.encoder.blocks:
            x_full_encoder_input = blk(x_full_encoder_input)
        
        if self.encoder.norm is not None:
            x_full_encoder_input = self.encoder.norm(x_full_encoder_input)
            
        # Feature extraction strategy
        if hasattr(self.encoder, 'fc_norm') and self.encoder.fc_norm is not None: # e.g. for DeiT type models from timm
             return self.encoder.fc_norm(x_full_encoder_input[:, 0]) # CLS token after fc_norm
        elif hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            return x_full_encoder_input[:, 0]  # Return CLS token feature
        else: # No CLS token, use global average pooling of patch features
            return x_full_encoder_input.mean(dim=1)


class MAELinearClassifier(nn.Module):
    def __init__(self, mae_full_model_state_dict_path, encoder_model_name=MAE_ENCODER_MODEL_NAME, 
                 num_classes_local=100, img_size_local=MAE_IMG_SIZE):
        super().__init__()
        # Reconstruct only the encoder part for linear probing
        self.encoder_for_lp = timm.create_model(encoder_model_name, pretrained=False, num_classes=0, global_pool='')
        
        try:
            full_mae_model_state_dict = torch.load(mae_full_model_state_dict_path, map_location=device)
            encoder_state_dict = {}
            # Filter for 'encoder.' prefixed keys from the full MAE_ViT model state_dict
            for k, v in full_mae_model_state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '', 1)] = v
            
            if not encoder_state_dict:
                 # If no 'encoder.' prefix, assume it's already an encoder-only state_dict
                 # This might happen if PRETRAINED_MAE_MODEL_PATH saved only mae_model.encoder.state_dict()
                 print(f"Warning: No 'encoder.' prefixed keys found in {mae_full_model_state_dict_path}. Attempting to load as is.")
                 self.encoder_for_lp.load_state_dict(full_mae_model_state_dict)
            else:
                self.encoder_for_lp.load_state_dict(encoder_state_dict)

            print(f"Successfully loaded pretrained MAE encoder weights into timm model for LP from {mae_full_model_state_dict_path}")

        except Exception as e:
            print(f"ERROR loading MAE encoder state_dict for LP: {e}. Using a randomly initialized encoder.")

        for param in self.encoder_for_lp.parameters():
            param.requires_grad = False
        self.encoder_for_lp.eval()

        # Determine feature dimension from encoder
        encoder_embed_dim = self.encoder_for_lp.embed_dim # e.g. 768 for ViT-Base
        self.classifier_head = nn.Linear(encoder_embed_dim, num_classes_local)

    def forward(self, x_imgs): # x_imgs is (N, C, H, W)
        with torch.no_grad():
            # Mimic the feature extraction logic of MAE_ViT.extract_encoder_features
            x_patches = self.encoder_for_lp.patch_embed(x_imgs)
            if hasattr(self.encoder_for_lp, 'pos_embed') and self.encoder_for_lp.pos_embed is not None:
                 x_patches = x_patches + self.encoder_for_lp.pos_embed[:, 1:x_patches.size(1)+1, :]

            if hasattr(self.encoder_for_lp, 'cls_token') and self.encoder_for_lp.cls_token is not None:
                cls_token = self.encoder_for_lp.cls_token.expand(x_patches.shape[0], -1, -1)
                cls_token = cls_token + self.encoder_for_lp.pos_embed[:, :1, :]
                final_encoder_input = torch.cat((cls_token, x_patches), dim=1)
            else:
                final_encoder_input = x_patches
            
            for blk in self.encoder_for_lp.blocks:
                final_encoder_input = blk(final_encoder_input)
            
            if self.encoder_for_lp.norm is not None:
                final_encoder_input = self.encoder_for_lp.norm(final_encoder_input)

            if hasattr(self.encoder_for_lp, 'fc_norm') and self.encoder_for_lp.fc_norm is not None:
                 features = self.encoder_for_lp.fc_norm(final_encoder_input[:, 0])
            elif hasattr(self.encoder_for_lp, 'cls_token') and self.encoder_for_lp.cls_token is not None:
                features = final_encoder_input[:, 0] # CLS token
            else:
                features = final_encoder_input.mean(dim=1) # GAP
                
        logits = self.classifier_head(features)
        return logits

# ==============================================================================
# MAIN MAE PIPELINE FUNCTION
# ==============================================================================
def main_mae_pipeline():
    global device, LABELS_JSON_PATH, train_sub_folders, VAL_DIR_ACTUAL, NUM_CLASSES, class_to_idx, idx_to_class_name, sorted_class_ids, history_mae, PRETRAINED_MAE_MODEL_PATH

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
        # ... (Label loading logic - same as in run_simclr.py, ensure it sets NUM_CLASSES) ...
        if not isinstance(labels_map_raw, dict): raise ValueError("Labels.json content is not a dictionary.")
        if len(labels_map_raw) != 100: print(f"Warning: Expected 100 classes, but found {len(labels_map_raw)} in Labels.json.")
        
        sorted_class_ids_from_json = sorted(labels_map_raw.keys())
        class_to_idx.update({cid: idx for idx, cid in enumerate(sorted_class_ids_from_json)})
        idx_to_class_name.update({idx: labels_map_raw[cid].split(',')[0].strip() for idx, cid in enumerate(sorted_class_ids_from_json)})
        NUM_CLASSES = len(sorted_class_ids_from_json)
        sorted_class_ids.extend(sorted_class_ids_from_json)
        
        print(f"Successfully loaded Labels.json for MAE. Number of classes: {NUM_CLASSES}")
        if NUM_CLASSES == 0: raise ValueError("NUM_CLASSES is 0 for MAE. Check Labels.json.")
    except FileNotFoundError: print(f"CRITICAL ERROR: {LABELS_JSON_PATH} not found. Exiting."); exit()
    except Exception as e: print(f"CRITICAL ERROR loading Labels.json for MAE: {e}. Exiting."); exit()


    SKIP_MAE_PRETRAINING = False # Set to True to load existing model and skip pretraining

    if not SKIP_MAE_PRETRAINING:
        print("\n--- Starting MAE Pretraining ---")
        mae_pretrain_transform = T.Compose([
            T.RandomResizedCrop(MAE_IMG_SIZE, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mae_pretrain_datasets_list = []
        for train_part_path in train_sub_folders:
            try:
                dataset_part = ImageFolder(train_part_path, transform=mae_pretrain_transform)
                if len(dataset_part) > 0: mae_pretrain_datasets_list.append(dataset_part)
            except Exception as e: print(f"Error loading MAE pretrain data from {train_part_path}: {e}")
        
        if not mae_pretrain_datasets_list: print("CRITICAL: No MAE pretrain data loaded. Exiting."); exit()
        mae_pretrain_full_dataset = ConcatDataset(mae_pretrain_datasets_list)
        mae_pretrain_loader = DataLoader(mae_pretrain_full_dataset, batch_size=MAE_PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False, drop_last=True)
        print(f"MAE pretrain dataset size: {len(mae_pretrain_full_dataset)} images.")

        mae_model = MAE_ViT( # Uses global MAE_ config variables
            encoder_model_name=MAE_ENCODER_MODEL_NAME, img_size=MAE_IMG_SIZE, mask_ratio=MAE_MASKING_RATIO,
            decoder_embed_dim=MAE_DECODER_EMBED_DIM, decoder_depth=MAE_DECODER_DEPTH,
            decoder_num_heads=MAE_DECODER_NUM_HEADS, mlp_ratio_decoder=MAE_MLP_RATIO_DECODER,
            norm_pix_loss=MAE_NORM_PIX_LOSS
        ).to(device)
        
# Optimizer for MAE (AdamW with specific weight decay settings often used)
        # Using timm's create_optimizer_v2, which handles weight decay grouping correctly
        mae_optimizer = timm.optim.create_optimizer_v2(
            mae_model,
            opt='adamw',
            lr=MAE_LR_PRETRAIN,
            weight_decay=MAE_WEIGHT_DECAY_PRETRAIN,
            betas=(0.9, 0.95)
        )
        print(f"Optimizer created: {type(mae_optimizer)}")

        # Optional: LR Scheduler (e.g., Cosine warmup)
        # mae_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(mae_optimizer, T_0=10, T_mult=2) # Example

        start_time_mae_pretrain = time.time()
        print(f"Starting MAE pretraining for {MAE_PRETRAIN_EPOCHS} epochs...")
        for epoch in range(MAE_PRETRAIN_EPOCHS):
            mae_model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(mae_pretrain_loader, desc=f"MAE Pretrain Epoch {epoch+1}/{MAE_PRETRAIN_EPOCHS}")
            for (images, _) in progress_bar: # Labels from ImageFolder are ignored
                images = images.to(device, non_blocking=True if device.type != 'mps' else False)
                mae_optimizer.zero_grad()
                loss, _, _ = mae_model(images)
                loss.backward()
                mae_optimizer.step()
                # if mae_scheduler: mae_scheduler.step(epoch + i / len(mae_pretrain_loader)) # If per-iteration scheduler

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_epoch_loss = epoch_loss / len(mae_pretrain_loader)
            history_mae['mae_pretrain_loss'].append(avg_epoch_loss)
            print(f"MAE Pretrain Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
            # if mae_scheduler: mae_scheduler.step() # If per-epoch scheduler

        print(f"MAE Pretraining Finished. Total time: {(time.time()-start_time_mae_pretrain)/60:.2f} minutes.")
        torch.save(mae_model.state_dict(), PRETRAINED_MAE_MODEL_PATH) # Save the full MAE model state
        print(f"Full MAE model state saved to {PRETRAINED_MAE_MODEL_PATH}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_mae['mae_pretrain_loss'], label='MAE Pretraining Loss')
        plt.title('MAE Pretraining Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig('mae_pretrain_loss_curve.png'); print("MAE pretraining loss curve saved to mae_pretrain_loss_curve.png"); plt.close()
    else:
        print(f"\n--- SKIPPING MAE Pretraining ---")
        if not os.path.exists(PRETRAINED_MAE_MODEL_PATH):
            print(f"CRITICAL ERROR: Pretrained MAE model {PRETRAINED_MAE_MODEL_PATH} not found. Exiting."); exit()
        print(f"Using existing MAE model from: {PRETRAINED_MAE_MODEL_PATH}")

    # ==============================================================================
    # SECTION 2: LINEAR PROBING FOR MAE ENCODER
    # ==============================================================================
    print("\n--- Starting Linear Probing for MAE Pretrained Encoder ---")
    lp_train_transform_mae = T.Compose([
        T.RandomResizedCrop(MAE_IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC), 
        T.RandomHorizontalFlip(), T.ToTensor(), 
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    lp_val_transform_mae = T.Compose([
        T.Resize(int(MAE_IMG_SIZE / 0.875), interpolation=T.InterpolationMode.BICUBIC), # Standard MAE/ViT eval resize
        T.CenterCrop(MAE_IMG_SIZE), T.ToTensor(), 
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    lp_train_datasets_list_mae = []
    for train_part_path in train_sub_folders:
        try:
            dataset_part = ImageFolder(train_part_path, transform=lp_train_transform_mae)
            if len(dataset_part) > 0: lp_train_datasets_list_mae.append(dataset_part)
        except Exception as e: print(f"Error loading LP train data for MAE from {train_part_path}: {e}")
    if not lp_train_datasets_list_mae: print("CRITICAL: No LP train data for MAE. Exiting."); exit()
    
    lp_train_full_dataset_mae = ConcatDataset(lp_train_datasets_list_mae)
    lp_train_loader_mae = DataLoader(lp_train_full_dataset_mae, batch_size=LINEAR_PROBE_BATCH_SIZE_GENERAL, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"MAE LP train dataset size: {len(lp_train_full_dataset_mae)} images.")
    
    try:
        lp_val_dataset_mae = ImageFolder(VAL_DIR_ACTUAL, transform=lp_val_transform_mae)
        if len(lp_val_dataset_mae) == 0: raise ValueError("MAE LP Validation dataset is empty.")
    except Exception as e: print(f"CRITICAL: Could not load MAE LP val data from {VAL_DIR_ACTUAL}: {e}. Exiting."); exit()
    lp_val_loader_mae = DataLoader(lp_val_dataset_mae, batch_size=LINEAR_PROBE_BATCH_SIZE_GENERAL, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type != 'mps' else False)
    print(f"MAE LP val dataset size: {len(lp_val_dataset_mae)} images.")

    mae_lp_model = MAELinearClassifier(
        mae_full_model_state_dict_path=PRETRAINED_MAE_MODEL_PATH,
        encoder_model_name=MAE_ENCODER_MODEL_NAME,
        num_classes_local=NUM_CLASSES,
        img_size_local=MAE_IMG_SIZE
    ).to(device)
    
    lp_criterion_mae = nn.CrossEntropyLoss()
    lp_optimizer_mae = optim.AdamW(mae_lp_model.classifier_head.parameters(), lr=LR_LINEAR_PROBE_GENERAL, weight_decay=WEIGHT_DECAY_GENERAL)
    
    print(f"\n--- Starting MAE Linear Probe Training for {LINEAR_PROBE_EPOCHS_GENERAL} epochs ---")
    start_time_mae_lp = time.time()
    best_val_f1_mae = 0.0

    for epoch in range(LINEAR_PROBE_EPOCHS_GENERAL):
        mae_lp_model.train() # Only classifier head is trainable
        # ... (Training loop for MAE LP - similar to SimCLR LP, using mae_lp_model, history_mae, etc.) ...
        running_loss_train = 0.0; correct_train = 0; total_train = 0
        train_bar = tqdm(lp_train_loader_mae, desc=f"MAE LP Train Epoch {epoch+1}/{LINEAR_PROBE_EPOCHS_GENERAL}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device, non_blocking=True if device.type != 'mps' else False), labels.to(device, non_blocking=True if device.type != 'mps' else False)
            lp_optimizer_mae.zero_grad(); outputs = mae_lp_model(inputs); loss = lp_criterion_mae(outputs, labels)
            loss.backward(); lp_optimizer_mae.step()
            running_loss_train += loss.item()*inputs.size(0); _, predicted = torch.max(outputs.data,1)
            total_train += labels.size(0); correct_train += (predicted==labels).sum().item()
            train_bar.set_postfix({'loss': loss.item()})
        epoch_train_loss=running_loss_train/total_train; epoch_train_acc=correct_train/total_train
        history_mae['mae_linear_probe_train_loss'].append(epoch_train_loss); history_mae['mae_linear_probe_train_acc'].append(epoch_train_acc)
        
        mae_lp_model.eval()
        # ... (Validation loop for MAE LP - similar to SimCLR LP) ...
        running_val_loss=0.0; correct_val=0; total_val=0; all_val_preds_mae=[]; all_val_labels_mae=[]
        val_bar = tqdm(lp_val_loader_mae, desc=f"MAE LP Val Epoch {epoch+1}/{LINEAR_PROBE_EPOCHS_GENERAL}")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device, non_blocking=True if device.type != 'mps' else False), labels.to(device, non_blocking=True if device.type != 'mps' else False)
                outputs = mae_lp_model(inputs); loss = lp_criterion_mae(outputs, labels)
                running_val_loss += loss.item()*inputs.size(0); _, predicted = torch.max(outputs.data,1)
                total_val += labels.size(0); correct_val += (predicted==labels).sum().item()
                all_val_preds_mae.extend(predicted.cpu().numpy()); all_val_labels_mae.extend(labels.cpu().numpy())
                val_bar.set_postfix({'loss': loss.item()})
        epoch_val_loss=running_val_loss/total_val; epoch_val_acc=correct_val/total_val
        epoch_val_f1_mae = f1_score(all_val_labels_mae, all_val_preds_mae, average='macro', zero_division=0)
        history_mae['mae_linear_probe_val_loss'].append(epoch_val_loss); history_mae['mae_linear_probe_val_acc'].append(epoch_val_acc); history_mae['mae_linear_probe_val_f1'].append(epoch_val_f1_mae)
        print(f"MAE LP Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1_mae:.4f}")
        if epoch_val_f1_mae > best_val_f1_mae: best_val_f1_mae = epoch_val_f1_mae # Optional: save best

    print(f"MAE Linear Probing Finished. Total time: {(time.time()-start_time_mae_lp)/60:.2f} minutes.")
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1,3,1); plt.plot(history_mae['mae_linear_probe_train_loss'],label='Train Loss'); plt.plot(history_mae['mae_linear_probe_val_loss'],label='Val Loss'); plt.title('MAE LP Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,2); plt.plot(history_mae['mae_linear_probe_train_acc'],label='Train Acc'); plt.plot(history_mae['mae_linear_probe_val_acc'],label='Val Acc'); plt.title('MAE LP Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,3); plt.plot(history_mae['mae_linear_probe_val_f1'],label='Val F1'); plt.title('MAE LP F1 Score'); plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('mae_linear_probe_curves.png'); print("MAE linear probing curves saved to mae_linear_probe_curves.png"); plt.close()

    print("\nMAE Script Finished.")

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    main_mae_pipeline()