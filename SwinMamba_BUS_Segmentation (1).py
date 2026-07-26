

import os
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Swin Transformer from torchvision (available in torchvision >= 0.13)
from torchvision.models import swin_s, Swin_S_Weights

# ============================================================================
# CONFIGURATION
# ============================================================================

# ⚠️ CHANGE THIS TO YOUR DATA PATH
DATA_ROOT = r'D:/Jafar/Ultrasound/BUSI/All/'
SAVE_DIR  = r'D:/Jafar/Ultrasound/BUSI/results_swinmamba/'

IMG_SIZE     = 512
BATCH_SIZE   = 4        # Reduced for Swin (larger model than EfficientNet-B4)
EPOCHS       = 1000
BASE_LR      = 3e-5     # Lower LR suits Transformers
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42
NUM_WORKERS  = 0        # Windows safe
GRAD_CLIP    = 1.0
DS_WEIGHT    = 0.3      # Deep supervision loss weight
BOUNDARY_WEIGHT = 0.2   # Boundary loss weight
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print(f"\n{'='*80}")
print(f"  SwinMamba-FPN Breast Ultrasound Segmentation")
print(f"{'='*80}")
print(f"  Device : {DEVICE}")
print(f"  Data   : {DATA_ROOT}")
print(f"  Img    : {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")
print(f"{'='*80}\n")

# ============================================================================
# DATA AUGMENTATION  (same proven pipeline, kept compatible)
# ============================================================================

def get_training_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ElasticTransform(alpha=120, sigma=6, p=0.2),   # NEW: elastic for US
        A.GridDistortion(p=0.2),                          # NEW: grid distortion
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# DATASET  (identical to your working version)
# ============================================================================

class BUSIDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.transform:
            t     = self.transform(image=image, mask=mask)
            image = t['image']
            mask  = t['mask']

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def find_image_mask_pairs(root_dir: Path):
    print(f"{'='*80}\nDATA LOADING\n{'='*80}")
    print(f"Root: {root_dir}")
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    images_dir = masks_dir = None
    for img_f, msk_f in [('images','masks'), ('Images','Masks')]:
        ip, mp = root_dir/img_f, root_dir/msk_f
        if ip.exists() and mp.exists():
            images_dir, masks_dir = ip, mp
            print(f"✓ Found: {img_f}/ and {msk_f}/")
            break
    if images_dir is None:
        images_dir = masks_dir = root_dir
        print("Using flat structure")

    image_files = []
    for ext in ['*.png','*.jpg','*.PNG','*.JPG']:
        image_files.extend(list(images_dir.glob(ext)))
    image_files = sorted([f for f in image_files if 'mask' not in f.stem.lower()])
    print(f"Found {len(image_files)} images")

    image_paths, mask_paths = [], []
    for img_path in image_files:
        for pattern in [f"{img_path.stem}_mask*.png", f"{img_path.stem}_mask*.jpg"]:
            matches = list(masks_dir.glob(pattern))
            if matches:
                image_paths.append(str(img_path))
                mask_paths.append(str(matches[0]))
                break

    print(f"Matched {len(image_paths)}/{len(image_files)} pairs")
    return image_paths, mask_paths


def create_dataloaders(root, batch_size, val_ratio, test_ratio, seed):
    ip, mp = find_image_mask_pairs(Path(root))
    tr_i, tmp_i, tr_m, tmp_m = train_test_split(ip, mp,
        test_size=val_ratio+test_ratio, random_state=seed)
    va_i, te_i, va_m, te_m = train_test_split(tmp_i, tmp_m,
        test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)

    print(f"Train={len(tr_i)}, Val={len(va_i)}, Test={len(te_i)}")

    kw = dict(num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(BUSIDataset(tr_i, tr_m, get_training_transforms()),
                              batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(BUSIDataset(va_i, va_m, get_validation_transforms()),
                              batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(BUSIDataset(te_i, te_m, get_validation_transforms()),
                              batch_size=batch_size, shuffle=False, **kw)
    print("✓ DataLoaders ready\n")
    return train_loader, val_loader, test_loader

# ============================================================================
# BLOCK 1 — CBAM  (Channel + Spatial Attention Module)
# ============================================================================
# Applied at every FPN level. Forces the model to attend to WHAT (channel)
# and WHERE (spatial) — critical for irregular ultrasound lesion shapes.

class ChannelAttention(nn.Module):
    """Squeeze-Excitation style channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """7×7 conv spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Full CBAM = Channel Attention → Spatial Attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ============================================================================
# BLOCK 2 — LIGHTWEIGHT MAMBA-STYLE SSM BOTTLENECK
# ============================================================================
# Implements the core idea of Visual Mamba (VMamba) WITHOUT requiring the
# mamba-ssm CUDA package — uses an efficient 1D state-space scan via
# depthwise conv + gating, which approximates SSM behavior and is fully
# pip-installable on any PyTorch setup.
#
# Why this instead of plain self-attention?
#  - O(N) vs O(N²) complexity → fits 12GB GPU at 512×512
#  - Captures sequential long-range context across spatial tokens
#  - Novel for ultrasound segmentation (2024 architecture)

class SSMScan(nn.Module):
    """1-D efficient state-space scan along H and W directions."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.dim     = dim
        self.d_state = d_state

        # Input projection
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)

        # Depthwise conv for local context (mimics SSM conv step)
        self.conv1d  = nn.Conv1d(dim, dim, kernel_size=d_conv,
                                 padding=d_conv-1, groups=dim, bias=True)

        # SSM parameters (A, B, C, D)
        self.x_proj  = nn.Linear(dim, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)

        # Learnable A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(dim))

        nn.init.normal_(self.dt_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim)"""
        B, L, D = x.shape
        residual = x

        # Split into z (gate) and x
        xz   = self.in_proj(x)               # (B, L, 2D)
        x_in = xz[..., :D]                   # (B, L, D)
        z    = xz[..., D:]                   # (B, L, D)

        # Local depthwise conv (causal-like)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]   # (B, D, L)
        x_conv = F.silu(x_conv).transpose(1, 2)                 # (B, L, D)

        # Compute B, C, dt from projection
        bcd  = self.x_proj(x_conv)                              # (B, L, 2S+1)
        dt   = F.softplus(self.dt_proj(bcd[..., :1]))           # (B, L, D)
        B_p  = bcd[..., 1:self.d_state + 1]                     # (B, L, S)
        C_p  = bcd[..., self.d_state + 1:]                      # (B, L, S)

        # Discretise A
        A  = -torch.exp(self.A_log.float())                     # (D, S)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,S)
        dB = dt.unsqueeze(-1) * B_p.unsqueeze(2)                # (B,L,D,S)

        # Sequential scan (cumulative state accumulation via cumprod trick)
        # Efficient approximate scan: use cumsum instead of true recurrence
        # (exact for linear systems in log-space)
        h   = (dB * x_conv.unsqueeze(-1))                       # (B,L,D,S)
        log_dA_cum = torch.cumsum(torch.log(dA.clamp(min=1e-6)), dim=1)
        h_scan = h * torch.exp(log_dA_cum)                      # (B,L,D,S)

        # Output: sum over states with C
        y = (h_scan * C_p.unsqueeze(2)).sum(-1)                 # (B,L,D)
        y = y + self.D * x_conv

        # Gate with z
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


class MambaBottleneck(nn.Module):
    """
    2-D Visual Mamba bottleneck:
    Scans features in 4 directions (H→, ←H, W↓, ↑W) and merges results.
    This is the cross-scan strategy from VMamba (Liu et al., 2024).
    """
    def __init__(self, channels: int, d_state: int = 16, depth: int = 2):
        super().__init__()
        self.norm_in = nn.LayerNorm(channels)
        # 4-directional SSM blocks
        self.ssm_blocks = nn.ModuleList([
            SSMScan(channels, d_state) for _ in range(depth * 4)
        ])
        self.depth   = depth
        self.merge   = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.norm_out = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape

        outputs = []
        for d in range(self.depth):
            base = d * 4
            # 4 directional scans
            flat_hw = x.flatten(2).transpose(1, 2)        # (B, H*W, C)
            flat_wh = x.permute(0,1,3,2).flatten(2).transpose(1,2)  # W-first
            r_hw    = flat_hw.flip(1)                     # reversed H*W
            r_wh    = flat_wh.flip(1)                     # reversed W-first

            y_hw  = self.ssm_blocks[base+0](flat_hw).transpose(1,2).view(B,C,H,W)
            y_wh  = self.ssm_blocks[base+1](flat_wh).transpose(1,2).view(B,C,W,H).permute(0,1,3,2)
            y_rhw = self.ssm_blocks[base+2](r_hw).flip(1).transpose(1,2).view(B,C,H,W)
            y_rwh = self.ssm_blocks[base+3](r_wh).flip(1).transpose(1,2).view(B,C,W,H).permute(0,1,3,2)

            x = self.norm_out(self.merge(torch.cat([y_hw, y_wh, y_rhw, y_rwh], dim=1)))

        return x

# ============================================================================
# BLOCK 3 — SWIN TRANSFORMER ENCODER WRAPPER
# ============================================================================

class SwinEncoder(nn.Module):
    """
    Wraps torchvision Swin-S and exposes 4 hierarchical feature maps:
      Stage 0: stride 4  → C=96
      Stage 1: stride 8  → C=192
      Stage 2: stride 16 → C=384
      Stage 3: stride 32 → C=768
    """
    CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        swin    = swin_s(weights=weights)

        # Swin feature extractor has 8 sub-layers in features:
        # 0: PatchEmbed, 1: Stage0, 2: PatchMerge, 3: Stage1,
        # 4: PatchMerge, 5: Stage2, 6: PatchMerge, 7: Stage3
        feats = swin.features
        self.stage0 = nn.Sequential(feats[0], feats[1])   # → stride 4,  C=96
        self.down1  = feats[2]                             # PatchMerge
        self.stage1 = feats[3]                             # → stride 8,  C=192
        self.down2  = feats[4]
        self.stage2 = feats[5]                             # → stride 16, C=384
        self.down3  = feats[6]
        self.stage3 = feats[7]                             # → stride 32, C=768

        print(f"✓ SwinEncoder loaded (pretrained={pretrained})")

    def forward(self, x):
        # Swin outputs (B, H, W, C) — convert to (B, C, H, W) for CNN layers
        p0 = self.stage0(x).permute(0,3,1,2)              # stride 4
        p1 = self.stage1(self.down1(p0.permute(0,2,3,1))).permute(0,3,1,2)   # stride 8
        p2 = self.stage2(self.down2(p1.permute(0,2,3,1))).permute(0,3,1,2)   # stride 16
        p3 = self.stage3(self.down3(p2.permute(0,2,3,1))).permute(0,3,1,2)   # stride 32
        return p0, p1, p2, p3   # (B,96,128,128), (B,192,64,64), (B,384,32,32), (B,768,16,16)

# ============================================================================
# BLOCK 4 — FPN NECK WITH CBAM
# ============================================================================
# Top-down feature pyramid with lateral connections.
# CBAM at every level recalibrates features before decoder fusion.

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network with CBAM attention at each level.
    Unifies all 4 encoder stages to FPN_CHANNELS.
    """
    def __init__(self, in_channels: List[int], fpn_channels: int = 256):
        super().__init__()
        # Lateral 1×1 projections
        self.lat = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1, bias=False) for c in in_channels
        ])
        # Output 3×3 smoothing convs
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels
        ])
        # CBAM at every FPN level
        self.cbam = nn.ModuleList([
            CBAM(fpn_channels) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """features = [p0, p1, p2, p3] — finest to coarsest."""
        # Lateral projections
        laterals = [lat(f) for lat, f in zip(self.lat, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i+1], size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False
            )

        # Smooth + CBAM
        out = [cbam(smooth(lat))
               for cbam, smooth, lat in zip(self.cbam, self.smooth, laterals)]
        return out   # [f0,f1,f2,f3] all with fpn_channels

# ============================================================================
# BLOCK 5 — ATTENTION GATE (for decoder skip connections)
# ============================================================================
# Suppresses irrelevant activations in skip features using gating signal
# from the coarser decoder level — crucial for noisy ultrasound backgrounds.

class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        f_g  : gating signal channels (from decoder above)
        f_l  : skip connection channels (from encoder)
        f_int: intermediate channels
        """
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

# ============================================================================
# BLOCK 6 — DECODER BLOCK
# ============================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

# ============================================================================
# MAIN MODEL — SwinMamba-FPN-Net
# ============================================================================

class SwinMambaFPN(nn.Module):
    """
    Full architecture:
      Swin-S Encoder → FPN+CBAM Neck → Mamba Bottleneck
      → Attention-Gated Decoder (×4) → Boundary Branch
      → Deep Supervision (×4 auxiliary heads)
    """
    FPN_CH = 256   # unified FPN channel width
    DEC_CH = [128, 96, 64, 32]  # decoder stage output channels

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        print("Building SwinMamba-FPN model...")

        # 1. Encoder
        self.encoder = SwinEncoder(pretrained=pretrained)
        enc_ch = SwinEncoder.CHANNELS   # [96, 192, 384, 768]

        # 2. FPN neck (produces 4 feature maps, each FPN_CH wide)
        self.fpn = FPNNeck(enc_ch, self.FPN_CH)

        # 3. Mamba bottleneck on coarsest FPN feature (f3, stride 32)
        self.mamba = MambaBottleneck(self.FPN_CH, d_state=16, depth=2)

        # 4. Attention gates for each skip connection
        d = self.DEC_CH
        f = self.FPN_CH
        half = f // 2
        self.ag3 = AttentionGate(f_g=f,    f_l=f, f_int=half)
        self.ag2 = AttentionGate(f_g=d[0], f_l=f, f_int=d[0]//2)
        self.ag1 = AttentionGate(f_g=d[1], f_l=f, f_int=d[1]//2)
        self.ag0 = AttentionGate(f_g=d[2], f_l=f, f_int=d[2]//2)

        # 5. Decoder blocks
        self.dec3 = DecoderBlock(f,    f,    d[0])   # 16→32 spatial
        self.dec2 = DecoderBlock(d[0], f,    d[1])   # 32→64
        self.dec1 = DecoderBlock(d[1], f,    d[2])   # 64→128
        self.dec0 = DecoderBlock(d[2], f,    d[3])   # 128→256 (128 for 512 input)

        # 6. Final output head (upsample to full resolution)
        self.final_up = nn.Sequential(
            nn.Conv2d(d[3], d[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(d[3]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(d[3], num_classes, 1)

        # 7. Deep supervision heads (one per decoder stage)
        self.ds3 = nn.Conv2d(d[0], num_classes, 1)
        self.ds2 = nn.Conv2d(d[1], num_classes, 1)
        self.ds1 = nn.Conv2d(d[2], num_classes, 1)
        self.ds0 = nn.Conv2d(d[3], num_classes, 1)

        # 8. Boundary detection branch (explicit edge learning)
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(d[3], d[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(d[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(d[3], 1, 1)
        )

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        print(f"✓ SwinMamba-FPN ready | Parameters: {n/1e6:.1f}M")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        target_size = x.shape[2:]   # (512, 512)

        # ── Encoder ──────────────────────────────────────────────────────────
        p0, p1, p2, p3 = self.encoder(x)   # strides: 4, 8, 16, 32

        # ── FPN neck (CBAM at each level) ─────────────────────────────────────
        f0, f1, f2, f3 = self.fpn([p0, p1, p2, p3])
        # f0: 128×128, f1: 64×64, f2: 32×32, f3: 16×16  (all FPN_CH)

        # ── Mamba bottleneck on coarsest features ────────────────────────────
        f3 = self.mamba(f3)   # same spatial size, enriched long-range context

        # ── Attention-gated decoder ───────────────────────────────────────────
        # Stage 3: 16×16 → 32×32
        sk3 = self.ag3(f3, f2)
        d3  = self.dec3(f3, sk3)

        # Stage 2: 32×32 → 64×64
        sk2 = self.ag2(d3, f1)
        d2  = self.dec2(d3, sk2)

        # Stage 1: 64×64 → 128×128
        sk1 = self.ag1(d2, f0)
        d1  = self.dec1(d2, sk1)

        # Stage 0: 128×128 → 256×256 (need extra FPN feature at stride 2)
        # We reuse f0 upsampled as the skip here
        f0_up = F.interpolate(f0, scale_factor=2, mode='bilinear', align_corners=False)
        sk0   = self.ag0(d1, f0_up)
        d0    = self.dec0(d1, sk0)

        # ── Final head (upsample to full 512×512) ────────────────────────────
        d0_up = F.interpolate(self.final_up(d0), size=target_size,
                              mode='bilinear', align_corners=False)
        main_out = self.head(d0_up)        # (B, 1, 512, 512)

        # ── Boundary branch ──────────────────────────────────────────────────
        boundary_out = F.interpolate(
            self.boundary_branch(d0), size=target_size,
            mode='bilinear', align_corners=False
        )                                  # (B, 1, 512, 512)

        # ── Deep supervision outputs (only used during training) ──────────────
        if self.training:
            ds_outs = [
                F.interpolate(self.ds3(d3), size=target_size, mode='bilinear', align_corners=False),
                F.interpolate(self.ds2(d2), size=target_size, mode='bilinear', align_corners=False),
                F.interpolate(self.ds1(d1), size=target_size, mode='bilinear', align_corners=False),
                F.interpolate(self.ds0(d0), size=target_size, mode='bilinear', align_corners=False),
            ]
            return main_out, boundary_out, ds_outs

        return main_out

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred_f = pred.view(pred.size(0), -1)
        tgt_f  = target.view(target.size(0), -1)
        inter  = (pred_f * tgt_f).sum(1)
        return 1 - ((2*inter + self.smooth) / (pred_f.sum(1) + tgt_f.sum(1) + self.smooth)).mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss — generalises Dice by separately weighting FP and FN.
    alpha=0.3, beta=0.7 penalises false negatives more → better recall.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred_f = pred.view(pred.size(0), -1)
        tgt_f  = target.view(target.size(0), -1)
        TP = (pred_f * tgt_f).sum(1)
        FP = (pred_f * (1 - tgt_f)).sum(1)
        FN = ((1 - pred_f) * tgt_f).sum(1)
        T  = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - T.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce   = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt   = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        return (self.alpha * (1 - pt)**self.gamma * ce).mean()


class BoundaryLoss(nn.Module):
    """
    Laplacian-based boundary loss.
    Computes boundary maps from ground-truth and prediction using a
    Laplacian filter, then computes BCE between predicted and GT boundaries.
    Drives the model to sharpen contour predictions.
    """
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.view(1,1,3,3))

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B,1,H,W) float
        b = F.conv2d(mask, self.lap_kernel, padding=1)
        return (b.abs() > 0.1).float()

    def forward(self, pred_boundary, target):
        target_boundary = self._get_boundary(target)
        return F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)


class CombinedLoss(nn.Module):
    """
    Master loss:
      Main = 0.4 × Dice + 0.3 × Tversky + 0.3 × Focal
      + BOUNDARY_WEIGHT × Boundary Loss on boundary head
      + DS_WEIGHT × mean(Deep Supervision losses)
    """
    def __init__(self):
        super().__init__()
        self.dice     = DiceLoss()
        self.tversky  = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal    = FocalLoss(alpha=0.25, gamma=2.0)
        self.boundary = BoundaryLoss()

    def seg_loss(self, pred, target):
        return (0.4 * self.dice(pred, target)
              + 0.3 * self.tversky(pred, target)
              + 0.3 * self.focal(pred, target))

    def forward(self, outputs, target):
        target = target.float()

        if isinstance(outputs, tuple):
            # Training mode: (main, boundary, [ds1, ds2, ds3, ds4])
            main_pred, boundary_pred, ds_preds = outputs

            main_loss = self.seg_loss(main_pred, target)
            bnd_loss  = self.boundary(boundary_pred, target)
            ds_loss   = sum(self.seg_loss(d, target) for d in ds_preds) / len(ds_preds)

            total = main_loss + BOUNDARY_WEIGHT * bnd_loss + DS_WEIGHT * ds_loss
            return total, main_loss.item(), bnd_loss.item(), ds_loss.item()

        else:
            # Inference mode: single tensor
            loss = self.seg_loss(outputs, target)
            return loss, loss.item(), 0.0, 0.0

# ============================================================================
# METRICS
# ============================================================================

@torch.no_grad()
def calculate_metrics(pred, target, threshold=0.5):
    # Use main output only if tuple
    if isinstance(pred, tuple):
        pred = pred[0]

    pb = (torch.sigmoid(pred) > threshold).float()
    tb = target.float()
    pf, tf = pb.view(-1), tb.view(-1)

    tp = (pf * tf).sum()
    fp = (pf * (1 - tf)).sum()
    fn = ((1 - pf) * tf).sum()
    tn = ((1 - pf) * (1 - tf)).sum()
    eps = 1e-7

    return {
        'dice':        ((2*tp + eps) / (2*tp + fp + fn + eps)).item(),
        'iou':         ((tp + eps) / (tp + fp + fn + eps)).item(),
        'precision':   ((tp + eps) / (tp + fp + eps)).item(),
        'recall':      ((tp + eps) / (tp + fn + eps)).item(),
        'specificity': ((tn + eps) / (tn + fp + eps)).item(),
        'accuracy':    ((tp + tn + eps) / (tp + tn + fp + fn + eps)).item(),
        'jaccard':     ((tp + eps) / (tp + fp + fn + eps)).item(),
    }

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    total_main = total_bnd = total_ds = 0.0

    pbar = tqdm(loader, desc='Train')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(images)
            loss, ml, bl, dl = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_main += ml; total_bnd += bl; total_ds += dl

        pbar.set_postfix(loss=f'{loss.item():.4f}', main=f'{ml:.4f}',
                         bnd=f'{bl:.4f}', ds=f'{dl:.4f}')

    N = len(loader)
    return total_loss/N, total_main/N, total_bnd/N, total_ds/N


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = []

    pbar = tqdm(loader, desc='Val')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        loss, _, _, _ = criterion(outputs, masks)
        total_loss += loss.item()

        metrics = calculate_metrics(outputs, masks)
        all_metrics.append(metrics)
        pbar.set_postfix(loss=f'{loss.item():.4f}', dice=f'{metrics["dice"]:.4f}')

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return total_loss / len(loader), avg

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def save_training_curves(train_losses, val_losses, metrics_history, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SwinMamba-FPN Training Curves', fontsize=16, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0,0].plot(epochs, train_losses, label='Train', color='royalblue')
    axes[0,0].plot(epochs, val_losses,   label='Val',   color='tomato')
    axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(True)

    for i, (metric, title) in enumerate([
        ('dice','Dice'), ('iou','IoU/Jaccard'),
        ('precision','Precision'), ('recall','Recall'), ('specificity','Specificity')
    ]):
        r, c = divmod(i+1, 3)
        axes[r,c].plot(epochs, [m[metric] for m in metrics_history],
                       color='seagreen')
        axes[r,c].set_title(f'Val {title}'); axes[r,c].grid(True)
        axes[r,c].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved training curves")


def save_predictions(model, loader, device, save_dir, num_samples=12):
    model.eval()
    imgs_all, masks_all, preds_all = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = torch.sigmoid(model(images)).cpu()
            imgs_all.append(images.cpu())
            masks_all.append(masks)
            preds_all.append(preds)
            if sum(len(x) for x in imgs_all) >= num_samples:
                break

    imgs_all  = torch.cat(imgs_all)[:num_samples]
    masks_all = torch.cat(masks_all)[:num_samples]
    preds_all = torch.cat(preds_all)[:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    for i in range(num_samples):
        img = imgs_all[i].numpy().transpose(1,2,0)
        img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
        img = np.clip(img, 0, 1)
        msk = masks_all[i, 0].numpy()
        prd = preds_all[i, 0].numpy()

        axes[i,0].imshow(img);                        axes[i,0].set_title('Input');       axes[i,0].axis('off')
        axes[i,1].imshow(msk, cmap='gray');           axes[i,1].set_title('GT Mask');     axes[i,1].axis('off')
        axes[i,2].imshow(prd, cmap='gray');           axes[i,2].set_title('Prediction');  axes[i,2].axis('off')
        axes[i,3].imshow(img); axes[i,3].imshow(prd, cmap='Reds', alpha=0.5)
        axes[i,3].set_title('Overlay'); axes[i,3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved prediction visualizations")


def save_metrics_table(metrics, save_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    data = [
        ['Dice Score',           f"{metrics['dice']:.4f}"],
        ['IoU / Jaccard',        f"{metrics['iou']:.4f}"],
        ['Precision',            f"{metrics['precision']:.4f}"],
        ['Recall (Sensitivity)', f"{metrics['recall']:.4f}"],
        ['Specificity',          f"{metrics['specificity']:.4f}"],
        ['Accuracy',             f"{metrics['accuracy']:.4f}"],
    ]
    tbl = ax.table(cellText=data, colLabels=['Metric','Value'],
                   cellLoc='left', loc='center', colWidths=[0.7, 0.3])
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.5)
    for j in range(2):
        tbl[(0,j)].set_facecolor('#1565C0')
        tbl[(0,j)].set_text_props(weight='bold', color='white')
    for i in range(1, len(data)+1):
        for j in range(2):
            if i % 2 == 0: tbl[(i,j)].set_facecolor('#E3F2FD')

    plt.title('SwinMamba-FPN — Test Set Results', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'metrics_table.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved metrics table")

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        DATA_ROOT, BATCH_SIZE, VAL_RATIO, TEST_RATIO, SEED
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nMODEL SETUP\n{'='*80}")
    model     = SwinMambaFPN(num_classes=1, pretrained=True).to(DEVICE)
    criterion = CombinedLoss().to(DEVICE)

    # Differential LR: lower LR for pretrained Swin encoder
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n,p in model.named_parameters() if 'encoder' not in n]
    optimizer  = torch.optim.AdamW([
        {'params': enc_params, 'lr': BASE_LR * 0.1},
        {'params': dec_params, 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-7
    )
    scaler = GradScaler()
    print("✓ Optimizer, scheduler, scaler ready")

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nTRAINING\n{'='*80}\n")
    best_dice   = 0.0
    train_losses, val_losses, metrics_history = [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 80)

        tr_loss, tr_main, tr_bnd, tr_ds = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE)
        va_loss, va_metrics = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        metrics_history.append(va_metrics)

        print(f"  Train  → Loss: {tr_loss:.4f}  [main:{tr_main:.4f}  bnd:{tr_bnd:.4f}  ds:{tr_ds:.4f}]")
        print(f"  Val    → Loss: {va_loss:.4f}")
        for k, v in va_metrics.items():
            print(f"           {k:<14s}: {v:.4f}")

        # Save best model
        if va_metrics['dice'] > best_dice:
            best_dice = va_metrics['dice']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"  ✓ New best Dice: {best_dice:.4f} — model saved")

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'epoch_{epoch+1}.pth'))

    # ── Test ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nTESTING\n{'='*80}")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'),
                                     map_location=DEVICE))
    _, test_metrics = validate(model, test_loader, criterion, DEVICE)

    print(f"\n  FINAL TEST RESULTS:")
    print(f"  {'='*40}")
    for k, v in test_metrics.items():
        print(f"  {k:<14s}: {v:.4f}")
    print(f"  Best Val Dice : {best_dice:.4f}")
    print(f"  {'='*40}")

    # Save results
    with open(os.path.join(SAVE_DIR, 'results.txt'), 'w') as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"best_val_dice: {best_dice:.4f}\n")

    # ── Visualizations ───────────────────────────────────────────────────────
    print(f"\n{'='*80}\nGENERATING VISUALIZATIONS\n{'='*80}")
    save_training_curves(train_losses, val_losses, metrics_history, SAVE_DIR)
    save_predictions(model, test_loader, DEVICE, SAVE_DIR, num_samples=12)
    save_metrics_table(test_metrics, SAVE_DIR)

    print(f"\n{'='*80}")
    print(f"  COMPLETE! Best Val Dice: {best_dice:.4f}")
    print(f"  All results saved to: {SAVE_DIR}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()