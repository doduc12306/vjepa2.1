#!/usr/bin/env python3
"""
Multi-View Sign Language Recognition với V-JEPA 2.1 backbone.
3-view fusion (kl, kf, kr) + Attentive Probe trên MM-WLAuslan dataset.

Chạy: python train_multiview_vsl.py
"""

import csv
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# CẤU HÌNH
# ============================================================================

class Config:
    # ----- Dữ liệu MM-WLAuslan -----
    data_root = "/mnt/sda1/VSLR_Storage/MM-WLAuslan"
    video_dir = f"{data_root}/videos"
    label_dir = f"{data_root}/labels_clean_200_full"
    train_label_file = f"{label_dir}/train_labels_clean.csv"
    val_label_file = f"{label_dir}/val_labels_clean.csv"
    test_label_file = f"{label_dir}/test_labels_clean.csv"

    # ----- Backbone V-JEPA 2.1 -----
    # "vit_gigantic_xformers" = ViT-G (2B, embed=1664)
    # "vit_giant_xformers"    = ViT-g (1B, embed=1408)  ← khuyến nghị cho 24GB
    # "vit_large"             = ViT-L (300M, embed=1024)
    backbone_arch = "vit_giant_xformers"
    backbone_ckpt_url = (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
    )
    backbone_ckpt_key = "target_encoder"
    backbone_ckpt_local = None  # Đặt path local nếu đã download sẵn

    img_size = 384
    patch_size = 16
    tubelet_size = 2
    num_frames = 16
    embed_dim = 1408  # Khớp với backbone_arch

    # ----- Probe -----
    probe_depth = 4
    probe_heads = 8
    num_classes = 200   # 200-class subset của MM-WLAuslan
    dropout = 0.1

    # ----- Huấn luyện -----
    batch_size = 1       # Nhỏ vì 3 views × ViT-g rất tốn VRAM
    num_epochs = 30
    lr = 3e-4
    weight_decay = 0.01
    warmup_epochs = 2
    num_workers = 4
    use_amp = True       # Mixed precision

    # ----- Checkpoint -----
    save_dir = "./checkpoints_multiview"
    save_every = 5       # Lưu mỗi N epoch


# ============================================================================
# TIỆN ÍCH: Load label từ CSV
# ============================================================================

def load_labels(label_file):
    """
    Load ánh xạ sample_id (str) → gloss_id (int) từ CSV.
    Tự động phát hiện cột chứa sample_id và gloss_id/label dựa trên
    tên header. Hỗ trợ nhiều format CSV phổ biến.
    """
    if label_file is None or not os.path.exists(label_file):
        print(f"[Labels] ⚠️  File không tồn tại: {label_file}")
        return None

    if label_file.endswith(".json"):
        with open(label_file, "r") as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}

    mapping = {}
    with open(label_file, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        return mapping

    # Phát hiện header: nếu dòng đầu chứa chữ cái → là header
    header = rows[0]
    has_header = any(not cell.strip().lstrip("-").isdigit() for cell in header
                     if cell.strip())

    if has_header:
        # Tìm cột sample_id (chứa 'sample', 'video', 'id', 'name', 'file')
        # Tìm cột label (chứa 'gloss', 'label', 'class', 'target')
        col_names = [h.strip().lower() for h in header]
        id_col = 0   # mặc định cột đầu
        label_col = 1  # mặc định cột thứ 2

        # Ưu tiên exact match trước, rồi mới substring match
        # CSV format: center,left,right,ID,label_id
        id_keywords = ["sample_id", "video_id", "id", "sample", "video",
                       "name", "file", "basename"]
        label_keywords = ["label_id", "gloss_id", "label", "gloss",
                          "class_id", "class", "target", "category"]

        id_found = False
        for kw in id_keywords:
            if id_found:
                break
            for i, name in enumerate(col_names):
                if name == kw or (kw in name and name not in
                                  [c for c in col_names if "label" in c]):
                    id_col = i
                    id_found = True
                    break

        label_found = False
        for kw in label_keywords:
            if label_found:
                break
            for i, name in enumerate(col_names):
                if kw in name:
                    label_col = i
                    label_found = True
                    break

        print(f"[Labels] Header: {header}")
        print(f"[Labels] Sử dụng cột {id_col} ('{header[id_col]}') làm ID, "
              f"cột {label_col} ('{header[label_col]}') làm label")
        data_rows = rows[1:]
    else:
        # Không có header: giả định cột 0 = id, cột 1 = label
        id_col, label_col = 0, 1
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(id_col, label_col):
            continue
        # Trích sample_id: lấy phần số từ tên file
        # Ví dụ: "16649_kf_rgb.mp4" → "16649", hoặc "16649" → "16649"
        raw_id = row[id_col].strip()
        sid = raw_id.split("_")[0]  # Lấy phần trước dấu _ đầu tiên
        sid = re.sub(r"[^0-9]", "", sid)  # Chỉ giữ số
        if sid:
            try:
                mapping[sid] = int(row[label_col].strip())
            except ValueError:
                continue

    print(f"[Labels] Loaded {len(mapping)} entries từ {label_file}")
    return mapping


# ============================================================================
# DATASET: Multi-View MM-WLAuslan
# ============================================================================

class MultiViewDataset(Dataset):
    """
    Mỗi sample gồm 3 video (kl, kf, kr) cùng sample_id.
    Nếu thiếu view → padding bằng zero tensor.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    VIEWS = ["kl", "kf", "kr"]  # Left, Front(center), Right

    def __init__(self, split_dir, label_map=None, num_frames=16,
                 img_size=384, num_classes=3215):
        super().__init__()
        self.split_dir = split_dir
        self.label_map = label_map
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_classes = num_classes
        self.short_side = int(256.0 / 224 * img_size)

        # Quét thư mục, nhóm theo sample_id
        # Chỉ giữ samples có label (nếu label_map được cung cấp)
        self.samples = self._scan_videos()
        if label_map:
            before = len(self.samples)
            self.samples = [(sid, vp) for sid, vp in self.samples
                           if sid in label_map]
            print(f"[Dataset] {os.path.basename(split_dir)}: "
                  f"{len(self.samples)}/{before} samples có label (REAL)")
        else:
            print(f"[Dataset] {os.path.basename(split_dir)}: "
                  f"{len(self.samples)} samples (FAKE labels)")

    def _scan_videos(self):
        """Nhóm video theo sample_id. Trả về list[(sample_id, {view: path})]"""
        groups = defaultdict(dict)
        for fname in os.listdir(self.split_dir):
            if not fname.endswith(".mp4"):
                continue
            # Pattern: {sample_id}_{view}_rgb.mp4
            m = re.match(r"^(\d+)_(k[flr])_rgb\.mp4$", fname)
            if m:
                sid, view = m.group(1), m.group(2)
                groups[sid][view] = os.path.join(self.split_dir, fname)
        # Sắp xếp theo sample_id
        return sorted(groups.items(), key=lambda x: int(x[0]))

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path):
        """Đọc video, sample frames, tiền xử lý → tensor (C, T, H, W)."""
        from decord import VideoReader, cpu
        import src.datasets.utils.video.transforms as vtf
        import src.datasets.utils.video.volume_transforms as vol

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return None

        # Sample đều num_frames frame
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total)
            pad = np.full(self.num_frames - total, total - 1, dtype=int)
            indices = np.concatenate([indices, pad])

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

        # Tiền xử lý: Resize → CenterCrop → ToTensor → Normalize
        transform = vtf.Compose([
            vtf.Resize(self.short_side, interpolation="bilinear"),
            vtf.CenterCrop(size=(self.img_size, self.img_size)),
            vol.ClipToTensor(),
            vtf.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        frame_list = [frames[i] for i in range(frames.shape[0])]
        return transform(frame_list)  # (C, T, H, W)

    def __getitem__(self, idx):
        sid, view_paths = self.samples[idx]

        # Load 3 views (hoặc zero tensor nếu thiếu)
        videos = []
        for view in self.VIEWS:
            if view in view_paths:
                try:
                    tensor = self._load_video(view_paths[view])
                    if tensor is not None:
                        videos.append(tensor)
                        continue
                except Exception:
                    pass
            # Thiếu view → zero tensor
            C, T = 3, self.num_frames
            H = W = self.img_size
            videos.append(torch.zeros(C, T, H, W))

        # Label
        if self.label_map and sid in self.label_map:
            label = self.label_map[sid]
        else:
            label = int(sid) % self.num_classes  # Fake fallback

        # Trả về tuple 3 tensors + label
        return videos[0], videos[1], videos[2], label


# ============================================================================
# MODEL
# ============================================================================

class AttentiveProbe(nn.Module):
    """TransformerEncoder + Learnable Query + CrossAttention + Classifier."""

    def __init__(self, embed_dim, num_heads=8, depth=4,
                 num_classes=3215, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Self-Attention Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm_enc = nn.LayerNorm(embed_dim)

        # Learnable Query + Cross-Attention
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ca = nn.LayerNorm(embed_dim)

        # FFN
        hid = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, embed_dim), nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Classifier
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """x: (B, N, D) → logits (B, num_classes)"""
        x = self.norm_enc(self.encoder(x))

        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            self.norm_q(q), self.norm_kv(x), self.norm_kv(x)
        )
        q = self.norm_ca(q + attn_out)
        q = self.norm_ffn(q + self.ffn(q))

        return self.head(q.squeeze(1))


class MultiViewSLRModel(nn.Module):
    """
    3-View Fusion:
    1. Backbone (frozen): V-JEPA 2.1 – xử lý tuần tự từng view để tiết kiệm VRAM
    2. Concat features: cat([feat_L, feat_C, feat_R], dim=1)
    3. Probe (trainable): TransformerEncoder + Query + CrossAttn + Linear
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = self._build_backbone(cfg)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.probe = AttentiveProbe(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.probe_heads,
            depth=cfg.probe_depth,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _build_backbone(cfg):
        """Khởi tạo V-JEPA 2.1 encoder và load pretrained weights."""
        # QUAN TRỌNG: V-JEPA 2.1 dùng VisionTransformer riêng
        # ở app.vjepa_2_1.models, KHÁC với src.models
        import app.vjepa_2_1.models.vision_transformer as vit21

        encoder = vit21.__dict__[cfg.backbone_arch](
            patch_size=cfg.patch_size,
            img_size=(cfg.img_size, cfg.img_size),
            num_frames=cfg.num_frames,
            tubelet_size=cfg.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        # Load weights
        ckpt_path = cfg.backbone_ckpt_local
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[Backbone] Loading from local: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        else:
            print(f"[Backbone] Downloading: {cfg.backbone_ckpt_url}")
            sd = torch.hub.load_state_dict_from_url(
                cfg.backbone_ckpt_url, map_location="cpu"
            )
        enc_sd = sd[cfg.backbone_ckpt_key]
        enc_sd = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in enc_sd.items()}
        msg = encoder.load_state_dict(enc_sd, strict=False)
        print(f"[Backbone] Loaded: {msg}")
        return encoder

    def _extract_features(self, video):
        """Trích xuất features từ 1 view. video: (B, C, T, H, W)."""
        with torch.no_grad():
            feats = self.backbone(video)  # (B, N, D)
        return feats

    def forward(self, v_left, v_center, v_right):
        """
        3 views: mỗi cái (B, C, T, H, W).
        Xử lý tuần tự qua backbone để tiết kiệm VRAM.
        """
        feat_l = self._extract_features(v_left)
        feat_c = self._extract_features(v_center)
        feat_r = self._extract_features(v_right)

        # Ghép 3 chuỗi đặc trưng: (B, 3*N, D)
        fused = torch.cat([feat_l, feat_c, feat_r], dim=1)

        # Probe (có gradient)
        logits = self.probe(fused)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Luôn giữ backbone eval
        return self


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def collate_fn(batch):
    """Custom collate cho tuple (v_left, v_center, v_right, label)."""
    vl = torch.stack([b[0] for b in batch])
    vc = torch.stack([b[1] for b in batch])
    vr = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return vl, vc, vr, labels


def warmup_cosine_lr(optimizer, epoch, step, steps_per_epoch, cfg):
    """Warmup + Cosine Annealing learning rate."""
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    current = epoch * steps_per_epoch + step

    if current < warmup_steps:
        lr = cfg.lr * current / max(warmup_steps, 1)
    else:
        progress = (current - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = 1e-6 + (cfg.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg,
                    scaler=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    ipe = len(loader)

    for step, (vl, vc, vr, labels) in enumerate(loader):
        lr = warmup_cosine_lr(optimizer, epoch, step, ipe, cfg)
        vl = vl.to(device, non_blocking=True)
        vc = vc.to(device, non_blocking=True)
        vr = vr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler and cfg.use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(vl, vc, vr)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.probe.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(vl, vc, vr)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.probe.parameters(), 1.0)
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += logits.detach().argmax(1).eq(labels).sum().item()
        total_samples += bs

        if step % 20 == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [E{epoch+1}][{step}/{ipe}] loss={loss.item():.4f} "
                  f"lr={lr:.2e} mem={mem:.1f}GB")

    avg_loss = total_loss / max(total_samples, 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for vl, vc, vr, labels in loader:
        vl = vl.to(device, non_blocking=True)
        vc = vc.to(device, non_blocking=True)
        vr = vr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg.use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(vl, vc, vr)
                loss = criterion(logits, labels)
        else:
            logits = model(vl, vc, vr)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += logits.argmax(1).eq(labels).sum().item()
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB)")

    # ---- Labels (riêng cho từng split) ----
    print("\n" + "=" * 70)
    print("LOADING LABELS")
    print("=" * 70)
    train_labels = load_labels(cfg.train_label_file)
    val_labels = load_labels(cfg.val_label_file)

    # ---- Dataset ----
    train_dir = os.path.join(cfg.video_dir, "train")
    valid_dir = os.path.join(cfg.video_dir, "valid")

    train_dataset = MultiViewDataset(
        train_dir, train_labels, cfg.num_frames, cfg.img_size, cfg.num_classes)
    val_dataset = MultiViewDataset(
        valid_dir, val_labels, cfg.num_frames, cfg.img_size, cfg.num_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=cfg.num_workers > 0,
    )
    print(f"[Data] Train: {len(train_dataset)} samples, "
          f"{len(train_loader)} iters/epoch")
    print(f"[Data] Valid: {len(val_dataset)} samples")

    # ---- Model ----
    model = MultiViewSLRModel(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total_p/1e6:.0f}M | "
          f"Trainable (Probe): {train_p/1e6:.1f}M | "
          f"Frozen (Backbone): {(total_p-train_p)/1e6:.0f}M")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp else None

    # ---- Checkpoint dir ----
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_acc = 0.0

    # ---- Training ----
    print("\n" + "=" * 70)
    print("BẮT ĐẦU HUẤN LUYỆN MULTI-VIEW")
    print("=" * 70)

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, cfg,
            scaler)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, cfg)
        elapsed = time.time() - t0

        print(f"\n>>> Epoch {epoch+1}/{cfg.num_epochs} ({elapsed:.0f}s)")
        print(f"    Train: loss={train_loss:.4f} acc={train_acc:.2f}%")
        print(f"    Valid: loss={val_loss:.4f} acc={val_acc:.2f}%")

        # Lưu best checkpoint (CHỈ PROBE)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            path = os.path.join(cfg.save_dir, "best_probe.pt")
            torch.save({
                "epoch": epoch + 1,
                "probe_state_dict": model.probe.state_dict(),
                "best_val_acc": best_val_acc,
                "config": {k: v for k, v in vars(cfg).items()
                           if not k.startswith("_")},
            }, path)
            print(f"    ✅ Best saved: {path} (acc={best_val_acc:.2f}%)")

        # Lưu checkpoint định kỳ
        if (epoch + 1) % cfg.save_every == 0:
            path = os.path.join(cfg.save_dir, f"probe_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "probe_state_dict": model.probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": train_acc,
                "val_acc": val_acc,
            }, path)
            print(f"    💾 Periodic saved: {path}")

    print(f"\n{'='*70}")
    print(f"HOÀN TẤT! Best Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
