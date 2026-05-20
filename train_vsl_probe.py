#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script huấn luyện Attentive Probe cho bài toán Nhận dạng Ngôn ngữ Ký hiệu (VSL)
sử dụng V-JEPA 2.1 làm backbone (đóng băng) và video thô làm đầu vào.

Chạy trên Colab/Kaggle:
    1. Mount Google Drive
    2. cd vjepa2 && pip install -e .
    3. python train_vsl_probe.py

Tác giả: AI Engineer
"""

import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# CẤU HÌNH TỔNG THỂ
# ============================================================================

class Config:
    """Tập trung toàn bộ siêu tham số để dễ chỉnh sửa."""

    # ----- Đường dẫn dữ liệu -----
    video_root = "/content/drive/MyDrive/Downloaded_Folder"
    filter_keyword = "___center_"  # Chỉ lấy video chứa chuỗi này

    # ----- Backbone V-JEPA 2.1 -----
    # Dùng ViT-Large/16 (300M params, resolution 384) – phù hợp VRAM Colab
    backbone_name = "vit_large"           # tên hàm trong vision_transformer
    backbone_ckpt_url = (
        "https://dl.fbaipublicfiles.com/vjepa2/"
        "vjepa2_1_vitl_dist_vitG_384.pt"
    )
    backbone_ckpt_key = "ema_encoder"     # key trong state_dict
    img_size = 384                        # resolution đầu vào
    patch_size = 16
    tubelet_size = 2
    num_frames = 16                       # số frame sample từ mỗi video
    embed_dim = 1024                      # embed_dim của ViT-Large

    # ----- Attentive Probe -----
    probe_depth = 2                       # số layer TransformerEncoder
    probe_heads = 8
    num_classes = 200                     # số từ vựng ký hiệu

    # ----- Huấn luyện -----
    batch_size = 2
    num_epochs = 20
    lr = 3e-4
    weight_decay = 0.01
    num_workers = 2

    # ----- Checkpoint -----
    save_dir = "./checkpoints_vsl"
    ckpt_path = None  # Đường dẫn local tới file .pt (None = tự download)


# ============================================================================
# DATASET – Đọc video thô, tiền xử lý theo chuẩn V-JEPA 2.1
# ============================================================================

class VideoDataset(Dataset):
    """
    Dataset đọc video thô từ thư mục, lọc theo keyword, tiền xử lý
    và trả về tensor (C, T, H, W) cùng fake label.

    Pipeline tiền xử lý:
        1. Đọc video bằng decord (nhanh, hỗ trợ GPU decoding)
        2. Sample đều num_frames frame từ video
        3. Resize cạnh ngắn → short_side_size (giữ tỉ lệ)
        4. Center Crop → (img_size x img_size)
        5. Chuyển sang tensor float [0, 1]
        6. Normalize theo ImageNet mean/std (chuẩn V-JEPA 2.1)
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, video_root, filter_keyword, num_frames=16,
                 img_size=384):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        # Tính short_side_size theo công thức chuẩn của V-JEPA 2
        self.short_side_size = int(256.0 / 224 * img_size)

        # Thu thập tất cả file video chứa keyword
        all_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            all_files.extend(glob.glob(os.path.join(video_root, "**", ext),
                                       recursive=True))
        self.video_paths = sorted(
            [f for f in all_files if filter_keyword in os.path.basename(f)]
        )
        if len(self.video_paths) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy video nào chứa '{filter_keyword}' "
                f"trong '{video_root}'"
            )
        print(f"[VideoDataset] Tìm thấy {len(self.video_paths)} video.")

    def __len__(self):
        return len(self.video_paths)

    def _load_and_sample_frames(self, video_path):
        """
        Bước 1-2: Đọc video bằng decord và sample đều num_frames frame.
        Trả về numpy array (T, H, W, C) uint8.
        """
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames <= 0:
            raise RuntimeError(f"Video rỗng: {video_path}")

        # Sample đều num_frames frame từ toàn bộ video
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1,
                                  self.num_frames, dtype=int)
        else:
            # Nếu video quá ngắn, lặp lại frame cuối
            indices = np.arange(total_frames)
            pad = np.full(self.num_frames - total_frames,
                          total_frames - 1, dtype=int)
            indices = np.concatenate([indices, pad])

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        return frames

    def _preprocess(self, frames):
        """
        Bước 3-6: Resize, CenterCrop, ToTensor, Normalize.
        Input:  numpy (T, H, W, C) uint8
        Output: tensor (C, T, H, W) float32 đã normalize
        """
        import src.datasets.utils.video.transforms as vtf
        import src.datasets.utils.video.volume_transforms as vol

        transform = vtf.Compose([
            # Bước 3: Resize cạnh ngắn về short_side_size, giữ tỉ lệ
            vtf.Resize(self.short_side_size, interpolation="bilinear"),
            # Bước 4: Cắt giữa thành hình vuông (img_size x img_size)
            vtf.CenterCrop(size=(self.img_size, self.img_size)),
            # Bước 5: Chuyển list[ndarray] → tensor (C, T, H, W), chia 255
            vol.ClipToTensor(),
            # Bước 6: Chuẩn hoá theo ImageNet mean/std
            vtf.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # vtf.Compose nhận list of numpy arrays (T, H, W, C)
        frame_list = [frames[i] for i in range(frames.shape[0])]
        tensor = transform(frame_list)  # (C, T, H, W)
        return tensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self._load_and_sample_frames(video_path)
        tensor = self._preprocess(frames)  # (C, T, H, W)

        # Fake label: dùng index mod num_classes
        label = idx % Config.num_classes
        return tensor, label


# ============================================================================
# MODEL – VSL_Model = Frozen Backbone + Attentive Probe
# ============================================================================

class AttentiveProbe(nn.Module):
    """
    Attentive Probe gồm:
        1. TransformerEncoder (self-attention) để tinh chỉnh features
        2. Learnable Query Token + Cross-Attention để nén chuỗi → 1 vector
        3. Linear head phân loại

    Cross-Attention: query (1 token) attend vào toàn bộ chuỗi encoder output.
    """

    def __init__(self, embed_dim=1024, num_heads=8, depth=2,
                 num_classes=200, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # --- 1. TransformerEncoder (Self-Attention) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )
        self.norm_after_enc = nn.LayerNorm(embed_dim)

        # --- 2. Learnable Query Token + Cross-Attention ---
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)

        # Cross-Attention: Q từ query_token, K/V từ encoder output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_after_ca = nn.LayerNorm(embed_dim)

        # FFN sau cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_after_ffn = nn.LayerNorm(embed_dim)

        # --- 3. Classification Head ---
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, N, D) – features từ backbone
        return: (B, num_classes) – logits
        """
        # Self-attention refinement
        x = self.transformer_encoder(x)
        x = self.norm_after_enc(x)

        # Cross-attention: query attend vào encoder output
        B = x.size(0)
        q = self.query_token.expand(B, -1, -1)       # (B, 1, D)
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(x)
        attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
        q = q + attn_out                              # residual
        q = self.norm_after_ca(q)

        # FFN + residual
        q = q + self.ffn(q)
        q = self.norm_after_ffn(q)

        # Squeeze temporal dim và phân loại
        q = q.squeeze(1)                               # (B, D)
        logits = self.classifier(q)                    # (B, num_classes)
        return logits


class VSL_Model(nn.Module):
    """
    End-to-end model:
        - Backbone: V-JEPA 2.1 ViT (đóng băng hoàn toàn)
        - Probe: AttentiveProbe (huấn luyện)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ---- 1. Khởi tạo backbone V-JEPA 2.1 ----
        self.backbone = self._build_backbone(cfg)

        # Đóng băng toàn bộ backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---- 2. Khởi tạo Attentive Probe ----
        self.probe = AttentiveProbe(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.probe_heads,
            depth=cfg.probe_depth,
            num_classes=cfg.num_classes,
        )

    @staticmethod
    def _build_backbone(cfg):
        """
        Xây dựng encoder V-JEPA 2.1 từ source code và load pretrained weights.
        """
        import src.models.vision_transformer as vit_module

        # Tạo kiến trúc ViT với các tham số khớp V-JEPA 2.1
        encoder = vit_module.__dict__[cfg.backbone_name](
            patch_size=cfg.patch_size,
            img_size=(cfg.img_size, cfg.img_size),
            num_frames=cfg.num_frames,
            tubelet_size=cfg.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )

        # Load pretrained weights
        ckpt_path = cfg.ckpt_path
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[Backbone] Đang tải weights từ: {cfg.backbone_ckpt_url}")
            state_dict = torch.hub.load_state_dict_from_url(
                cfg.backbone_ckpt_url, map_location="cpu"
            )
        else:
            print(f"[Backbone] Đang tải weights từ file local: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu",
                                    weights_only=True)

        # Làm sạch key và load
        enc_sd = state_dict[cfg.backbone_ckpt_key]
        enc_sd = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in enc_sd.items()}
        msg = encoder.load_state_dict(enc_sd, strict=False)
        print(f"[Backbone] Load weights msg: {msg}")
        return encoder

    def forward(self, x):
        """
        x: (B, C, T, H, W) – video tensor đã tiền xử lý
        return: (B, num_classes) – logits
        """
        # Backbone: chạy KHÔNG tính gradient để tiết kiệm VRAM
        with torch.no_grad():
            features = self.backbone(x)   # (B, N_tokens, embed_dim)

        # Probe: BẬT gradient
        logits = self.probe(features)     # (B, num_classes)
        return logits

    def train(self, mode=True):
        """Override: luôn giữ backbone ở eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self


# ============================================================================
# TRAINING LOOP
# ============================================================================

def compute_accuracy(logits, labels):
    """Tính top-1 accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    scaler=None):
    """Huấn luyện 1 epoch, trả về (avg_loss, accuracy %)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (videos, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)   # (B, C, T, H, W)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(videos)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += compute_accuracy(logits.detach(), labels)
        total_samples += bs

        if step % 10 == 0:
            print(f"  [Epoch {epoch+1}][Step {step}/{len(loader)}] "
                  f"Loss={loss.item():.4f}  "
                  f"Mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main():
    cfg = Config()

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (cảnh báo: sẽ rất chậm!)")

    # ---- Dataset & DataLoader ----
    print("\n" + "=" * 60)
    print("KHỞI TẠO DATASET")
    print("=" * 60)
    dataset = VideoDataset(
        video_root=cfg.video_root,
        filter_keyword=cfg.filter_keyword,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    print(f"[DataLoader] {len(dataset)} videos, "
          f"{len(loader)} iterations/epoch, batch_size={cfg.batch_size}")

    # ---- Model ----
    print("\n" + "=" * 60)
    print("KHỞI TẠO MODEL")
    print("=" * 60)
    model = VSL_Model(cfg).to(device)

    # Đếm tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"[Model] Tổng params: {total_params/1e6:.1f}M")
    print(f"[Model] Trainable params (Probe): {trainable_params/1e6:.1f}M")
    print(f"[Model] Frozen params (Backbone): "
          f"{(total_params-trainable_params)/1e6:.1f}M")

    # ---- Optimizer & Scheduler ----
    # Chỉ optimize probe parameters
    optimizer = torch.optim.AdamW(
        model.probe.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Cosine annealing scheduler
    total_steps = cfg.num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    # AMP scaler cho GPU
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ---- Checkpoint directory ----
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_acc = 0.0

    # ---- Training Loop ----
    print("\n" + "=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 60)

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        avg_loss, accuracy = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch, scaler
        )
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n>>> Epoch {epoch+1}/{cfg.num_epochs} | "
              f"Loss={avg_loss:.4f} | Acc={accuracy:.2f}% | "
              f"LR={current_lr:.6f} | Time={elapsed:.1f}s")

        # Lưu checkpoint tốt nhất (CHỈ LƯU PROBE, KHÔNG LƯU BACKBONE)
        if accuracy > best_acc:
            best_acc = accuracy
            save_path = os.path.join(cfg.save_dir, "best_probe.pt")
            torch.save({
                "epoch": epoch + 1,
                "probe_state_dict": model.probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_acc,
                "config": {
                    "embed_dim": cfg.embed_dim,
                    "probe_depth": cfg.probe_depth,
                    "probe_heads": cfg.probe_heads,
                    "num_classes": cfg.num_classes,
                    "num_frames": cfg.num_frames,
                    "img_size": cfg.img_size,
                },
            }, save_path)
            print(f"    ✅ Đã lưu best probe checkpoint: {save_path} "
                  f"(Acc={best_acc:.2f}%)")

        # Lưu checkpoint mới nhất
        latest_path = os.path.join(cfg.save_dir, "latest_probe.pt")
        torch.save({
            "epoch": epoch + 1,
            "probe_state_dict": model.probe.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
        }, latest_path)

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT! Best Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint: {os.path.join(cfg.save_dir, 'best_probe.pt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
