#!/usr/bin/env python3
"""
Multi-View Sign Language Recognition (MM-WLAuslan)
V-JEPA 2.1 backbone (frozen) + Attentive Probe (trainable)

Usage:
    python train_multiview_vsl.py                  # Train từ đầu
    python train_multiview_vsl.py --resume          # Resume từ checkpoint
"""

import argparse
import csv
import math
import os
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # ── Data ──
    data_root       = "/mnt/sda1/VSLR_Storage/MM-WLAuslan"
    video_dir       = f"{data_root}/videos"
    label_dir       = f"{data_root}/labels_clean_200_full"
    train_label     = f"{label_dir}/train_labels_clean.csv"
    val_label       = f"{label_dir}/val_labels_clean.csv"
    test_label      = f"{label_dir}/test_labels_clean.csv"

    # ── Backbone V-JEPA 2.1 ──
    # vjepa2_1_vit_base_384     (86M,  embed=768)
    # vjepa2_1_vit_large_384    (304M, embed=1024)
    # vjepa2_1_vit_giant_384    (1B,   embed=1408)
    # vjepa2_1_vit_gigantic_384 (2B,   embed=1664)
    hub_name   = "vjepa2_1_vit_base_384"
    embed_dim  = 768
    img_size   = 384
    patch_size = 16
    tubelet    = 2
    num_frames = 16

    # ── Probe ──
    probe_depth   = 4
    probe_heads   = 8
    num_queries   = 4      # Nhiều query tokens → capture đa dạng hơn
    num_classes   = 200
    dropout       = 0.1
    mlp_ratio     = 4.0

    # ── Training ──
    batch_size    = 64
    grad_accum    = 1      # Gradient accumulation steps (effective_bs = 64*1)
    num_epochs    = 30
    lr            = 2e-3
    min_lr        = 1e-6
    weight_decay  = 0.05
    warmup_epochs = 3
    num_workers   = 8
    use_amp       = True
    compile_model = False  # torch.compile (PyTorch 2.0+, tăng tốc ~20%)
    gpu_id        = "1"    # CUDA device

    # ── Checkpoint ──
    save_dir   = "./checkpoints_multiview"
    save_every = 5
    log_every  = 10        # Log mỗi N steps


# ============================================================================
# LABEL LOADER
# ============================================================================

def load_labels(path):
    """Load CSV: auto-detect ID & label columns. Returns {sample_id: label}."""
    if not path or not os.path.exists(path):
        print(f"  ⚠ Label file not found: {path}")
        return None

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}

    header = rows[0]
    col = {h.strip().lower(): i for i, h in enumerate(header)}

    # Tìm cột ID và label
    id_col = col.get("id", col.get("sample_id", col.get("video_id", 0)))
    label_col = col.get("label_id", col.get("gloss_id",
                col.get("label", col.get("class_id", 1))))

    mapping = {}
    for row in rows[1:]:
        if len(row) <= max(id_col, label_col):
            continue
        sid = re.sub(r"\D", "", row[id_col].split("_")[0])
        if sid:
            try:
                mapping[sid] = int(row[label_col].strip())
            except ValueError:
                pass

    print(f"  ✔ {Path(path).name}: {len(mapping)} entries "
          f"(cols: {header[id_col]}→ID, {header[label_col]}→label)")
    return mapping


# ============================================================================
# DATASET
# ============================================================================

class MultiViewDataset(Dataset):
    """3-view video dataset (kl=Left, kf=Front, kr=Right)."""

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    VIEWS = ("kl", "kf", "kr")

    def __init__(self, split_dir, labels=None, cfg=None):
        cfg = cfg or Config()
        self.split_dir = split_dir
        self.labels = labels or {}
        self.num_frames = cfg.num_frames
        self.img_size = cfg.img_size
        self.num_classes = cfg.num_classes
        self.short_side = int(256 / 224 * cfg.img_size)

        # Cache transform
        import src.datasets.utils.video.transforms as vtf
        import src.datasets.utils.video.volume_transforms as vol
        self._transform = vtf.Compose([
            vtf.Resize(self.short_side, interpolation="bilinear"),
            vtf.CenterCrop(size=(self.img_size, self.img_size)),
            vol.ClipToTensor(),
            vtf.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # Scan & filter
        all_samples = self._scan()
        if labels:
            self.samples = [(s, v) for s, v in all_samples if s in labels]
            print(f"  {os.path.basename(split_dir)}: "
                  f"{len(self.samples)}/{len(all_samples)} samples (labeled)")
        else:
            self.samples = all_samples
            print(f"  {os.path.basename(split_dir)}: {len(self.samples)} samples")

    def _scan(self):
        groups = defaultdict(dict)
        for f in os.listdir(self.split_dir):
            m = re.match(r"^(\d+)_(k[flr])_rgb\.mp4$", f)
            if m:
                groups[m[1]][m[2]] = os.path.join(self.split_dir, f)
        return sorted(groups.items(), key=lambda x: int(x[0]))

    def _load_video(self, path):
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        if n <= 0:
            return None
        if n >= self.num_frames:
            idx = np.linspace(0, n - 1, self.num_frames, dtype=int)
        else:
            idx = np.concatenate([
                np.arange(n),
                np.full(self.num_frames - n, n - 1, dtype=int)
            ])
        frames = vr.get_batch(idx).asnumpy()
        return self._transform([frames[i] for i in range(len(idx))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, paths = self.samples[idx]
        zero = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        views = []
        for v in self.VIEWS:
            if v in paths:
                try:
                    t = self._load_video(paths[v])
                    views.append(t if t is not None else zero)
                except Exception:
                    views.append(zero)
            else:
                views.append(zero)
        label = self.labels.get(sid, int(sid) % self.num_classes)
        return views[0], views[1], views[2], label


# ============================================================================
# MODEL
# ============================================================================

class AttentiveProbe(nn.Module):
    """
    Multi-Query Attentive Probe:
    - TransformerEncoder xử lý chuỗi features
    - Nhiều learnable query tokens + Cross-Attention
    - FFN + Classifier
    """

    def __init__(self, embed_dim, num_heads=8, depth=4, num_queries=4,
                 num_classes=200, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # View-type embeddings (Left / Front / Right)
        self.view_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        # Self-Attention Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.norm_enc = nn.LayerNorm(embed_dim)

        # Multi-Query Cross-Attention
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q  = nn.LayerNorm(embed_dim)
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

    def forward(self, x, num_temporal):
        """
        x: (B, 3*T, D) — 3 views concatenated along sequence dim.
        num_temporal: T (tokens per view).
        """
        B, N, D = x.shape

        # Thêm view embeddings: mỗi view nhận embedding riêng
        ve = self.view_embed.repeat(1, num_temporal, 1)  # (1, 3*T, D)
        x = x + ve[:, :N, :]

        # Self-Attention
        x = self.norm_enc(self.encoder(x))

        # Cross-Attention: queries attend to encoded features
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            self.norm_q(q), self.norm_kv(x), self.norm_kv(x)
        )
        q = self.norm_ca(q + attn_out)
        q = self.norm_ffn(q + self.ffn(q))

        # Pool queries → single vector → classify
        return self.head(q.mean(dim=1))


class MultiViewSLRModel(nn.Module):
    """Frozen V-JEPA 2.1 backbone + Trainable Attentive Probe."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_temporal = cfg.num_frames // cfg.tubelet

        # Backbone (frozen)
        self.backbone = self._load_backbone(cfg)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Probe (trainable)
        self.probe = AttentiveProbe(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.probe_heads,
            depth=cfg.probe_depth,
            num_queries=cfg.num_queries,
            num_classes=cfg.num_classes,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _load_backbone(cfg):
        print(f"[Backbone] Loading {cfg.hub_name} ...")
        encoder, _ = torch.hub.load(
            '.', cfg.hub_name, source='local', pretrained=True,
        )
        print(f"[Backbone] ✔ {cfg.hub_name} loaded")
        return encoder

    def _pool_spatial(self, feats):
        """(B, T*S, D) → (B, T, D) via spatial mean pooling."""
        B, N, D = feats.shape
        T = self.num_temporal
        return feats.view(B, T, N // T, D).mean(dim=2)

    @torch.no_grad()
    def _extract(self, video):
        """Extract & pool features from one view."""
        return self._pool_spatial(self.backbone(video))

    def forward(self, v_left, v_front, v_right):
        feat_l = self._extract(v_left)    # (B, T, D)
        feat_f = self._extract(v_front)
        feat_r = self._extract(v_right)

        fused = torch.cat([feat_l, feat_f, feat_r], dim=1)  # (B, 3T, D)
        return self.probe(fused, self.num_temporal)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


# ============================================================================
# TRAINING
# ============================================================================

def collate_fn(batch):
    vl = torch.stack([b[0] for b in batch])
    vc = torch.stack([b[1] for b in batch])
    vr = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return vl, vc, vr, labels


def get_lr(epoch, step, steps_per_epoch, cfg):
    """Warmup + Cosine Annealing → returns lr value."""
    total = cfg.num_epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch
    cur = epoch * steps_per_epoch + step
    if cur < warmup:
        return cfg.lr * cur / max(warmup, 1)
    progress = (cur - warmup) / max(total - warmup, 1)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def topk_accuracy(logits, labels, topk=(1, 5)):
    """Compute top-k accuracy."""
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels.unsqueeze(1))
    res = {}
    for k in topk:
        if k <= logits.size(1):
            res[k] = correct[:, :k].any(dim=1).float().sum().item()
        else:
            res[k] = correct[:, :maxk].any(dim=1).float().sum().item()
    return res


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    epoch, cfg):
    model.train()
    stats = {"loss": 0, "top1": 0, "top5": 0, "n": 0}
    ipe = len(loader)

    pbar = tqdm(enumerate(loader), total=ipe,
                desc=f"  ⚡ Train E{epoch+1:02d}",
                bar_format='{l_bar}{bar:30}{r_bar}',
                dynamic_ncols=True, leave=True)

    optimizer.zero_grad(set_to_none=True)

    for step, (vl, vc, vr, labels) in pbar:
        # LR schedule
        lr = get_lr(epoch, step, ipe, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        vl = vl.to(device, non_blocking=True)
        vc = vc.to(device, non_blocking=True)
        vr = vr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(vl, vc, vr)
            loss = criterion(logits, labels) / cfg.grad_accum

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (step + 1) % cfg.grad_accum == 0 or step == ipe - 1:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.probe.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.probe.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Stats
        bs = labels.size(0)
        stats["loss"] += loss.item() * cfg.grad_accum * bs
        acc = topk_accuracy(logits.detach(), labels)
        stats["top1"] += acc[1]
        stats["top5"] += acc[5]
        stats["n"] += bs

        if step % cfg.log_every == 0:
            n = stats["n"]
            mem = torch.cuda.max_memory_allocated() / 1e9
            pbar.set_postfix_str(
                f"loss={stats['loss']/n:.3f} "
                f"top1={100*stats['top1']/n:.1f}% "
                f"top5={100*stats['top5']/n:.1f}% "
                f"lr={lr:.1e} mem={mem:.1f}G")

    n = max(stats["n"], 1)
    return stats["loss"]/n, 100*stats["top1"]/n, 100*stats["top5"]/n


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    stats = {"loss": 0, "top1": 0, "top5": 0, "n": 0}

    pbar = tqdm(loader, total=len(loader),
                desc="  ✔ Valid     ",
                bar_format='{l_bar}{bar:30}{r_bar}',
                dynamic_ncols=True, leave=True)

    for vl, vc, vr, labels in pbar:
        vl = vl.to(device, non_blocking=True)
        vc = vc.to(device, non_blocking=True)
        vr = vr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(vl, vc, vr)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        stats["loss"] += loss.item() * bs
        acc = topk_accuracy(logits, labels)
        stats["top1"] += acc[1]
        stats["top5"] += acc[5]
        stats["n"] += bs

        n = stats["n"]
        pbar.set_postfix_str(
            f"top1={100*stats['top1']/n:.1f}% "
            f"top5={100*stats['top5']/n:.1f}%")

    n = max(stats["n"], 1)
    return stats["loss"]/n, 100*stats["top1"]/n, 100*stats["top5"]/n


# ============================================================================
# MAIN
# ============================================================================

def save_checkpoint(model, optimizer, scaler, epoch, best_acc, cfg, name):
    path = os.path.join(cfg.save_dir, name)
    torch.save({
        "epoch": epoch,
        "probe": model.probe.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "best_val_acc": best_acc,
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} ({gpu_mem:.0f}GB)")

    # ── Labels ──
    print("\n┌─── Loading Labels ───")
    train_labels = load_labels(cfg.train_label)
    val_labels = load_labels(cfg.val_label)
    print("└─────────────────────")

    # ── Dataset ──
    print("\n┌─── Building Datasets ───")
    train_ds = MultiViewDataset(
        os.path.join(cfg.video_dir, "train"), train_labels, cfg)
    val_ds = MultiViewDataset(
        os.path.join(cfg.video_dir, "valid"), val_labels, cfg)
    print("└─────────────────────────")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2,
    )

    # ── Model ──
    model = MultiViewSLRModel(cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model.probe = torch.compile(model.probe)
        print("[Model] ✔ torch.compile enabled for probe")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total_p/1e6:.0f}M | "
          f"Trainable: {train_p/1e6:.1f}M | "
          f"Frozen: {(total_p-train_p)/1e6:.0f}M")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp else None

    # ── Resume ──
    start_epoch = 0
    best_val_acc = 0.0
    ckpt_path = os.path.join(cfg.save_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.probe.load_state_dict(ckpt["probe"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt.get("best_val_acc", 0)
        print(f"[Resume] ✔ Epoch {start_epoch}, best_acc={best_val_acc:.2f}%")

    # ── Training ──
    eff_bs = cfg.batch_size * cfg.grad_accum
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ 🚀 MULTI-VIEW SIGN LANGUAGE RECOGNITION                        │
│   Backbone : {cfg.hub_name:<50s}│
│   Probe    : {cfg.probe_depth}L / {cfg.probe_heads}H / {cfg.num_queries}Q {'':40s}│
│   Classes  : {cfg.num_classes:<5d} | Batch: {eff_bs:<4d} | LR: {cfg.lr:<20s}│
│   Train    : {len(train_ds):<5d} | Valid: {len(val_ds):<28d}│
└──────────────────────────────────────────────────────────────────┘""")

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch, cfg)

        vl_loss, vl_top1, vl_top5 = evaluate(
            model, val_loader, criterion, device, cfg)

        elapsed = time.time() - t0
        is_best = vl_top1 > best_val_acc

        # Epoch summary
        star = " ⭐" if is_best else ""
        print(f"\n  ┌── Epoch {epoch+1:02d}/{cfg.num_epochs} "
              f"{'─'*42} ({elapsed:.0f}s)")
        print(f"  │ Train │ loss={tr_loss:.4f}  "
              f"top1={tr_top1:.2f}%  top5={tr_top5:.2f}%")
        print(f"  │ Valid │ loss={vl_loss:.4f}  "
              f"top1={vl_top1:.2f}%  top5={vl_top5:.2f}%{star}")

        # Save best
        if is_best:
            best_val_acc = vl_top1
            p = save_checkpoint(model, optimizer, scaler,
                                epoch + 1, best_val_acc, cfg, "best_probe.pt")
            print(f"  │ ✔ Best │ {best_val_acc:.2f}% → {p}")

        # Save periodic + latest (for resume)
        save_checkpoint(model, optimizer, scaler,
                        epoch + 1, best_val_acc, cfg, "latest.pt")
        if (epoch + 1) % cfg.save_every == 0:
            p = save_checkpoint(model, optimizer, scaler,
                                epoch + 1, best_val_acc, cfg,
                                f"probe_epoch{epoch+1:02d}.pt")
            print(f"  │ 💾 Save │ {p}")

        print(f"  └{'─'*62}")

    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ ✅ DONE! Best Val Top-1: {best_val_acc:.2f}%{'':<37s}│
└──────────────────────────────────────────────────────────────────┘""")


if __name__ == "__main__":
    main()
