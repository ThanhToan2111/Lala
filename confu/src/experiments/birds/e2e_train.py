"""End-to-end ConFu vs ConFu++ on the Bird-MML subset (trainable encoders).

Closes the frozen-encoder gap: everything (encoders + fusion + objectives) is
trained jointly, matching the spirit of the original ConFu bird protocol but on
the released 71-species subset.

Encoders (shared initializations across arms):
    image: ResNet18 (ImageNet-pretrained init, 3ch, 224px)
    audio: ResNet18 (from scratch, 1ch log-mel 128xT, 500-10kHz, per-clip norm)
    text:  MiniLM-L6 pretrained init, fine-tuned (max 48 tokens)

Objectives per arm (identical encoders/schedule/capacity at inference):
    confu      : pairwise + higher-order symmetric InfoNCE
    confu_u    : + task head (base/full) and conditional-utility loss
    confu_u_s1 : + per-pair adversarial de-shortcutting (S1)

Evaluation each run (best-val checkpoint):
    test accuracy of the full head, lower-order head; probe predictability
    R2(zi -> zij); per-modality shuffle drops through the full head;
    zero/shuffle interaction drops; correction-regression vs base head.

Usage:
    python -m src.experiments.birds.e2e_train --arm confu --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from PIL import Image
from torchvision import transforms

from src.modules.encoders.resnet import create_image_encoder, create_audio_encoder
from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import symmetric_info_nce

PAIRS = [(0, 1), (0, 2), (1, 2)]
MODALITY = ["image", "audio", "text"]


class BirdSubsetDataset(Dataset):
    def __init__(self, csv_path: str, root: str, indices: np.ndarray, tokenizer,
                 image_transform) -> None:
        df = pd.read_csv(csv_path).iloc[indices].reset_index(drop=True)
        self.root = root
        self.df = df
        self.tok = tokenizer
        self.image_transform = image_transform
        taxa = sorted(df["scientific_name"].unique())
        self.labels = df["scientific_name"].map({n: i for i, n in enumerate(taxa)}).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = self.image_transform(
                Image.open(os.path.join(self.root, row["photo_file"].replace("bird-mml/", "", 1))).convert("RGB"))
        except Exception:
            img = torch.zeros(3, 224, 224)
        try:
            wav, sr = torchaudio.load(os.path.join(self.root, row["audio_file"].replace("bird-mml/", "", 1)))
            wav = wav.mean(0)
            if sr != 22050:
                wav = torchaudio.functional.resample(wav, sr, 22050)
            wav = wav[: 22050 * 5]  # 5 s cap keeps mel width manageable
        except Exception:
            wav = torch.zeros(22050)
        enc = self.tok(str(row["combined_caption"]), padding="max_length", truncation=True,
                       max_length=48, return_tensors="pt")
        return (img, wav, enc["input_ids"][0], enc["attention_mask"][0],
                int(self.labels[idx]))


def mel_batch(wavs: list[Tensor], device: str) -> Tensor:
    import torchaudio as ta
    lens = [len(w) for w in wavs]
    maxlen = max(lens)
    x = torch.zeros(len(wavs), maxlen)
    for i, w in enumerate(wavs):
        x[i, : len(w)] = w
    mel = ta.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512,
                                       n_mels=128, f_min=500, f_max=10000).to(device)
    db = ta.transforms.AmplitudeToDB().to(device)
    with torch.no_grad():
        m = db(mel(x.to(device)))
        m = (m - m.mean()) / (m.std() + 1e-6)
    return m.unsqueeze(1)


class E2EConFu(nn.Module):
    def __init__(self, embed_dim: int, fusion_hidden: int, n_classes: int,
                 text_finetune: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel
        self.image_encoder = create_image_encoder(embed_dim, in_channels=3, backbone="resnet18")
        # imagenet init for image resnet18
        from torchvision.models import resnet18, ResNet18_Weights
        pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        enc_state = self.image_encoder.state_dict()
        for k, v in pretrained.state_dict().items():
            if k in enc_state and enc_state[k].shape == v.shape and not k.startswith("fc"):
                enc_state[k] = v
        self.image_encoder.load_state_dict(enc_state)
        self.audio_encoder = create_audio_encoder(embed_dim, in_channels=1, backbone="resnet18")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, embed_dim)
        if not text_finetune:
            for p in self.text_model.parameters():
                p.requires_grad_(False)
        self.fusions = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * embed_dim, fusion_hidden), nn.GELU(),
                          nn.Linear(fusion_hidden, embed_dim))
            for _ in PAIRS
        )
        self.head_base = nn.Linear(3 * embed_dim, n_classes)
        self.head_full = nn.Linear(6 * embed_dim, n_classes)

    def encode(self, img, mel, ids, mask) -> list[Tensor]:
        z1 = F.normalize(self.image_encoder(img)[0], dim=-1)
        z2 = F.normalize(self.audio_encoder(mel)[0], dim=-1)
        hidden = self.text_model(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        z3 = F.normalize(self.text_proj(pooled), dim=-1)
        return [z1, z2, z3]

    def forward(self, img, mel, ids, mask) -> dict[str, Tensor]:
        z = self.encode(img, mel, ids, mask)
        out = {"z1": z[0], "z2": z[1], "z3": z[2]}
        for (i, j), fusion in zip(PAIRS, self.fusions):
            out[f"z{i+1}{j+1}"] = F.normalize(fusion(torch.cat([z[i], z[j]], -1)), dim=-1)
        base = torch.cat([out["z1"], out["z2"], out["z3"]], -1)
        full = torch.cat([base, out["z12"], out["z13"], out["z23"]], -1)
        out["logits_base"] = self.head_base(base)
        out["logits_full"] = self.head_full(full)
        return out


def confu_terms(out: dict[str, Tensor]) -> Tensor:
    z1, z2, z3 = out["z1"], out["z2"], out["z3"]
    pairwise = symmetric_info_nce(z1, z2) + symmetric_info_nce(z1, z3) + symmetric_info_nce(z2, z3)
    higher = (symmetric_info_nce(out["z12"], z3) + symmetric_info_nce(out["z13"], z2)
              + symmetric_info_nce(out["z23"], z1))
    return (pairwise + higher) / 6.0


@torch.no_grad()
def evaluate(model, loader, device, shuffle_modality=None, zero_interaction=False):
    model.eval()
    preds_f, preds_b, ys = [], [], []
    store = {"z1": [], "z2": [], "z3": [], "z12": [], "z13": [], "z23": []}
    for img, wav, ids, mask, y in loader:
        img, ids, mask = img.to(device), ids.to(device), mask.to(device)
        mel = mel_batch(wav, device)
        if shuffle_modality is not None:
            perm = torch.randperm(len(y), device=device)
            if shuffle_modality == 0: img = img[perm]
            elif shuffle_modality == 1: mel = mel[perm]
            else: ids, mask = ids[perm], mask[perm]
        out = model(img, mel, ids, mask)
        if zero_interaction:
            base = torch.cat([out["z1"], out["z2"], out["z3"]], -1)
            full = torch.cat([base, torch.zeros_like(out["z12"]), torch.zeros_like(out["z13"]),
                              torch.zeros_like(out["z23"])], -1)
            logits_full = model.head_full(full)
        else:
            logits_full = out["logits_full"]
        preds_f.append(logits_full.argmax(1).cpu())
        preds_b.append(out["logits_base"].argmax(1).cpu())
        ys.append(y)
        for k in store:
            store[k].append(out[k].cpu())
    y = torch.cat(ys); pf = torch.cat(preds_f); pb = torch.cat(preds_b)
    corr = ((pb != y) & (pf == y)).sum().item()
    regr = ((pb == y) & (pf != y)).sum().item()
    return {
        "acc_full": (pf == y).float().mean().item(),
        "acc_base": (pb == y).float().mean().item(),
        "gain_full_over_base": ((pf == y).float().mean() - (pb == y).float().mean()).item(),
        "net_correction": corr - regr,
        "reps": {k: torch.cat(v) for k, v in store.items()},
        "labels": y,
    }


def predictability(source: Tensor, target: Tensor, hidden=256, epochs=20, seed=0, device="cuda") -> float:
    torch.manual_seed(seed)
    s_mean, s_std = source.mean(0), source.std(0).clamp_min(1e-6)
    t_mean, t_std = target.mean(0), target.std(0).clamp_min(1e-6)
    src, tgt = (source - s_mean) / s_std, (target - t_mean) / t_std
    pred = nn.Sequential(nn.Linear(src.shape[1], hidden), nn.GELU(), nn.Linear(hidden, tgt.shape[1])).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    n = len(src)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=gen).split(1024):
            loss = F.mse_loss(pred(src[idx].to(device)), tgt[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    pred.eval()
    with torch.no_grad():
        mse = F.mse_loss(pred(src.to(device)), tgt.to(device)).item()
    return 1.0 - mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/bird_mml/bird-mml-subset.csv")
    parser.add_argument("--root", default="data/bird_mml")
    parser.add_argument("--out", default="results/birds/e2e")
    parser.add_argument("--arm", choices=["confu", "confu_u", "confu_u_s1"], default="confu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lambda_confu", type=float, default=1.0)
    parser.add_argument("--lambda_utility", type=float, default=0.2)
    parser.add_argument("--utility_margin", type=float, default=0.0)
    parser.add_argument("--lambda_shortcut", type=float, default=0.5)
    parser.add_argument("--shortcut_tau", type=float, default=0.5)
    parser.add_argument("--k_adv", type=int, default=1)
    parser.add_argument("--adv_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    split = dict(np.load("data/bird_mml/split.npz"))
    # split indices were built on the FULL subset csv order; remap is identity here
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds_tr = BirdSubsetDataset(args.csv, args.root, split["train"], tokenizer, img_tf)
    ds_va = BirdSubsetDataset(args.csv, args.root, split["val"], tokenizer, img_tf)
    ds_te = BirdSubsetDataset(args.csv, args.root, split["test"], tokenizer, img_tf)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, num_workers=args.num_workers)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, num_workers=args.num_workers)

    model = E2EConFu(args.embed_dim, args.fusion_hidden, n_classes=71).to(device)
    use_s1 = args.arm == "confu_u_s1"
    adversaries = nn.ModuleList()
    standardizers = []
    for _ in PAIRS:
        adversaries.append(AdversarialPredictor(args.embed_dim, 256))
        adversaries.append(AdversarialPredictor(args.embed_dim, 256))
        standardizers.append(EMAStandardizer(args.embed_dim).to(device))
    adversaries = adversaries.to(device)
    opt_gen = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_adv = torch.optim.AdamW(adversaries.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state, bad = -1.0, None, 0
    log_lines = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        meters = {"task": 0.0, "confu": 0.0, "utility": 0.0, "shortcut": 0.0, "n": 0}
        for img, wav, ids, mask, y in loader_tr:
            img, ids, mask, y = img.to(device), ids.to(device), mask.to(device), y.to(device)
            mel = mel_batch(wav, device)
            out = model(img, mel, ids, mask)
            loss = args.lambda_confu * confu_terms(out)
            ce_full = F.cross_entropy(out["logits_full"], y, reduction="none")
            ce_base = F.cross_entropy(out["logits_base"], y, reduction="none")
            if args.arm in ("confu_u", "confu_u_s1"):
                task = ce_full.mean() + 0.5 * ce_base.mean()
                utility = F.relu(ce_full - ce_base.detach() + args.utility_margin).mean()
                loss = loss + args.lambda_task * task + args.lambda_utility * utility
            else:
                # fair baseline: ConFu + fully trained task heads (no utility term)
                task = ce_full.mean() + 0.5 * ce_base.mean()
                loss = loss + args.lambda_task * task
            if use_s1:
                for k, (i, j) in enumerate(PAIRS):
                    key = f"z{i+1}{j+1}"
                    standardizers[k].update(out[key])
                    t_const = standardizers[k].standardize(out[key].detach())
                    for _ in range(args.k_adv):
                        loss_adv = (F.mse_loss(adversaries[2*k](out[f"z{i+1}"].detach()), t_const)
                                    + F.mse_loss(adversaries[2*k+1](out[f"z{j+1}"].detach()), t_const))
                        opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()
                with torch.no_grad():
                    preds = [(adversaries[2*k](out[f"z{i+1}"]), adversaries[2*k+1](out[f"z{j+1}"]))
                             for k, (i, j) in enumerate(PAIRS)]
                pen = torch.stack([
                    shortcut_hinge_loss(preds[k][0], preds[k][1],
                                        standardizers[k].standardize(out[f"z{PAIRS[k][0]+1}{PAIRS[k][1]+1}"]),
                                        args.shortcut_tau)[0]
                    for k in range(len(PAIRS))]).sum()
                warmup = min(1.0, (epoch + 1) / max(1.0, args.adv_warmup_fraction * args.epochs))
                loss = loss + (args.lambda_shortcut * warmup) * pen
                meters["shortcut"] += pen.item() * len(y)
            opt_gen.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_gen.step()
            meters["task"] += ce_full.mean().item() * len(y)
            meters["n"] += len(y)
        sched.step()
        va = evaluate(model, loader_va, device)
        line = (f"ep{epoch} val_full={va['acc_full']*100:.2f} val_base={va['acc_base']*100:.2f} "
                f"task={meters['task']/meters['n']:.3f} shortcut={meters['shortcut']/max(1,meters['n']):.3f} "
                f"({time.perf_counter()-t0:.0f}s)")
        print(line, flush=True)
        log_lines.append(line)
        if va["acc_full"] > best_val:
            best_val, bad = va["acc_full"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- evaluation battery on test ----
    te = evaluate(model, loader_te, device)
    result = {"arm": args.arm, "seed": args.seed, "best_val_full": best_val,
              "acc_full": te["acc_full"], "acc_base": te["acc_base"],
              "gain_full_over_base": te["gain_full_over_base"],
              "net_correction": te["net_correction"],
              "train_hours": round((time.perf_counter() - started) / 3600, 2),
              "parameters_inference": sum(p.numel() for p in model.parameters())}
    for i, name in enumerate(MODALITY):
        sh = evaluate(model, loader_te, device, shuffle_modality=i)
        result[f"shuffle_{name}_drop"] = te["acc_full"] - sh["acc_full"]
        del sh
    zero = evaluate(model, loader_te, device, zero_interaction=True)
    result["zero_interaction_drop"] = te["acc_full"] - zero["acc_full"]
    reps, labels = te["reps"], te["labels"]
    for (i, j) in PAIRS:
        key = f"z{i+1}{j+1}"
        result[f"R2_z{i+1}_to_{key}"] = predictability(reps[f"z{i+1}"], reps[key], device=device)
        result[f"R2_z{j+1}_to_{key}"] = predictability(reps[f"z{j+1}"], reps[key], device=device)
    path = os.path.join(args.out, f"{args.arm}_seed_{args.seed}.json")
    with open(path, "w") as f:
        json.dump({k: v for k, v in result.items() if not isinstance(v, dict)} |
                  {"log": log_lines, "config": vars(args)}, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
