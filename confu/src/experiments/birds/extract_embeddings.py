"""Extract frozen pretrained embeddings for the Bird-MML dataset (Zenodo 18920487).

Modalities per row of data/bird_mml/bird-mml.csv:
    image:  photo_file                 -> torchvision ResNet50 (ImageNet) penultimate 2048-d
    audio:  audio_file (wav clips)     -> torchaudio WAV2VEC2_BASE mean-pooled 768-d
    text:   combined_caption           -> all-MiniLM-L6-v2 (pretrained) 384-d

Frozen-encoder protocol (documented choice): complementarity is analyzed on
frozen unimodal features so that all fusion/probe comparisons share identical
inputs; no encoder training confounds.

Output: data/bird_mml/embeddings.npz with img/aud/txt [N, D], labels [N],
species names, row indices. Run modalities separately with --only if needed.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


def extract_images(df: pd.DataFrame, root: str, batch: int, device: str) -> np.ndarray:
    from PIL import Image
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1]).eval().to(device)
    preprocess = weights.transforms()
    out = np.zeros((len(df), 2048), dtype=np.float32)
    paths = df["photo_file"].tolist()
    for start in range(0, len(df), batch):
        tensors = []
        for p in paths[start:start + batch]:
            try:
                img = Image.open(os.path.join(root, p)).convert("RGB")
                tensors.append(preprocess(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(x).flatten(1).cpu().numpy()
        out[start:start + len(tensors)] = feats
        if start % (batch * 50) == 0:
            print(f"  image {start}/{len(df)}", flush=True)
    return out


def extract_audio_ast(df: pd.DataFrame, root: str, batch: int, device: str) -> np.ndarray:
    """Audio Spectrogram Transformer (AudioSet-pretrained) — correct audio domain."""
    import torchaudio
    from transformers import ASTFeatureExtractor, ASTModel

    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name)
    model = ASTModel.from_pretrained(name).eval().to(device)
    out = np.zeros((len(df), 768), dtype=np.float32)
    paths = df["audio_file"].tolist()
    for start in range(0, len(df), batch):
        waves = []
        for p in paths[start:start + batch]:
            try:
                wav, sr = torchaudio.load(os.path.join(root, p))
                wav = wav.mean(0)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav[: 16000 * 10]
            except Exception:
                wav = torch.zeros(16000)
            waves.append(wav.numpy())
        inputs = fe(waves, sampling_rate=16000, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            pooled = hidden.mean(1)  # CLS + patch tokens mean-pooled
        out[start:start + len(waves)] = pooled.cpu().numpy()
        if start % (batch * 20) == 0:
            print(f"  audio-ast {start}/{len(df)}", flush=True)
    return out


def extract_audio(df: pd.DataFrame, root: str, batch: int, device: str) -> np.ndarray:
    import torchaudio
    from torchaudio.pipelines import WAV2VEC2_BASE

    bundle = WAV2VEC2_BASE
    model = bundle.get_model().eval().to(device)
    out = np.zeros((len(df), 768), dtype=np.float32)
    paths = df["audio_file"].tolist()
    for start in range(0, len(df), batch):
        waves, lengths = [], []
        for p in paths[start:start + batch]:
            try:
                wav, sr = torchaudio.load(os.path.join(root, p))
                wav = wav.mean(0)
                if sr != bundle.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
                wav = wav[: bundle.sample_rate * 10]  # cap at 10 s
            except Exception:
                wav = torch.zeros(bundle.sample_rate)
            waves.append(wav)
            lengths.append(len(wav))
        maxlen = max(lengths)
        x = torch.zeros(len(waves), maxlen)
        for i, w in enumerate(waves):
            x[i, : len(w)] = w
        x = x.to(device)
        with torch.no_grad():
            feats, _ = model.extract_features(x)
            hidden = feats[-1]  # last transformer layer [B, T, 768]
            mask = torch.arange(hidden.shape[1], device=device)[None, :] < (
                torch.tensor(lengths, device=device)[:, None] // 320  # wav2vec hop
            ).clamp_min(1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        out[start:start + len(waves)] = pooled.cpu().numpy()
        if start % (batch * 20) == 0:
            print(f"  audio {start}/{len(df)}", flush=True)
    return out


def extract_text(df: pd.DataFrame, batch: int, device: str, column: str = "combined_caption") -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(device)
    out = np.zeros((len(df), 384), dtype=np.float32)
    texts = df[column].fillna("").tolist()
    for start in range(0, len(df), batch):
        enc = tokenizer(texts[start:start + batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        out[start:start + pooled.shape[0]] = pooled.cpu().numpy()
        if start % (batch * 100) == 0:
            print(f"  text {start}/{len(df)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/bird_mml/bird-mml.csv")
    parser.add_argument("--root", default="data/bird_mml")
    parser.add_argument("--out", default="data/bird_mml/embeddings.npz")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--only", choices=["image", "audio", "text"], default=None)
    parser.add_argument("--text_column", default="combined_caption")
    parser.add_argument("--audio_model", choices=["wav2vec2", "ast"], default="wav2vec2")
    parser.add_argument("--key", default=None, help="override output array key (e.g. txt_blip)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    print(f"{len(df)} rows; device={device}", flush=True)
    taxa = sorted(df["scientific_name"].unique())
    labels = df["scientific_name"].map({n: i for i, n in enumerate(taxa)}).to_numpy()

    store = {}
    if os.path.exists(args.out):
        store = dict(np.load(args.out, allow_pickle=True))
    started = time.time()
    if args.only in (None, "image"):
        store["img"] = extract_images(df, args.root, args.batch, device)
        np.savez(args.out, **store)
        print(f"image done {time.time()-started:.0f}s", flush=True)
    if args.only in (None, "audio"):
        if args.audio_model == "ast":
            store[args.key or "aud"] = extract_audio_ast(df, args.root, args.batch, device)
        else:
            store[args.key or "aud"] = extract_audio(df, args.root, args.batch, device)
        store.setdefault("labels", labels)
        store.setdefault("species", np.array(taxa))
        np.savez(args.out, **store)
        print(f"audio done {time.time()-started:.0f}s", flush=True)
    if args.only in (None, "text"):
        store[args.key or "txt"] = extract_text(df, args.batch, device, args.text_column)
        np.savez(args.out, **store)
        print(f"text done {time.time()-started:.0f}s", flush=True)
    store["labels"] = labels
    store["species"] = np.array(taxa)
    np.savez(args.out, **store)
    print(f"saved {args.out}: " + ", ".join(f"{k}{v.shape}" for k, v in store.items() if hasattr(v, 'shape')))


if __name__ == "__main__":
    main()
