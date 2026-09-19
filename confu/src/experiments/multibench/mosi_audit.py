"""Frozen-feature dependency and conditional-utility audit for ConFu runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.affect import AffectDataModule
from src.modules.models.confu import ConFu


PAIR_NAMES = {
    "vision_audio": ("r1", "r2", "r12", "r12_vision_shuffle", "r12_audio_shuffle"),
    "vision_text": ("r1", "r3", "r13", "r13_vision_shuffle", "r13_text_shuffle"),
    "audio_text": ("r2", "r3", "r23", "r23_audio_shuffle", "r23_text_shuffle"),
}


def effective_rank(features):
    singular_values = torch.linalg.svdvals(features - features.mean(0, keepdim=True))
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def extract_split(model, loader, device, seed, with_single_shuffles=False):
    values = {key: [] for key in ("r1", "r2", "r3", "r12", "r13", "r23", "labels")}
    if with_single_shuffles:
        values.update({key: [] for key in ("r12_vision_shuffle", "r12_audio_shuffle", "r13_vision_shuffle", "r13_text_shuffle", "r23_audio_shuffle", "r23_text_shuffle")})
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for image, audio, text, labels in loader:
            image, audio, text = image.to(device), audio.to(device), text.to(device)
            r12, r13, r23, r1, r2, r3 = model(image, audio, text)
            for key, value in zip(("r12", "r13", "r23", "r1", "r2", "r3"), (r12, r13, r23, r1, r2, r3)):
                values[key].append(value.cpu())
            values["labels"].append(labels.cpu())
            if with_single_shuffles:
                def shuffled(value):
                    return value[torch.randperm(value.size(0), generator=generator).to(device)]
                values["r12_vision_shuffle"].append(model(shuffled(image), audio, text)[0].cpu())
                values["r12_audio_shuffle"].append(model(image, shuffled(audio), text)[0].cpu())
                values["r13_vision_shuffle"].append(model(shuffled(image), audio, text)[1].cpu())
                values["r13_text_shuffle"].append(model(image, audio, shuffled(text))[1].cpu())
                values["r23_audio_shuffle"].append(model(image, shuffled(audio), text)[2].cpu())
                values["r23_text_shuffle"].append(model(image, audio, shuffled(text))[2].cpu())
    return {key: torch.cat(value).numpy() for key, value in values.items()}


def fit_linear(train_x, train_y, valid_x, valid_y):
    best_c, best_score = None, -1.0
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=300, random_state=0))
        classifier.fit(train_x, train_y)
        score = accuracy_score(valid_y, classifier.predict(valid_x))
        if score > best_score:
            best_c, best_score = c, score
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, solver="lbfgs", max_iter=300, random_state=0))
    classifier.fit(np.concatenate((train_x, valid_x)), np.concatenate((train_y, valid_y)))
    return classifier, best_c, best_score


def fit_nonlinear(train_x, train_y, valid_x, valid_y):
    # Fixed probe: validation is included only to match the final linear-probe protocol.
    classifier = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4, batch_size=128, early_stopping=True, max_iter=300, random_state=0))
    classifier.fit(np.concatenate((train_x, valid_x)), np.concatenate((train_y, valid_y)))
    return classifier


def score(classifier, features, labels):
    return float(accuracy_score(labels, classifier.predict(features)))


def pair_audit(name, train, valid, test, seed, run_nonlinear=True):
    left, right, interaction, left_shuffled, right_shuffled = PAIR_NAMES[name]
    lower_train = np.concatenate((train[left], train[right]), axis=1)
    lower_valid = np.concatenate((valid[left], valid[right]), axis=1)
    lower_test = np.concatenate((test[left], test[right]), axis=1)
    full_train = np.concatenate((lower_train, train[interaction]), axis=1)
    full_valid = np.concatenate((lower_valid, valid[interaction]), axis=1)
    full_test = np.concatenate((lower_test, test[interaction]), axis=1)
    labels_train, labels_valid, labels_test = train["labels"], valid["labels"], test["labels"]
    lower_linear, lower_c, lower_valid_score = fit_linear(lower_train, labels_train, lower_valid, labels_valid)
    full_linear, full_c, full_valid_score = fit_linear(full_train, labels_train, full_valid, labels_valid)
    interaction_linear, interaction_c, _ = fit_linear(train[interaction], labels_train, valid[interaction], labels_valid)
    nonlinear_lower = fit_nonlinear(lower_train, labels_train, lower_valid, labels_valid) if run_nonlinear else None
    nonlinear_full = fit_nonlinear(full_train, labels_train, full_valid, labels_valid) if run_nonlinear else None
    permutation = np.random.default_rng(seed).permutation(len(labels_test))
    clean = score(full_linear, full_test, labels_test)
    raw_left = np.concatenate((lower_test, test[left_shuffled]), axis=1)
    raw_right = np.concatenate((lower_test, test[right_shuffled]), axis=1)
    zero = np.concatenate((lower_test, np.zeros_like(test[interaction])), axis=1)
    shuffled = np.concatenate((lower_test, test[interaction][permutation]), axis=1)
    predictor = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr"))
    predictor.fit(np.concatenate((lower_train, lower_valid)), np.concatenate((train[interaction], valid[interaction])))
    predicted = predictor.predict(lower_test)
    cosine = np.sum(predicted * test[interaction], axis=1) / (np.linalg.norm(predicted, axis=1) * np.linalg.norm(test[interaction], axis=1) + 1e-12)
    return {
        "linear": {
            "interaction": score(interaction_linear, test[interaction], labels_test),
            "lower": score(lower_linear, lower_test, labels_test),
            "lower_plus_interaction": clean,
            "conditional_gain": clean - score(lower_linear, lower_test, labels_test),
            "selected_c": {"lower": lower_c, "full": full_c, "interaction": interaction_c},
            "validation_accuracy": {"lower": lower_valid_score, "full": full_valid_score},
        },
        "nonlinear": ({
            "lower": score(nonlinear_lower, lower_test, labels_test),
            "lower_plus_interaction": score(nonlinear_full, full_test, labels_test),
            "conditional_gain": score(nonlinear_full, full_test, labels_test) - score(nonlinear_lower, lower_test, labels_test),
        } if run_nonlinear else {"status": "skipped"}),
        "intervention_drop": {
            "zero_interaction": clean - score(full_linear, zero, labels_test),
            "shuffle_interaction": clean - score(full_linear, shuffled, labels_test),
            f"shuffle_{left[1:]}": clean - score(full_linear, raw_left, labels_test),
            f"shuffle_{right[1:]}": clean - score(full_linear, raw_right, labels_test),
        },
        "representation": {
            "mean_feature_variance": float(np.var(test[interaction], axis=0).mean()),
            "effective_rank": effective_rank(torch.from_numpy(test[interaction])),
            "predictability_from_lower_r2": float(r2_score(test[interaction], predicted, multioutput="uniform_average")),
            "predictability_from_lower_cosine": float(cosine.mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", default="mosi")
    parser.add_argument("--pickle-name", default="mosi_data.pkl")
    parser.add_argument("--skip-nonlinear", action="store_true")
    parser.add_argument("--model-type", choices=("confu", "confu_plus"), default="confu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.model_type == "confu_plus":
        from src.experiments.multibench.urfunny_complementarity import ConFuPlus
        model = ConFuPlus.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
    else:
        model = ConFu.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
    dm = AffectDataModule(batch_size=args.batch_size, num_workers=args.num_workers, pickle_name=args.pickle_name, dataset_name=args.dataset)
    dm.setup()
    train = extract_split(model, dm.train_dataloader(), device, args.seed)
    valid = extract_split(model, dm.val_dataloader(), device, args.seed + 1)
    test = extract_split(model, dm.test_dataloader(), device, args.seed + 2, with_single_shuffles=True)
    all_lower = np.concatenate((test["r1"], test["r2"], test["r3"]), axis=1)
    all_classifier, all_c, all_valid = fit_linear(
        np.concatenate((train["r1"], train["r2"], train["r3"]), axis=1), train["labels"],
        np.concatenate((valid["r1"], valid["r2"], valid["r3"]), axis=1), valid["labels"],
    )
    result = {
        "protocol": {"dataset": args.dataset, "modalities": ["vision", "audio", "text"], "split_sizes": {"train": len(train["labels"]), "valid": len(valid["labels"]), "test": len(test["labels"])}, "checkpoint": str(args.checkpoint), "seed": args.seed},
        "unimodal_and_all_linear": {
            **{key: score(fit_linear(train[key], train["labels"], valid[key], valid["labels"])[0], test[key], test["labels"]) for key in ("r1", "r2", "r3")},
            "r1_r2_r3": score(all_classifier, all_lower, test["labels"]),
            "all_selected_c": all_c,
            "all_validation_accuracy": all_valid,
        },
        "pairs": {name: pair_audit(name, train, valid, test, args.seed + index, not args.skip_nonlinear) for index, name in enumerate(PAIR_NAMES)},
        "higher_order": {"r123": "not implemented by original ConFu baseline; no higher-order claim is evaluated"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
