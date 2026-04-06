"""
PromptCraft experiments on MovieLens-100K using the project's native training stack.

This runner keeps preprocessing, dataloading, training, and evaluation consistent
with existing baselines/hybrid experiments and only changes SASRec item
initialization via prompt-based LLM embeddings.
"""

import argparse
import json
import os
import pickle
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import get_dataloaders
from src.data.preprocess_ml100k import ML100KPreprocessor
from src.data.promptcraft import (
    PROMPT_STYLES,
    load_or_generate_raw_embeddings,
    pca_compress_embeddings,
)
from src.models.sasrec import SASRec
from src.train.trainer import Trainer


BASELINE_STYLE = "baseline_sasrec"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_ml100k_processed(args) -> None:
    """Run the same ML-100K preprocessing used by the project if missing."""
    preprocessor = ML100KPreprocessor(
        raw_data_dir=args.raw_data_dir,
        min_rating=args.min_rating,
        min_seq_len=args.min_seq_len,
    )

    # PromptCraft needs raw metadata (u.item) even if processed data already exists.
    item_meta_path = os.path.join(args.raw_data_dir, "ml-100k", "u.item")
    if not os.path.exists(item_meta_path):
        print("Raw ML-100K metadata not found. Downloading/extracting raw files...")
        preprocessor.download()

    if os.path.exists(args.data_path):
        print(f"Using existing processed data: {args.data_path}")
        return

    print("Processed ML-100K data not found. Running project preprocessor...")
    preprocessor.preprocess(args.data_path)


def create_sasrec(num_items: int, args) -> SASRec:
    return SASRec(
        num_items=num_items,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )


def initialize_promptcraft_weights(model: SASRec, processed_data: dict, style: str, args) -> Dict[str, object]:
    raw_embeddings, raw_path = load_or_generate_raw_embeddings(
        processed_data=processed_data,
        raw_data_dir=args.raw_data_dir,
        style=style,
        cache_dir=args.embedding_cache_dir,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        use_fp16=not args.embedding_fp32,
        force_recompute=args.force_reembed,
    )

    compressed, explained_var = pca_compress_embeddings(
        raw_embeddings=raw_embeddings,
        out_dim=args.d_model,
        random_state=args.seed,
    )

    with torch.no_grad():
        model.item_emb.weight.copy_(torch.from_numpy(compressed))
        model.item_emb.weight[0].zero_()

    return {
        "embedding_mode": "promptcraft",
        "style": style,
        "raw_embeddings_path": raw_path,
        "raw_shape": list(raw_embeddings.shape),
        "compressed_shape": list(compressed.shape),
        "pca_explained_variance": explained_var,
    }


def run_single_style(
    style: str,
    processed_data: dict,
    train_loader,
    val_loader,
    test_loader,
    device,
    args,
) -> Dict[str, object]:
    num_items = processed_data["config"]["num_items"]
    model = create_sasrec(num_items=num_items, args=args)

    if style == BASELINE_STYLE:
        init_info = {
            "embedding_mode": "random_init",
            "style": style,
        }
    else:
        print(f"Preparing PromptCraft embeddings for style={style} ...")
        init_info = initialize_promptcraft_weights(
            model=model,
            processed_data=processed_data,
            style=style,
            args=args,
        )
        print(
            f"  PCA explained variance ({style}): "
            f"{init_info['pca_explained_variance']:.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    style_slug = style.lower()
    exp_name = f"promptcraft_{style_slug}_{timestamp}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config_to_save = vars(args).copy()
    config_to_save.update(
        {
            "model": "sasrec",
            "promptcraft_style": style,
            "num_users": processed_data["config"]["num_users"],
            "num_items": num_items,
            "dataset": processed_data["config"].get("dataset", "ml-100k"),
            "initialization": init_info,
        }
    )

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=None,
        edge_weight=None,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=exp_dir,
    )

    history = trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every,
    )

    history_json = {
        "train_loss": history["train_loss"],
        "val_metrics": history["val_metrics"],
        "best_epoch": history["best_epoch"],
        "best_val_metric": history["best_val_metric"],
    }
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history_json, f, indent=2)

    test_metrics, grouped_metrics = trainer.test(use_best_model=True)

    results = {
        "style": style,
        "initialization": init_info,
        "test_metrics": test_metrics,
        "grouped_metrics": grouped_metrics,
        "best_epoch": history["best_epoch"],
        "best_val_metric": history["best_val_metric"],
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted style={style}. Results saved to: {exp_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "style": style,
        "exp_dir": exp_dir,
        "results": results,
    }


def summarize_and_save(all_outputs: List[Dict[str, object]], args) -> None:
    summary_rows = []
    aggregate_json = {}

    for out in all_outputs:
        style = out["style"]
        exp_dir = out["exp_dir"]
        result = out["results"]
        metrics = result["test_metrics"]

        row = {
            "style": style,
            "exp_dir": exp_dir,
            "HR@10": metrics.get("HR@10", 0.0),
            "NDCG@10": metrics.get("NDCG@10", 0.0),
            "MRR@10": metrics.get("MRR@10", 0.0),
            "HR@20": metrics.get("HR@20", 0.0),
            "NDCG@20": metrics.get("NDCG@20", 0.0),
            "MRR@20": metrics.get("MRR@20", 0.0),
            "best_epoch": result.get("best_epoch", 0),
            "best_val_metric": result.get("best_val_metric", 0.0),
        }
        summary_rows.append(row)
        aggregate_json[style] = result

    summary_rows = sorted(summary_rows, key=lambda x: x["NDCG@10"], reverse=True)

    summary_dir = os.path.join(args.save_dir, "promptcraft")
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(summary_dir, f"summary_{timestamp}.csv")
    json_path = os.path.join(summary_dir, f"summary_{timestamp}.json")

    # Keep dependencies minimal by writing CSV manually.
    csv_cols = list(summary_rows[0].keys()) if summary_rows else []
    with open(csv_path, "w") as f:
        if csv_cols:
            f.write(",".join(csv_cols) + "\n")
            for row in summary_rows:
                values = [str(row[col]) for col in csv_cols]
                f.write(",".join(values) + "\n")

    with open(json_path, "w") as f:
        json.dump(aggregate_json, f, indent=2)

    print("\n" + "=" * 78)
    print("PROMPTCRAFT SUMMARY (sorted by NDCG@10)")
    print("=" * 78)
    print(f"{'Style':<22} {'HR@10':>8} {'NDCG@10':>10} {'MRR@10':>8} {'BestEp':>8}")
    print("-" * 78)
    for row in summary_rows:
        print(
            f"{row['style']:<22} "
            f"{row['HR@10']:>8.4f} "
            f"{row['NDCG@10']:>10.4f} "
            f"{row['MRR@10']:>8.4f} "
            f"{row['best_epoch']:>8}"
        )
    print("-" * 78)
    print(f"CSV summary:  {csv_path}")
    print(f"JSON summary: {json_path}")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PromptCraft SASRec experiments on MovieLens-100K"
    )

    # Data and preprocessing
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ml-100k/processed/sequences.pkl",
        help="Path to processed ML-100K data (project format)",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data/ml-100k/raw",
        help="Directory for raw ML-100K files",
    )
    parser.add_argument(
        "--min_rating",
        type=int,
        default=4,
        help="Implicit positive threshold used by preprocessor",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=5,
        help="Minimum sequence length used by preprocessor",
    )

    # PromptCraft setup
    parser.add_argument(
        "--styles",
        nargs="+",
        default=PROMPT_STYLES,
        choices=PROMPT_STYLES,
        help="PromptCraft styles to run",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip random-initialized SASRec baseline",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model name for FlagEmbedding",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default="data/ml-100k/embeddings",
        help="Where raw per-style embeddings are cached",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=256,
        help="Batch size used when encoding item texts",
    )
    parser.add_argument(
        "--embedding_max_length",
        type=int,
        default=128,
        help="Max token length for encoder input",
    )
    parser.add_argument(
        "--embedding_fp32",
        action="store_true",
        help="Use fp32 for embeddings (default uses fp16 when available)",
    )
    parser.add_argument(
        "--force_reembed",
        action="store_true",
        help="Force recomputation even if cached style embeddings exist",
    )

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=50)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 78)
    print("PROMPTCRAFT ML-100K EXPERIMENTS")
    print("=" * 78)
    print(f"Device: {device}")
    print(f"Styles: {', '.join(args.styles)}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience} | d_model: {args.d_model}")
    print("=" * 78)

    ensure_ml100k_processed(args)

    with open(args.data_path, "rb") as f:
        processed_data = pickle.load(f)

    print(
        f"Loaded data: users={processed_data['config']['num_users']:,}, "
        f"items={processed_data['config']['num_items']:,}"
    )

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )

    print(
        f"Dataloaders ready: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )

    styles_to_run = list(args.styles)
    if not args.skip_baseline:
        styles_to_run = [BASELINE_STYLE] + styles_to_run

    all_outputs: List[Dict[str, object]] = []
    for style in styles_to_run:
        print("\n" + "#" * 78)
        print(f"Running style: {style}")
        print("#" * 78)
        try:
            out = run_single_style(
                style=style,
                processed_data=processed_data,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                args=args,
            )
            all_outputs.append(out)
        except Exception as exc:
            print(f"Error while running style={style}: {exc}")
            traceback.print_exc()
            continue

    if not all_outputs:
        raise RuntimeError("No PromptCraft runs completed successfully.")

    summarize_and_save(all_outputs=all_outputs, args=args)


if __name__ == "__main__":
    main()
