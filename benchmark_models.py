#!/usr/bin/env python3
"""
 （immuneKG ）

 :
 python benchmark_models.py --epochs 100
 python benchmark_models.py --epochs 100 --models TransE,TransR,RESCAL,ComplEx,DistMult,ConvE,ConvKB,RGCN
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pykeen.pipeline import pipeline

from src.data_loader import KGDataLoader
from src.utils import (
    load_config, setup_device,
    print_banner, print_info, print_stat, print_success, print_warning
)


MODEL_CATEGORIES = {
    "TransE": "Distance-based",
    "TransR": "Distance-based",
    "RESCAL": "Semantic matching",
    "ComplEx": "Semantic matching",
    "DistMult": "Semantic matching",
    "ConvE": "Neural network",
    "ConvKB": "Neural network",
    "RGCN": "Neural network",
}

DEFAULT_MODELS = list(MODEL_CATEGORIES.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="immuneKG ")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=" ， TransE,TransR,RESCAL,ComplEx,DistMult,ConvE,ConvKB,RGCN",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="model_benchmark")
    return parser.parse_args()


def _get_metric(metric_results, keys: List[str]) -> float:
    for key in keys:
        try:
            value = metric_results.get_metric(key)
            if value is not None:
                return float(value)
        except Exception:
            continue
    return float("nan")


def _build_rank_evaluator_metrics(metric_names: List[str]):
    converted_metrics = []
    converted_kwargs = []
    for name in metric_names:
        s = str(name).strip().lower()
        m = re.fullmatch(r"hits_at_(\d+)", s)
        if m:
            converted_metrics.append("hits@k")
            converted_kwargs.append({"k": int(m.group(1))})
        else:
            converted_metrics.append(name)
            converted_kwargs.append(None)
    return converted_metrics, converted_kwargs


def _model_kwargs(model_name: str, embedding_dim: int) -> dict:
    # ， ， 100 epoch 
    kwargs = {}
    if model_name in {"TransE", "TransR", "RESCAL", "ComplEx", "DistMult", "ConvKB"}:
        kwargs["embedding_dim"] = embedding_dim
    if model_name == "ConvE":
        kwargs.update(
            embedding_dim=embedding_dim,
            input_dropout=0.2,
            feature_map_dropout=0.2,
            output_dropout=0.3,
        )
    if model_name == "RGCN":
        kwargs.update(
            embedding_dim=embedding_dim,
            num_layers=2,
        )
    return kwargs


def _to_markdown(df: pd.DataFrame) -> str:
    cols = ["Category", "Model", "MRR", "N = 1", "N = 3", "N = 10", "N = 100"]
    view = df[cols].copy()
    for c in cols[2:]:
        view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    return view.to_markdown(index=False)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = setup_device(config)

    print_banner("immuneKG ")
    print_info(f" : {args.models}")
    print_info(f" : {args.epochs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = KGDataLoader(config, config["output"]["work_dir"])
    data = loader.load_and_process()
    _ = loader.build_triples_factory(data)
    train_tf, valid_tf, test_tf = loader.build_split_factories(data)
    eval_cfg = config.get("evaluation", {})
    eval_metrics = eval_cfg.get(
        "metrics",
        ["mean_reciprocal_rank", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "hits_at_100"],
    )
    rank_metrics, rank_metrics_kwargs = _build_rank_evaluator_metrics(eval_metrics)

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid_models = [m for m in selected_models if m not in MODEL_CATEGORIES]
    if invalid_models:
        raise ValueError(f" : {invalid_models}")

    rows = []
    for model_name in selected_models:
        print_info(f" : {model_name}")
        try:
            result = pipeline(
                model=model_name,
                training=train_tf,
                validation=valid_tf,
                testing=test_tf,
                model_kwargs=_model_kwargs(model_name, args.embedding_dim),
                training_kwargs=dict(
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                ),
                optimizer="Adam",
                optimizer_kwargs=dict(lr=args.learning_rate),
                evaluator_kwargs=dict(
                    filtered=True,
                    metrics=rank_metrics,
                    metrics_kwargs=rank_metrics_kwargs,
                ),
                evaluation_kwargs=dict(batch_size=config["evaluation"]["batch_size"]),
                random_seed=config["training"]["kg"]["random_seed"],
                device=str(device),
            )
            mr = result.metric_results
            row = {
                "Category": MODEL_CATEGORIES[model_name],
                "Model": model_name,
                "MRR": _get_metric(mr, ["mean_reciprocal_rank", "both.realistic.mean_reciprocal_rank"]),
                "N = 1": _get_metric(mr, ["hits_at_1", "both.realistic.hits_at_1"]),
                "N = 3": _get_metric(mr, ["hits_at_3", "both.realistic.hits_at_3"]),
                "N = 10": _get_metric(mr, ["hits_at_10", "both.realistic.hits_at_10"]),
                "N = 100": _get_metric(mr, ["hits_at_100", "both.realistic.hits_at_100"]),
            }
            rows.append(row)
            print_success(f"{model_name}  : MRR={row['MRR']:.4f}, Hits@10={row['N = 10']:.4f}")
        except Exception as e:
            print_warning(f"{model_name}  ， : {e}")
            rows.append({
                "Category": MODEL_CATEGORIES[model_name],
                "Model": model_name,
                "MRR": float("nan"),
                "N = 1": float("nan"),
                "N = 3": float("nan"),
                "N = 10": float("nan"),
                "N = 100": float("nan"),
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise RuntimeError(" ")

    category_order = ["Distance-based", "Semantic matching", "Neural network"]
    result_df["__cat_order"] = result_df["Category"].map({k: i for i, k in enumerate(category_order)})
    result_df = result_df.sort_values(["__cat_order", "MRR"], ascending=[True, False]).drop(columns=["__cat_order"])
    result_df = result_df.reset_index(drop=True)

    csv_path = output_dir / f"{args.output_prefix}_epochs{args.epochs}.csv"
    md_path = output_dir / f"{args.output_prefix}_epochs{args.epochs}.md"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_to_markdown(result_df), encoding="utf-8")

    print_success(f"  CSV: {csv_path}")
    print_success(f"  Markdown: {md_path}")
    print_info(" : Category | Model | MRR | N=1 | N=3 | N=10 | N=100")


if __name__ == "__main__":
    main()
