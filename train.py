#!/usr/bin/env python3
"""
immuneKG — Training Entry Point (v2)

Pipeline stages:
    1. Load and preprocess knowledge graph data
    2. Build triples & encode five-dimensional disease features
    3. Train ComplEx KG embedding model
    4. Build PyG heterogeneous + homogeneous projection graph
    5. Train HeteroPNA-Attn GNN (PNA aggregation + heterogeneous attention)
    6. Train three-source feature fusion network (KG + GNN + disease features)
    7. Novelty-enhanced target scoring and ranking

Usage:
    python train.py
    python train.py --config configs/custom.yaml
    python train.py --force-retrain
    python train.py --no-gnn
    python train.py --no-novelty
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_device, Timer,
    print_banner, print_stage, print_success, print_warning,
    print_info, print_stat, print_error
)
from src.data_loader import KGDataLoader
from src.feature_encoder import DiseaseFeatureEncoder
from src.trainer import ImmKGTrainer
from src.scorer import TargetScorer, PredictionReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='immuneKG v2 — HeteroPNA-Attn + novelty-enhanced target discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py
    python train.py --config configs/custom.yaml
    python train.py --force-retrain
    python train.py --no-gnn
    python train.py --no-novelty
        """
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining of all models (ignore checkpoints)')
    parser.add_argument('--force-rescore', action='store_true',
                        help='Force re-scoring only (skip retraining)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='KG score weight (default: 0.5)')
    parser.add_argument('--no-gnn', action='store_true',
                        help='Skip GNN stage (fall back to ComplEx + FFN only)')
    parser.add_argument('--no-novelty', action='store_true',
                        help='Skip novelty scoring')
    return parser.parse_args()


def main():
    args = parse_args()

    print_banner("immuneKG v2 Training System")
    print_info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Config: {args.config}")
    print_info(f"GNN module: {'disabled' if args.no_gnn else 'enabled (HeteroPNA-Attn)'}")
    print_info(f"Novelty scoring: {'disabled' if args.no_novelty else 'enabled'}")

    config = load_config(args.config)
    device = setup_device(config)

    work_dir = config['output']['work_dir']
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    timer = Timer()
    timer.start('total')

    use_gnn = not args.no_gnn
    use_novelty = not args.no_novelty
    total_stages = 7 if use_gnn else 5
    current_stage = 0

    # Stage 1: Load KG data
    current_stage += 1
    print_stage(current_stage, total_stages, "Load and preprocess knowledge graph data")
    timer.start('stage_data_loading')
    data_loader = KGDataLoader(config, work_dir)
    data = data_loader.load_and_process()
    duration = timer.stop('stage_data_loading')
    print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Stage 2: Build triples & encode features
    current_stage += 1
    print_stage(current_stage, total_stages, "Build triples & encode five-dimensional disease features")
    timer.start('stage_triples_features')
    print_info("-- 2a. Build PyKEEN TriplesFactory --")
    triples_factory = data_loader.build_triples_factory(data)
    print_info("-- 2b. Encode five-dimensional disease features --")
    feature_encoder = DiseaseFeatureEncoder(config, work_dir)
    feature_result = feature_encoder.encode_all_features(data)
    duration = timer.stop('stage_triples_features')
    print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Stage 3: Train ComplEx KG embedding
    current_stage += 1
    print_stage(current_stage, total_stages, "Train KG embedding model (ComplEx)")
    timer.start('stage_kg_training')
    trainer = ImmKGTrainer(config, device, work_dir)
    kg_result = trainer.train_kg_embeddings(
        triples_factory, force_retrain=args.force_retrain, data=data
    )
    duration = timer.stop('stage_kg_training')
    print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Stages 4 & 5: GNN (optional)
    gnn_embeddings = None
    graph_result = None
    degree_map = None

    if use_gnn:
        current_stage += 1
        print_stage(current_stage, total_stages, "Build PyG heterogeneous graph + homogeneous projection")
        timer.start('stage_graph_building')
        try:
            from src.graph_builder import HeteroGraphBuilder, check_pyg_available
            if check_pyg_available():
                graph_builder = HeteroGraphBuilder(config, work_dir)
                graph_result = graph_builder.build_graphs(data, kg_result)
                if graph_result is not None:
                    degree_map = graph_builder.compute_target_degrees(data, graph_result)
                    print_stat("Degree map entities", f"{len(degree_map):,}")
            else:
                print_warning("PyG not available, skipping GNN stage")
                use_gnn = False
        except ImportError as e:
            print_warning(f"GNN dependency import failed: {e}")
            use_gnn = False
        duration = timer.stop('stage_graph_building')
        print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

        if use_gnn and graph_result is not None:
            current_stage += 1
            print_stage(current_stage, total_stages,
                        "Train HeteroPNA-Attn GNN (PNA aggregation + heterogeneous attention)")
            timer.start('stage_gnn_training')
            try:
                from src.gnn_module import GNNTrainer
                gnn_trainer = GNNTrainer(config, device, work_dir)
                gnn_model, gnn_embeddings = gnn_trainer.train_gnn(
                    graph_result, kg_result, force_retrain=args.force_retrain
                )
                print_stat("GNN embedding shape", gnn_embeddings.shape)
            except Exception as e:
                print_warning(f"GNN training failed: {e}")
                gnn_embeddings = None
                import traceback
                traceback.print_exc()
            duration = timer.stop('stage_gnn_training')
            print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Stage N-1: Train fusion network
    current_stage += 1
    fusion_mode = "three-source (KG+GNN+features)" if gnn_embeddings is not None else "two-source (KG+features)"
    print_stage(current_stage, total_stages, f"Train feature fusion network ({fusion_mode})")
    timer.start('stage_fusion_training')
    fusion_model = trainer.train_fusion_network(
        kg_result, feature_result,
        force_retrain=args.force_retrain,
        gnn_embeddings=gnn_embeddings
    )
    duration = timer.stop('stage_fusion_training')
    print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Stage N: Scoring
    current_stage += 1
    scoring_desc = "Target scoring and ranking"
    if use_novelty and degree_map:
        scoring_desc += " + novelty penalty"
    print_stage(current_stage, total_stages, scoring_desc)
    timer.start('stage_scoring')

    target_entities = data_loader.get_target_entities(data)
    target_diseases = data_loader.get_prediction_disease_entities(data)

    print_info("=== Pre-scoring diagnostics ===")
    print_stat("Candidate targets", f"{len(target_entities):,}")
    entity_to_id = kg_result.training.entity_to_id
    disease_in_model = sum(1 for d in data['disease_entities'] if d in entity_to_id)
    targets_in_model = sum(1 for t in target_entities if t in entity_to_id)
    print_stat("Diseases in model", f"{disease_in_model}/{len(data['disease_entities'])}")
    print_stat("Targets in model", f"{targets_in_model}/{len(target_entities):,}")
    print_stat("Scoring mode", fusion_mode + (" + novelty" if use_novelty and degree_map else ""))

    scorer = TargetScorer(config, device, work_dir)
    scorer.alpha = args.alpha

    scores_df = scorer.score_targets(
        kg_result=kg_result,
        data=data,
        feature_result=feature_result,
        fusion_model=fusion_model,
        target_diseases=target_diseases,
        target_entities=target_entities,
        gnn_embeddings=gnn_embeddings,
        degree_map=degree_map if (use_novelty and degree_map) else None,
        graph_result=graph_result,
    )

    if len(scores_df) > 0:
        print_info("Generating per-disease prediction reports...")
        PredictionReportGenerator.generate_per_disease_report(
            scores_df=scores_df,
            target_diseases=target_diseases,
            entity_info=data['entity_info'],
            output_dir=work_dir,
            top_k=config['prediction']['top_k']
        )

    duration = timer.stop('stage_scoring')
    print_success(f"Stage {current_stage} complete ({timer.format_duration(duration)})")

    # Summary
    total_duration = timer.stop('total')
    timer.save(str(Path(work_dir) / config['output']['filenames']['timing']))

    print_banner("Training and Scoring Complete")
    print(timer.summary())

    print_info("\nOutput files:")
    print_stat("KG model weights", Path(work_dir) / config['output']['filenames']['kg_weights'])
    print_stat("Fusion model weights", Path(work_dir) / config['output']['filenames']['fusion_model'])
    if gnn_embeddings is not None:
        print_stat("GNN model weights", Path(work_dir) / 'gnn_model.pth')
        print_stat("GNN embeddings", Path(work_dir) / 'gnn_embeddings.pkl')
    print_stat("Training log", Path(work_dir) / config['output']['filenames']['training_log'])

    if len(scores_df) > 0:
        print_info(f"\nFinal result: {len(scores_df)} targets scored")

    return scores_df


if __name__ == '__main__':
    main()
