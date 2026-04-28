"""
immuneKG Training Module

Two training phases:
    Phase A: KG embedding training (PyKEEN ComplEx)
    Phase B: Feature fusion network training (FFN)
"""

import time
import pickle
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from .model import (
    FeatureFusionNetwork, FusionDataset,
    MultiSourceFusionNetwork, MultiSourceFusionDataset,
    save_fusion_model, load_fusion_model
)
from .utils import (
    Timer, print_banner, print_stage, print_success, print_warning,
    print_info, print_stat, print_error, checkpoint_exists,
    save_checkpoint, load_checkpoint
)

def _build_rank_evaluator_metrics(metric_names):
    """
    Convert hits_at_10-style metric names to PyKEEN RankBasedEvaluator format.
    """
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


class ImmKGTrainer:
    """
    immuneKG Trainer
    
    Executes two training phases:
        Phase A: PyKEEN ComplEx KG embedding training
        Phase B: Feature fusion network (FFN) training
    """
    
    def __init__(self, config: dict, device: torch.device, work_dir: str):
        """
        Args:
            config: full configuration dict
            device: compute device (GPU/CPU)
            work_dir: working directory
        """
        self.config = config
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file paths
        filenames = config['output']['filenames']
        self.kg_model_path = self.work_dir / filenames['kg_model']
        self.kg_weights_path = self.work_dir / filenames['kg_weights']
        self.fusion_model_path = self.work_dir / filenames['fusion_model']
        self.fusion_config_path = self.work_dir / filenames['fusion_config']
        self.training_log_path = self.work_dir / filenames['training_log']
        
        # Training log
        self.training_log = {
            'kg_training': {},
            'fusion_training': {}
        }
    
    # ================================================================
    # Phase A: KG embedding training
    # ================================================================
    
    def train_kg_embeddings(self, triples_factory: TriplesFactory,
                             force_retrain: bool = False,
                             data: dict = None):
        """
        Train KG embedding model using PyKEEN (ComplEx)
        
        Supports checkpointing: loads existing model if available.
        Supports pre-split and random-split modes.
        
        Args:
            triples_factory: PyKEEN TriplesFactory (full graph)
            force_retrain: force retraining (ignore checkpoints)
            data: preprocessed KG data dict (contains split_info)
            
        Returns:
            PyKEEN pipeline result object
        """
        # === Checkpoint check ===
        if self.kg_model_path.exists() and not force_retrain:
            print_success("Found trained KG model checkpoint, loading...")
            result = load_checkpoint(self.kg_model_path, "KG embedding model")
            self._print_kg_result(result)
            return result
        
        if force_retrain and self.kg_model_path.exists():
            print_warning("Force retraining KG embedding model...")
            self.kg_model_path.unlink()
            if self.kg_weights_path.exists():
                self.kg_weights_path.unlink()
        
        # ---- Training config ----
        kg_config = self.config['training']['kg']
        model_config = self.config['model']
        eval_config = self.config['evaluation']
        eval_metrics = eval_config.get('metrics', [
            'mean_reciprocal_rank', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10', 'hits_at_100'
        ])
        rank_metrics, rank_metrics_kwargs = _build_rank_evaluator_metrics(eval_metrics)
        
        num_epochs = kg_config['num_epochs']
        batch_size = kg_config['batch_size']
        lr = kg_config['learning_rate']
        seed = kg_config['random_seed']
        emb_dim = model_config['embedding_dim']
        kg_model_name = model_config['kg_model']
        use_presplit = kg_config.get('use_presplit', False)
        use_inverse = bool(kg_config.get('create_inverse_triples', True))
        
        # ---- Data split ----
        if use_presplit and data is not None and data.get('split_info'):
            # Use pre-split train/valid/test
            print_info("Using pre-split data (train/valid/test.tsv)...")
            from .data_loader import KGDataLoader as _KGDataLoader
            _loader = _KGDataLoader(self.config, str(self.work_dir))
            training_factory, valid_factory, testing_factory = _loader.build_split_triples_factories(
                data, triples_factory
            )
            print_stat("Train triples", f"{training_factory.num_triples:,}")
            print_stat("Valid triples", f"{valid_factory.num_triples:,}")
            print_stat("Test triples",  f"{testing_factory.num_triples:,}")
        else:
            # Fallback: random split
            train_ratio = kg_config.get('train_ratio', 0.8)
            print_info("Using random split for train/test...")
            training_factory, testing_factory = triples_factory.split(
                [train_ratio, 1.0 - train_ratio], random_state=seed
            )
            valid_factory = testing_factory
            print_stat("Train triples", f"{training_factory.num_triples:,}")
            print_stat("Test triples",  f"{testing_factory.num_triples:,}")
        
        print_stat("Entities", f"{training_factory.num_entities:,}")
        print_stat("Relations", f"{training_factory.num_relations}")
        
        # ---- Training parameters ----
        print_info("Training parameters:")
        print_stat("Model", kg_model_name)
        print_stat("Embedding dim", emb_dim)
        print_stat("Epochs", num_epochs)
        print_stat("Batch size", batch_size)
        print_stat("Learning rate", lr)
        print_stat("Optimizer", kg_config['optimizer'])
        print_stat("Device", str(self.device))
        print_stat("Split mode", "pre-split" if use_presplit else "random")
        print_stat("Inverse relations", str(use_inverse))
        
        # ---- Start training ----
        print_info("Starting KG embedding training...")
        print_warning(f"Estimated 1-3 hours ({num_epochs} epochs)...")
        
        start_time = time.time()
        
        pipeline_kwargs = dict(
            training=training_factory,
            validation=valid_factory,
            testing=testing_factory,
            model=kg_model_name,
            model_kwargs=dict(
                embedding_dim=emb_dim,
            ),
            training_kwargs=dict(
                num_epochs=num_epochs,
                batch_size=batch_size,
            ),
            optimizer=kg_config['optimizer'],
            optimizer_kwargs=dict(lr=lr),
            evaluator_kwargs=dict(
                filtered=eval_config['filtered'],
                metrics=rank_metrics,
                metrics_kwargs=rank_metrics_kwargs,
            ),
            evaluation_kwargs=dict(
                batch_size=eval_config['batch_size'],
            ),
            random_seed=seed,
            device=str(self.device),
        )
        # Map common KG training config fields
        if kg_config.get('training_loop'):
            pipeline_kwargs['training_loop'] = kg_config['training_loop']
        if kg_config.get('loss'):
            pipeline_kwargs['loss'] = kg_config['loss']
        if kg_config.get('loss_kwargs'):
            pipeline_kwargs['loss_kwargs'] = kg_config['loss_kwargs']
        if kg_config.get('negative_sampler'):
            pipeline_kwargs['negative_sampler'] = kg_config['negative_sampler']
        if kg_config.get('negative_sampler_kwargs'):
            pipeline_kwargs['negative_sampler_kwargs'] = kg_config['negative_sampler_kwargs']
        if kg_config.get('regularizer'):
            pipeline_kwargs['regularizer'] = kg_config['regularizer']
        if kg_config.get('regularizer_kwargs'):
            pipeline_kwargs['regularizer_kwargs'] = kg_config['regularizer_kwargs']
        if kg_config.get('stopper'):
            pipeline_kwargs['stopper'] = kg_config['stopper']
        if kg_config.get('stopper_kwargs'):
            pipeline_kwargs['stopper_kwargs'] = kg_config['stopper_kwargs']

        result = pipeline(**pipeline_kwargs)
        
        duration = time.time() - start_time
        
        # ---- Output training results ----
        print_success(f"KG embedding training complete! ({duration:.2f}s / {duration/60:.1f}min)")
        self._print_kg_result(result)
        
        # ---- Record training log ----
        log_entry = {
            'model': kg_model_name,
            'embedding_dim': emb_dim,
            'num_epochs': num_epochs,
            'duration_seconds': round(duration, 2),
        }
        
        if hasattr(result, 'losses') and result.losses:
            log_entry['initial_loss'] = float(result.losses[0])
            log_entry['final_loss'] = float(result.losses[-1])
        
        if hasattr(result, 'metric_results') and result.metric_results:
            try:
                log_entry['test_mrr'] = float(result.metric_results.get_metric('mean_reciprocal_rank'))
                log_entry['test_hits_at_10'] = float(result.metric_results.get_metric('hits_at_10'))
                log_entry['test_hits_at_100'] = float(result.metric_results.get_metric('hits_at_100'))
            except:
                pass
        
        self.training_log['kg_training'] = log_entry
        self._save_training_log()
        
        # ---- Save model ----
        save_checkpoint(result, self.kg_model_path, "KG embedding model")
        torch.save(result.model.state_dict(), self.kg_weights_path)
        print_success(f"Model weights saved: {self.kg_weights_path.name}")
        
        return result
    
    # ================================================================
    # Phase B: Feature fusion network training
    # ================================================================
    
    def train_fusion_network(self, kg_result, feature_result: dict,
                              force_retrain: bool = False,
                              gnn_embeddings: np.ndarray = None) -> nn.Module:
        """
        Train feature fusion network

        Uses three-source MultiSourceFusionNetwork when GNN embeddings are available;
        falls back to FeatureFusionNetwork otherwise (backward compatible).

        Args:
            kg_result: return value of train_kg_embeddings()
            feature_result: return value of DiseaseFeatureEncoder.encode_all_features()
            force_retrain: force retraining
            gnn_embeddings: GNN embedding matrix (optional)

        Returns:
            Trained fusion model
        """
        use_multi_source = gnn_embeddings is not None
        mode_str = "three-source (KG+GNN+features)" if use_multi_source else "two-source (KG+features)"
        print_info(f"Fusion mode: {mode_str}")

        # === Checkpoint check ===
        if self.fusion_model_path.exists() and not force_retrain:
            print_success("Found trained fusion network checkpoint, loading...")
            model = load_fusion_model(self.fusion_model_path, self.device)
            model.eval()
            return model

        # ---- Check feature availability ----
        feature_dim = feature_result.get('total_feature_dim', 0)
        if feature_dim == 0:
            print_warning("No disease features available, skipping fusion training")
            return None

        # ---- Extract KG embeddings ----
        print_info("Extracting KG entity embedding matrix...")
        kg_model = kg_result.model
        training_triples = kg_result.training
        entity_to_id = training_triples.entity_to_id

        with torch.no_grad():
            entity_repr = kg_model.entity_representations[0]
            all_entity_emb = entity_repr(
                indices=torch.arange(kg_model.num_entities, device=self.device)
            ).cpu().numpy()

        embedding_dim = all_entity_emb.shape[1]
        print_stat("Entity embedding matrix shape", all_entity_emb.shape)

        # ---- Build disease index ----
        disease_kg_ids = feature_result['disease_kg_ids']
        kg_to_row = feature_result['kg_to_row']
        feature_matrix = feature_result['feature_matrix']

        valid_indices = []
        valid_features = []

        for kg_id in disease_kg_ids:
            if kg_id in entity_to_id and kg_id in kg_to_row:
                kg_idx = entity_to_id[kg_id]
                feat_row = kg_to_row[kg_id]
                valid_indices.append(kg_idx)
                valid_features.append(feature_matrix[feat_row])

        if len(valid_indices) == 0:
            print_warning("No disease entities found in both KG and features, skipping fusion training")
            return None

        valid_features = np.array(valid_features, dtype=np.float32)
        print_stat("Diseases in fusion training", len(valid_indices))
        print_stat("Feature dim", feature_dim)
        print_stat("Embedding dim", embedding_dim)

        # ---- Build dataset ----
        if use_multi_source:
            dataset = MultiSourceFusionDataset(
                all_entity_emb, valid_features, valid_indices,
                gnn_embeddings=gnn_embeddings
            )
            print_stat("GNN embedding dim", gnn_embeddings.shape)
        else:
            dataset = FusionDataset(all_entity_emb, valid_features, valid_indices)

        total_size = len(dataset)
        if total_size == 1:
            # Small-sample guard: avoid train_size=0 causing DataLoader error
            train_dataset = dataset
            val_dataset = dataset
            train_size = 1
            val_size = 1
            print_warning("Only 1 fusion sample, train/valid will reuse same sample")
        else:
            train_size = max(1, int(0.8 * total_size))
            if train_size >= total_size:
                train_size = total_size - 1
            val_size = total_size - train_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        print_stat("Fusion train set", f"{train_size}  diseases")
        print_stat("Fusion valid set", f"{val_size}  diseases")

        # ---- Config ----
        fusion_config = self.config['training']['fusion']
        model_fusion_config = self.config['model']['fusion']

        num_epochs = fusion_config['num_epochs']
        batch_size = max(1, min(fusion_config['batch_size'], train_size))
        lr = fusion_config['learning_rate']
        weight_decay = fusion_config['weight_decay']
        patience = fusion_config['patience']
        min_delta = fusion_config['min_delta']

        # ---- Create fusion model ----
        if use_multi_source:
            # Small batch (batch_size=1): BatchNorm1d fails in train mode
            use_batch_norm = model_fusion_config.get('use_batch_norm', True)
            if batch_size < 2:
                use_batch_norm = False
                print_warning("Small sample training: BatchNorm disabled to prevent training crash")
            fusion_model = MultiSourceFusionNetwork(
                embedding_dim=embedding_dim,
                feature_dim=feature_dim,
                hidden_dims=model_fusion_config['hidden_dims'],
                dropout=model_fusion_config['dropout'],
                activation=model_fusion_config.get('activation', 'relu'),
                use_batch_norm=use_batch_norm,
            ).to(self.device)
        else:
            # Small batch (batch_size=1): BatchNorm1d fails in train mode
            use_batch_norm = model_fusion_config.get('use_batch_norm', True)
            if batch_size < 2:
                use_batch_norm = False
                print_warning("Small sample training: BatchNorm disabled to prevent training crash")
            fusion_model = FeatureFusionNetwork(
                embedding_dim=embedding_dim,
                feature_dim=feature_dim,
                hidden_dims=model_fusion_config['hidden_dims'],
                dropout=model_fusion_config['dropout'],
                activation=model_fusion_config.get('activation', 'relu'),
                use_batch_norm=use_batch_norm,
                fusion_strategy=model_fusion_config['fusion_strategy']
            ).to(self.device)

        print_info(f"Fusion network type: {type(fusion_model).__name__}")
        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
        print_stat("Total parameters", f"{total_params:,}")
        print_stat("Trainable parameters", f"{trainable_params:,}")

        # ---- Training setup ----
        optimizer = optim.Adam(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        val_batch_size = max(1, min(batch_size, val_size))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        # ---- Training loop ----
        print_info(f"Starting fusion network training ({num_epochs} epochs)...")

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(1, num_epochs + 1):
            # --- Training phase ---
            fusion_model.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if use_multi_source:
                    kg_emb, features, gnn_emb, target_emb = batch
                    gnn_emb = gnn_emb.to(self.device)
                else:
                    kg_emb, features, target_emb = batch
                    gnn_emb = None

                kg_emb = kg_emb.to(self.device)
                features = features.to(self.device)
                target_emb = target_emb.to(self.device)

                # Forward pass
                if use_multi_source:
                    fused_emb = fusion_model(kg_emb, features, gnn_emb)
                else:
                    fused_emb = fusion_model(kg_emb, features)
                loss = criterion(fused_emb, target_emb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)

            # --- Validation phase ---
            fusion_model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if use_multi_source:
                        kg_emb, features, gnn_emb, target_emb = batch
                        gnn_emb = gnn_emb.to(self.device)
                    else:
                        kg_emb, features, target_emb = batch
                        gnn_emb = None

                    kg_emb = kg_emb.to(self.device)
                    features = features.to(self.device)
                    target_emb = target_emb.to(self.device)

                    if use_multi_source:
                        fused_emb = fusion_model(kg_emb, features, gnn_emb)
                    else:
                        fused_emb = fusion_model(kg_emb, features)
                    val_loss = criterion(fused_emb, target_emb)
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch:3d}/{num_epochs} │ "
                      f"train_loss: {avg_train_loss:.6f} | "
                      f"val_loss: {avg_val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")

            # --- Early stopping check ---
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_fusion_model(
                    fusion_model, self.fusion_model_path,
                    config={
                        'hidden_dims': model_fusion_config['hidden_dims'],
                        'dropout': model_fusion_config['dropout'],
                        'best_val_loss': best_val_loss,
                        'best_epoch': epoch,
                    }
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print_warning(f"Early stopping: {patience} epochs without improvement")
                    break

        # ---- Load best model ----
        fusion_model = load_fusion_model(self.fusion_model_path, self.device)
        fusion_model.eval()

        # ---- Record log ----
        self.training_log['fusion_training'] = {
            'num_epochs_run': epoch,
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1]),
            'train_diseases': train_size,
            'val_diseases': val_size,
            'feature_dim': feature_dim,
            'embedding_dim': embedding_dim,
            'use_multi_source': use_multi_source,
        }
        self._save_training_log()

        print_success(f"Fusion network training complete! Best val loss: {best_val_loss:.6f}")

        return fusion_model
    
    # ================================================================
    # Helper methods
    # ================================================================
    
    def _print_kg_result(self, result):
        """Print KG training result summary"""
        print_info("=== KG embedding training results ===")
        
        if hasattr(result, 'losses') and result.losses:
            print_stat("Initial loss", f"{result.losses[0]:.4f}")
            print_stat("Final loss", f"{result.losses[-1]:.4f}")
            print_stat("Loss reduction", f"{result.losses[0] - result.losses[-1]:.4f}")
        
        if hasattr(result, 'metric_results') and result.metric_results:
            try:
                mrr = result.metric_results.get_metric('mean_reciprocal_rank')
                h1 = result.metric_results.get_metric('hits_at_1')
                h3 = result.metric_results.get_metric('hits_at_3')
                h10 = result.metric_results.get_metric('hits_at_10')
                h100 = result.metric_results.get_metric('hits_at_100')
                print_stat("Test MRR", f"{mrr:.4f}")
                print_stat("Test Hits@1", f"{h1:.4f}")
                print_stat("Test Hits@3", f"{h3:.4f}")
                print_stat("Test Hits@10", f"{h10:.4f}")
                print_stat("Test Hits@100", f"{h100:.4f}")
            except Exception:
                print_info("Evaluation metrics not available")
        
        print_stat("Model entities", f"{result.model.num_entities:,}")
        print_stat("Model relations", f"{result.model.num_relations}")
    
    def _save_training_log(self):
        """Save training log to JSON"""
        import json
        with open(self.training_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)

