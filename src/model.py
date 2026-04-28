"""
immuneKG 

 （Feature Fusion Network, FFN），
 KG ， 。

 :
 - gate: — KG 
 - concat: — KG MLP 
 - add: — KG 
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class FeatureFusionNetwork(nn.Module):
    """
 （FFN）
 
 :
 - KG : (batch_size, embedding_dim) — ComplEx 
 - : (batch_size, feature_dim) — 
 
 :
 - : (batch_size, embedding_dim) — 
 
 （ gate ）:
 : feature → Linear → BN → ReLU → Linear → feature_proj
 : [kg_emb; feature_proj] → Linear → Sigmoid → gate
 : gate * kg_emb + (1 - gate) * feature_proj
 """
    
    def __init__(self,
                 embedding_dim: int,
                 feature_dim: int,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.3,
                 activation: str = "relu",
                 use_batch_norm: bool = True,
                 fusion_strategy: str = "gate"):
        """
 Args:
 embedding_dim: KG （ComplEx embedding_dim）
 feature_dim: （ ）
 hidden_dims: MLP 
 dropout: Dropout 
 activation: (relu / gelu / leaky_relu)
 use_batch_norm: BatchNorm
 fusion_strategy: (gate / concat / add)
 """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy
        self.use_batch_norm = use_batch_norm
        
        # ---- ----
        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }.get(activation, nn.ReLU)
        
        # ---- (Feature Projection Branch) ----
        # KG 
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        # embedding_dim
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.feature_branch = nn.Sequential(*layers)
        
        # ---- (Fusion Layer) ----
        if fusion_strategy == 'gate':
            # ： 
            self.gate_layer = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
        elif fusion_strategy == 'concat':
            # ： embedding_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )
        elif fusion_strategy == 'add':
            # ： KG （ ）
            self.scale = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f" : {fusion_strategy}")
        
        # ---- ----
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, kg_embedding: torch.Tensor,
                disease_features: torch.Tensor) -> torch.Tensor:
        """
 
 
 Args:
 kg_embedding: KG , shape=(batch_size, embedding_dim)
 disease_features: , shape=(batch_size, feature_dim)
 
 Returns:
 , shape=(batch_size, embedding_dim)
 """
        # 
        feat_proj = self.feature_branch(disease_features)
        
        # 
        if self.fusion_strategy == 'gate':
            # 
            combined = torch.cat([kg_embedding, feat_proj], dim=-1)
            gate = self.gate_layer(combined)
            fused = gate * kg_embedding + (1.0 - gate) * feat_proj
            
        elif self.fusion_strategy == 'concat':
            # 
            combined = torch.cat([kg_embedding, feat_proj], dim=-1)
            fused = self.concat_proj(combined)
            
        elif self.fusion_strategy == 'add':
            # 
            fused = kg_embedding + self.scale * feat_proj
        
        # LayerNorm
        fused = self.output_norm(fused)
        
        return fused
    
    def get_config(self) -> dict:
        """ （ ）"""
        return {
            'embedding_dim': self.embedding_dim,
            'feature_dim': self.feature_dim,
            'fusion_strategy': self.fusion_strategy,
            'use_batch_norm': self.use_batch_norm,
        }


# ============================================================
# ：KG + GNN + 
# ============================================================

class MultiSourceFusionNetwork(nn.Module):
    """
 （Multi-Source FFN）

 FFN GNN ， :
 - ComplEx KG : 
 - HeteroPNA-Attn : + 
 - : 

 :
 disease_features → MLP → feat_proj (dim=emb_dim)
 gnn_embedding → Linear → gnn_proj (dim=emb_dim)

 [kg_emb; gnn_proj; feat_proj] → → fused_emb
 """

    def __init__(self,
                 embedding_dim: int,
                 feature_dim: int,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.3,
                 activation: str = "relu",
                 use_batch_norm: bool = True):
        """
 Args:
 embedding_dim: KG/GNN 
 feature_dim: 
 hidden_dims: MLP 
 dropout: Dropout 
 activation: 
 use_batch_norm: BatchNorm
 """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.use_batch_norm = use_batch_norm

        act_fn = {
            'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU
        }.get(activation, nn.ReLU)

        # ---- ----
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.feature_branch = nn.Sequential(*layers)

        # ---- GNN ----
        self.gnn_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- ----
        # : [kg_emb; gnn_proj; feat_proj] → 
        self.trigate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 3),
            nn.ReLU(),
            nn.Linear(embedding_dim * 3, 3),  # 
        )

        # ---- ----
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self,
                kg_embedding: torch.Tensor,
                disease_features: torch.Tensor,
                gnn_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
 

 Args:
 kg_embedding: KG , shape=(batch, emb_dim)
 disease_features: , shape=(batch, feat_dim)
 gnn_embedding: GNN , shape=(batch, emb_dim)， 

 Returns:
 , shape=(batch, emb_dim)
 """
        # 
        feat_proj = self.feature_branch(disease_features)

        if gnn_embedding is not None:
            # 
            gnn_proj = self.gnn_branch(gnn_embedding)

            # 
            combined = torch.cat([kg_embedding, gnn_proj, feat_proj], dim=-1)
            gate_logits = self.trigate(combined)  # (batch, 3)
            gate_weights = F.softmax(gate_logits, dim=-1)  # [0,1] =1

            # 
            fused = (gate_weights[:, 0:1] * kg_embedding +
                     gate_weights[:, 1:2] * gnn_proj +
                     gate_weights[:, 2:3] * feat_proj)
        else:
            # （ GNN gate ）
            combined = torch.cat([kg_embedding, feat_proj], dim=-1)
            # 
            gate = torch.sigmoid(
                nn.functional.linear(combined,
                                     self.trigate[0].weight[:self.embedding_dim, :self.embedding_dim*2],
                                     self.trigate[0].bias[:self.embedding_dim] if self.trigate[0].bias is not None else None)
            )
            fused = gate * kg_embedding + (1.0 - gate) * feat_proj

        return self.output_norm(fused)

    def get_config(self) -> dict:
        """ """
        return {
            'embedding_dim': self.embedding_dim,
            'feature_dim': self.feature_dim,
            'fusion_strategy': 'multi_source_trigate',
            'model_class': 'MultiSourceFusionNetwork',
            'use_batch_norm': self.use_batch_norm,
        }


# ============================================================
# 
# ============================================================

class MultiSourceFusionDataset(torch.utils.data.Dataset):
    """
 

 : (kg_embedding, disease_feature, gnn_embedding, target_embedding)
 """

    def __init__(self,
                 kg_embeddings: np.ndarray,
                 disease_features: np.ndarray,
                 entity_indices: list,
                 gnn_embeddings: np.ndarray = None):
        """
 Args:
 kg_embeddings: KG （ ）
 disease_features: （ ）
 entity_indices: KG 
 gnn_embeddings: GNN （ ， ）
 """
        assert len(disease_features) == len(entity_indices)

        self.kg_embeddings = torch.from_numpy(
            kg_embeddings[entity_indices].astype(np.float32)
        )
        self.disease_features = torch.from_numpy(
            disease_features.astype(np.float32)
        )
        self.target_embeddings = self.kg_embeddings.clone()

        # GNN （ ）
        if gnn_embeddings is not None:
            self.gnn_embeddings = torch.from_numpy(
                gnn_embeddings[entity_indices].astype(np.float32)
            )
            self.has_gnn = True
        else:
            self.gnn_embeddings = None
            self.has_gnn = False

    def __len__(self):
        return len(self.disease_features)

    def __getitem__(self, idx):
        if self.has_gnn:
            return (
                self.kg_embeddings[idx],
                self.disease_features[idx],
                self.gnn_embeddings[idx],
                self.target_embeddings[idx],
            )
        else:
            return (
                self.kg_embeddings[idx],
                self.disease_features[idx],
                torch.zeros_like(self.kg_embeddings[idx]),  # 
                self.target_embeddings[idx],
            )


# ============================================================
# 
# ============================================================

class FusionDataset(torch.utils.data.Dataset):
    """
 
 
 : (kg_embedding, disease_feature, target_embedding)
 target_embedding KG ，
 / 。
 
 ：
 - : MSE(fused_embedding, target_embedding)
 - : 
 """
    
    def __init__(self,
                 kg_embeddings: np.ndarray,
                 disease_features: np.ndarray,
                 entity_indices: list):
        """
 Args:
 kg_embeddings: KG （ ）
 disease_features: （ ）
 entity_indices: KG 
 """
        assert len(disease_features) == len(entity_indices), \
            f" ({len(disease_features)}) ({len(entity_indices)}) "
        
        self.kg_embeddings = torch.from_numpy(
            kg_embeddings[entity_indices].astype(np.float32)
        )
        self.disease_features = torch.from_numpy(
            disease_features.astype(np.float32)
        )
        self.target_embeddings = self.kg_embeddings.clone()  # 
    
    def __len__(self):
        return len(self.disease_features)
    
    def __getitem__(self, idx):
        return (
            self.kg_embeddings[idx],
            self.disease_features[idx],
            self.target_embeddings[idx]
        )


# ============================================================
# ： 
# ============================================================

class EnhancedScorer:
    """
 
 
 。
 
 :
 score(disease, target) = 
 α * ComplEx_score(disease, target) + 
 (1-α) * cosine_similarity(fused_disease_emb, target_emb)
 
 α ， 0.5。
 """
    
    def __init__(self, alpha: float = 0.5):
        """
 Args:
 alpha: KG （0~1 ）
 """
        self.alpha = alpha
    
    def compute_similarity_scores(self,
                                   disease_embeddings: torch.Tensor,
                                   target_embeddings: torch.Tensor) -> torch.Tensor:
        """
 
 
 Args:
 disease_embeddings: (num_diseases, emb_dim)
 target_embeddings: (num_targets, emb_dim)
 
 Returns:
 : (num_diseases, num_targets)
 """
        # L2 
        disease_norm = F.normalize(disease_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        
        # 
        similarity = torch.mm(disease_norm, target_norm.t())
        
        return similarity


# ============================================================
# 
# ============================================================

def save_fusion_model(model: FeatureFusionNetwork, filepath: str, config: dict = None):
    """
 
 
 Args:
 model: FeatureFusionNetwork 
 filepath: （.pth）
 config: 
 """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 
    torch.save(model.state_dict(), filepath)
    
    # 
    model_config = model.get_config()
    if config:
        model_config.update(config)
    
    config_path = filepath.with_suffix('.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)


def load_fusion_model(filepath: str, device: torch.device = None):
    """
 （ FeatureFusionNetwork MultiSourceFusionNetwork）
 
 Args:
 filepath: （.pth）
 device: 
 
 Returns:
 
 """
    filepath = Path(filepath)
    
    # 
    config_path = filepath.with_suffix('.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 
    model_class = config.get('model_class', 'FeatureFusionNetwork')
    
    if model_class == 'MultiSourceFusionNetwork':
        model = MultiSourceFusionNetwork(
            embedding_dim=config['embedding_dim'],
            feature_dim=config['feature_dim'],
            hidden_dims=config.get('hidden_dims', [256, 128]),
            dropout=config.get('dropout', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
        )
    else:
        model = FeatureFusionNetwork(
            embedding_dim=config['embedding_dim'],
            feature_dim=config['feature_dim'],
            fusion_strategy=config.get('fusion_strategy', 'gate'),
            hidden_dims=config.get('hidden_dims', [256, 128]),
            dropout=config.get('dropout', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
        )
    
    # 
    state_dict = torch.load(filepath, map_location=device or 'cpu', weights_only=False)
    model.load_state_dict(state_dict)
    
    if device:
        model = model.to(device)
    
    return model
