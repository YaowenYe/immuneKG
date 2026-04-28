"""
immuneKG GNN — PNA- （HeteroPNA-Attn）

 ：
 PNA（Principal Neighbourhood Aggregation） + 
 （HGT） 。

 :
 ┌────────────────────────────────────────────────────┐
 │ ComplEx (dim=128) │
 │ ↓ │
 │ ┌──────────────┐ ┌──────────────────────┐ │
 │ │ PNA │ │ HGT │ │
 │ │ ( ) │ │ ( +NeighborLoader)│ │
 │ │ 2 PNAConv │ │ 2 HGTConv │ │
 │ │ mean/max/ │ │ × │ │
 │ │ min/std │ │ │ │
 │ │ + │ │ │ │
 │ └──────┬───────┘ └──────────┬───────────┘ │
 │ │ │ │
 │ └──────────┬────────────┘ │
 │ ↓ │
 │ (Attention Gate) │
 │ ↓ │
 │ GNN (dim=128) │
 └────────────────────────────────────────────────────┘

 :
 - : 
 - : GNN ComplEx 
 - AP : （ ）
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

try:
    from torch_geometric.nn import PNAConv, HGTConv, Linear as PygLinear
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import HeteroData, Data
    from torch_geometric.utils import negative_sampling, degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .utils import (
    print_info, print_stat, print_success, print_warning, print_error,
    save_checkpoint, load_checkpoint, checkpoint_exists, Timer
)


# ============================================================
# PNA ： + 
# ============================================================

class PNABranch(nn.Module):
    """
 PNA（Principal Neighbourhood Aggregation） 

 ， 。

 (Aggregators):
 - mean: — 
 - max: — 
 - min: — 
 - std: — " "
 ★ std ： ，
 → 

 (Scalers):
 - identity: — 
 - amplification: — (log(deg+1))
 ★ 
 - attenuation: — (1/log(deg+1))
 """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 deg: torch.Tensor,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
 Args:
 in_channels: 
 hidden_channels: 
 out_channels: 
 deg: （PNA ）
 num_layers: PNA 
 dropout: Dropout 
 """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # PNA 
        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels

            # PNA 
            conv = PNAConv(
                in_channels=in_ch,
                out_channels=out_ch,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                towers=1,               # （1= ）
                pre_layers=1,            # MLP 
                post_layers=1,           # MLP 
                divide_input=False,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_ch))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
 

 Args:
 x: , shape=(num_nodes, in_channels)
 edge_index: , shape=(2, num_edges)

 Returns:
 , shape=(num_nodes, out_channels)
 """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)

            if i < self.num_layers - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # （ ）
            if x.shape == x_new.shape:
                x_new = x_new + x

            x = x_new

        return x


# ============================================================
# HGT ： 
# ============================================================

class HGTBranch(nn.Module):
    """
 HGT（Heterogeneous Graph Transformer） 

 。

 :
 - 
 - 
 - NeighborLoader 
 """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 metadata: tuple,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        """
 Args:
 in_channels: 
 hidden_channels: 
 out_channels: 
 metadata: (node_types, edge_types)
 num_layers: HGT 
 num_heads: 
 dropout: Dropout 
 """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            heads = min(num_heads, out_ch)  # heads 

            conv = HGTConv(
                in_channels=in_ch,
                out_channels=out_ch,
                metadata=metadata,
                heads=heads,
            )
            self.convs.append(conv)

            # LayerNorm
            norm_dict = nn.ModuleDict({
                ntype: nn.LayerNorm(out_ch)
                for ntype in metadata[0]  # node_types
            })
            self.norms.append(norm_dict)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
 

 Args:
 x_dict: { : }
 edge_index_dict: { : }

 Returns:
 { : }
 """
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict_new = conv(x_dict, edge_index_dict)

            # LayerNorm + + 
            for ntype in x_dict_new:
                x_new = norm_dict[ntype](x_dict_new[ntype])

                if i < self.num_layers - 1:
                    x_new = F.relu(x_new)
                    x_new = F.dropout(x_new, p=self.dropout, training=self.training)

                # 
                if ntype in x_dict and x_dict[ntype].shape == x_new.shape:
                    x_new = x_new + x_dict[ntype]

                x_dict_new[ntype] = x_new

            x_dict = x_dict_new

        return x_dict


# ============================================================
# GNN：PNA + HGT 
# ============================================================

class HeteroPNANet(nn.Module):
    """
 PNA- （HeteroPNA-Attn）

 ：
 PNA — 
 HGT — 

 ： 

 ？
 - PNA std + " " 
 - HGT " "（ disease→gene drug→gene ）
 - ， 
 """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 deg: torch.Tensor,
                 metadata: tuple,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        """
 Args:
 in_channels: （= ComplEx embedding_dim）
 hidden_channels: 
 out_channels: （= ComplEx embedding_dim）
 deg: PNA 
 metadata: 
 num_layers: 
 num_heads: HGT 
 dropout: Dropout 
 """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ---- PNA （ ）----
        self.pna_branch = PNABranch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            deg=deg,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ---- HGT （ ）----
        self.hgt_branch = HGTBranch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ---- ----
        # PNA HGT 
        self.gate_net = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

        # ---- ----
        self.output_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(self,
                homo_x: torch.Tensor,
                homo_edge_index: torch.Tensor,
                hetero_x_dict: Dict[str, torch.Tensor],
                hetero_edge_index_dict: Dict[tuple, torch.Tensor],
                type_offset: Dict[str, int]) -> torch.Tensor:
        """
 

 Args:
 homo_x: 
 homo_edge_index: 
 hetero_x_dict: 
 hetero_edge_index_dict: 
 type_offset: 

 Returns:
 , shape=(total_nodes, out_channels)
 """
        total_nodes = homo_x.size(0)

        # ---- PNA ： ----
        pna_out = self.pna_branch(homo_x, homo_edge_index)
        # pna_out: (total_nodes, out_channels)

        # ---- HGT ： ----
        hgt_out_dict = self.hgt_branch(hetero_x_dict, hetero_edge_index_dict)

        # 
        hgt_out = torch.zeros(total_nodes, self.out_channels,
                              device=homo_x.device)
        for ntype in sorted(type_offset.keys()):
            offset = type_offset[ntype]
            if ntype in hgt_out_dict:
                num_nodes = hgt_out_dict[ntype].size(0)
                hgt_out[offset:offset + num_nodes] = hgt_out_dict[ntype]

        # ---- ----
        gate = self.gate_net(torch.cat([pna_out, hgt_out], dim=-1))
        merged = gate * pna_out + (1.0 - gate) * hgt_out

        # ---- + ----
        output = self.output_proj(merged)
        if output.shape == homo_x.shape:
            output = output + homo_x  # ComplEx 

        return output

    def get_config(self) -> dict:
        """ """
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'model_type': 'HeteroPNANet',
        }


# ============================================================
# GNN 
# ============================================================

class GNNTrainer:
    """
 GNN 

 :
 1. — （ + ）
 2. — GNN ComplEx （ ）
 3. AP — （ ） 

 NeighborLoader （ ）。
 """

    def __init__(self, config: dict, device: torch.device, work_dir: str):
        """
 Args:
 config: 
 device: 
 work_dir: 
 """
        self.config = config
        self.device = device
        self.work_dir = Path(work_dir)

        # GNN 
        gnn_cfg = config.get('gnn', {})
        self.hidden_dim = gnn_cfg.get('hidden_dim', 128)
        self.num_layers = gnn_cfg.get('num_layers', 2)
        self.num_heads = gnn_cfg.get('num_heads', 4)
        self.dropout = gnn_cfg.get('dropout', 0.2)
        self.num_epochs = gnn_cfg.get('training', {}).get('num_epochs', 50)
        self.batch_size = gnn_cfg.get('training', {}).get('batch_size', 1024)
        self.lr = gnn_cfg.get('training', {}).get('learning_rate', 0.001)
        self.num_neighbors = gnn_cfg.get('training', {}).get('num_neighbors', [10, 5])
        self.recon_weight = gnn_cfg.get('training', {}).get('recon_weight', 0.5)
        self.ap_weight = gnn_cfg.get('training', {}).get('ap_weight', 0.1)
        self.force_cpu = bool(gnn_cfg.get('training', {}).get('force_cpu', True))

    def train_gnn(self, graph_result: dict, kg_result=None,
                  force_retrain: bool = False) -> Tuple[nn.Module, np.ndarray]:
        """
 HeteroPNA-Attn GNN

 Args:
 graph_result: （ HeteroGraphBuilder）
 kg_result: ComplEx 
 force_retrain: 

 Returns:
 (gnn_model, gnn_embeddings) 
 gnn_embeddings: shape=(total_nodes, emb_dim)
 """
        model_path = self.work_dir / 'gnn_model.pth'
        emb_path = self.work_dir / 'gnn_embeddings.pkl'

        # 
        if not force_retrain and model_path.exists() and emb_path.exists():
            print_info(" GNN ...")
            return self._load_gnn(graph_result, model_path, emb_path)

        print_info(" HeteroPNA-Attn GNN...")

        emb_dim = self.config['model']['embedding_dim']
        hetero_data = graph_result['hetero_data']
        homo_data = graph_result['homo_data']
        deg = graph_result['deg']
        metadata = graph_result['metadata']
        type_offset = graph_result['type_offset']
        total_nodes = graph_result['total_nodes']

        # GNN PyG/CUDA ， CPU 
        gnn_device = torch.device('cpu') if self.force_cpu else self.device
        if gnn_device.type != self.device.type:
            print_warning(f"GNN   {gnn_device}  （ ），  {self.device}")

        # ---- ----
        model = HeteroPNANet(
            in_channels=emb_dim,
            hidden_channels=self.hidden_dim,
            out_channels=emb_dim,
            deg=deg,
            metadata=metadata,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(gnn_device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_stat("GNN ", f"{param_count:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )

        # ---- ----
        # 
        homo_x = homo_data.x.to(gnn_device)
        homo_edge_index = homo_data.edge_index.to(gnn_device)

        hetero_x_dict = {
            ntype: hetero_data[ntype].x.to(gnn_device)
            for ntype in metadata[0]
        }
        hetero_edge_dict = {
            etype: hetero_data[etype].edge_index.to(gnn_device)
            for etype in metadata[1]
            if hasattr(hetero_data[etype], 'edge_index')
        }

        # 
        target_emb = homo_x.clone().detach()

        # AP （Average Popularity ）
        node_deg = degree(homo_edge_index[1], num_nodes=total_nodes).float()
        # AP ： 
        ap_weights = 1.0 / (torch.log(node_deg + 2.0))  # +2 log(1)=0
        ap_weights = ap_weights / ap_weights.mean()  # 
        ap_weights = ap_weights.to(gnn_device)

        print_stat("AP ", f"[{ap_weights.min():.3f}, {ap_weights.max():.3f}]")
        self._validate_graph_tensors(homo_x, homo_edge_index, hetero_x_dict, hetero_edge_dict, total_nodes)

        # ---- ----
        best_loss = float('inf')
        patience_counter = 0
        patience = 15

        timer = Timer()
        timer.start('gnn_training')

        for epoch in range(1, self.num_epochs + 1):
            model.train()

            # （ mini-batch）
            out = model(
                homo_x, homo_edge_index,
                hetero_x_dict, hetero_edge_dict,
                type_offset
            )

            # ---- 1: （AP ）----
            recon_diff = (out - target_emb) ** 2  # (N, D)
            recon_per_node = recon_diff.mean(dim=1)  # (N,)
            # AP ： 
            recon_loss = (recon_per_node * ap_weights).mean()

            # ---- 2: ----
            # ： 
            pos_src = homo_edge_index[0]
            pos_dst = homo_edge_index[1]

            # （ ）
            num_edges = pos_src.size(0)
            sample_size = min(num_edges, self.batch_size * 4)
            perm = torch.randperm(num_edges, device=gnn_device)[:sample_size]
            pos_src_s = pos_src[perm]
            pos_dst_s = pos_dst[perm]

            # 
            pos_score = (out[pos_src_s] * out[pos_dst_s]).sum(dim=1)

            # 
            neg_dst = torch.randint(0, total_nodes, (sample_size,), device=gnn_device)
            neg_score = (out[pos_src_s] * out[neg_dst]).sum(dim=1)

            # BPR 
            link_loss = -F.logsigmoid(pos_score - neg_score).mean()

            # ---- ----
            total_loss = (1 - self.recon_weight) * link_loss + self.recon_weight * recon_loss

            optimizer.zero_grad()
            total_loss.backward()

            # 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss.item())

            # ---- ----
            if epoch % 5 == 0 or epoch == 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch:3d}/{self.num_epochs} │ "
                      f"Total={total_loss:.6f} │ "
                      f"Link={link_loss:.6f} │ "
                      f"Recon={recon_loss:.6f} │ "
                      f"LR={lr:.6f}")

            # 
            if total_loss < best_loss - 1e-5:
                best_loss = total_loss.item()
                patience_counter = 0
                # 
                torch.save(model.state_dict(), str(model_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print_info(f"    (patience={patience})")
                    break

        duration = timer.stop('gnn_training')
        print_success(f"GNN   ( : {best_loss:.6f}, "
                      f" : {timer.format_duration(duration)})")

        # ---- ----
        model.load_state_dict(torch.load(str(model_path), map_location=gnn_device))
        model.eval()
        with torch.no_grad():
            gnn_embeddings = model(
                homo_x, homo_edge_index,
                hetero_x_dict, hetero_edge_dict,
                type_offset
            ).cpu().numpy()

        # 
        save_checkpoint(gnn_embeddings, str(emb_path))
        print_stat("GNN ", gnn_embeddings.shape)

        # 
        import json
        config_info = model.get_config()
        config_info.update({
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
        })
        with open(str(model_path.with_suffix('.json')), 'w') as f:
            json.dump(config_info, f, indent=2)

        return model, gnn_embeddings

    def _validate_graph_tensors(self, homo_x, homo_edge_index, hetero_x_dict, hetero_edge_dict, total_nodes):
        """ ， PyG 。"""
        if torch.isnan(homo_x).any() or torch.isinf(homo_x).any():
            raise ValueError(" NaN/Inf")
        if homo_edge_index.numel() == 0:
            raise ValueError(" ， GNN")
        max_idx = int(homo_edge_index.max().item())
        min_idx = int(homo_edge_index.min().item())
        if min_idx < 0 or max_idx >= total_nodes:
            raise ValueError(
                f"  edge_index  : min={min_idx}, max={max_idx}, total_nodes={total_nodes}"
            )
        for ntype, x in hetero_x_dict.items():
            if x.numel() == 0:
                raise ValueError(f"  '{ntype}'  ")
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(f"  '{ntype}'   NaN/Inf")
        for etype, edge_index in hetero_edge_dict.items():
            src_type, _, dst_type = etype
            src_n = hetero_x_dict[src_type].size(0)
            dst_n = hetero_x_dict[dst_type].size(0)
            if edge_index.numel() == 0:
                continue
            src_max = int(edge_index[0].max().item())
            dst_max = int(edge_index[1].max().item())
            src_min = int(edge_index[0].min().item())
            dst_min = int(edge_index[1].min().item())
            if src_min < 0 or dst_min < 0 or src_max >= src_n or dst_max >= dst_n:
                raise ValueError(
                    f"  {etype}: src[{src_min},{src_max}]/{src_n}, dst[{dst_min},{dst_max}]/{dst_n}"
                )

    def _load_gnn(self, graph_result, model_path, emb_path):
        """ GNN """
        import json

        emb_dim = self.config['model']['embedding_dim']
        deg = graph_result['deg']
        metadata = graph_result['metadata']

        model = HeteroPNANet(
            in_channels=emb_dim,
            hidden_channels=self.hidden_dim,
            out_channels=emb_dim,
            deg=deg,
            metadata=metadata,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        state = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state)
        model.eval()

        gnn_embeddings = load_checkpoint(str(emb_path))
        print_stat("GNN ", gnn_embeddings.shape)

        return model, gnn_embeddings


# ============================================================
# / 
# ============================================================

def save_gnn_model(model: HeteroPNANet, filepath: str):
    """ GNN """
    torch.save(model.state_dict(), filepath)


def load_gnn_embeddings(filepath: str) -> np.ndarray:
    """ GNN """
    return load_checkpoint(filepath)
