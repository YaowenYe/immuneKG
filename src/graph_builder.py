"""
immuneKG 

 PyTorch Geometric ，
 （HeteroData） （Data） 。

 :
 - build_hetero_graph(): （ / ）
 - build_homo_graph(): （ ）
 - compute_deg_histogram(): （PNA ）
 - init_node_features(): ComplEx 
"""

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from torch_geometric.data import HeteroData, Data
    from torch_geometric.utils import degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .utils import (
    print_info, print_stat, print_success, print_warning, print_error,
    save_checkpoint, load_checkpoint, checkpoint_exists
)


def check_pyg_available():
    """ PyTorch Geometric """
    if not HAS_PYG:
        print_error("PyTorch Geometric ！GNN torch-geometric。")
        print_info(" : pip install torch-geometric")
        print_info(" : https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
        return False
    return True


class HeteroGraphBuilder:
    """
 

 KG DataFrame PyG （HeteroData），
 （disease, gene/protein, drug ） （ ）。

 ， PNA 。
 """

    def __init__(self, config: dict, work_dir: str):
        """
 Args:
 config: 
 work_dir: （ ）
 """
        self.config = config
        self.work_dir = Path(work_dir)

    def build_graphs(self, data: dict, kg_result=None) -> dict:
        """
 

 Args:
 data: KG （ data_loader）
 kg_result: ComplEx （ ， ）

 Returns:
 :
 'hetero_data': PyG HeteroData 
 'homo_data': PyG Data （PNA ）
 'deg': （PNA ）
 'node_mapping': → ID 
 'global_id_map': ID→ 
 'metadata': (node_types, edge_types)
 """
        if not check_pyg_available():
            return None

        # 
        cache_path = self.work_dir / 'pyg_graphs.pkl'
        if checkpoint_exists(str(cache_path)):
            print_info(" PyG ...")
            cached = load_checkpoint(str(cache_path))
            print_stat(" ", len(cached['metadata'][0]))
            print_stat(" ", len(cached['metadata'][1]))
            return cached

        print_info(" PyG ...")

        df = data['dataframe']
        entity_info = data['entity_info']

        # ---- 1: ----
        # { : { ID: }}
        node_type_map = defaultdict(dict)

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=" ", ncols=80):
            x_id, x_type = str(row['x_id']), str(row.get('x_type', 'entity'))
            y_id, y_type = str(row['y_id']), str(row.get('y_type', 'entity'))

            if x_id not in node_type_map[x_type]:
                node_type_map[x_type][x_id] = len(node_type_map[x_type])
            if y_id not in node_type_map[y_type]:
                node_type_map[y_type][y_id] = len(node_type_map[y_type])

        # 
        for ntype, nodes in node_type_map.items():
            print_stat(f"    '{ntype}'", f"{len(nodes):,}  ")

        # ---- 2: ----
        # {(src_type, relation, dst_type): (src_indices, dst_indices)}
        edge_dict = defaultdict(lambda: ([], []))

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=" ", ncols=80):
            x_id = str(row['x_id'])
            y_id = str(row['y_id'])
            x_type = str(row.get('x_type', 'entity'))
            y_type = str(row.get('y_type', 'entity'))
            rel = str(row['relation'])

            # : ( , , )
            edge_key = (x_type, rel, y_type)
            src_idx = node_type_map[x_type][x_id]
            dst_idx = node_type_map[y_type][y_id]

            edge_dict[edge_key][0].append(src_idx)
            edge_dict[edge_key][1].append(dst_idx)

        print_stat(" ", len(edge_dict))

        # ---- 3: HeteroData ----
        hetero_data = HeteroData()

        # （ ， ComplEx ）
        emb_dim = self.config['model']['embedding_dim']
        for ntype, nodes in node_type_map.items():
            num_nodes = len(nodes)
            hetero_data[ntype].x = torch.randn(num_nodes, emb_dim)
            hetero_data[ntype].num_nodes = num_nodes

        # 
        for edge_type, (src_list, dst_list) in edge_dict.items():
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            hetero_data[edge_type].edge_index = edge_index

        # 
        metadata = hetero_data.metadata()
        print_stat(" ", metadata[0])
        print_stat(" ", len(metadata[1]))

        # ---- 4: （PNA ）----
        print_info(" （ PNA ）...")

        # ID : ID → 
        global_id_map = {}
        global_idx = 0
        for ntype in sorted(node_type_map.keys()):
            for entity_id in node_type_map[ntype]:
                global_id_map[entity_id] = global_idx
                global_idx += 1

        total_nodes = global_idx
        print_stat(" ", f"{total_nodes:,}")

        # 
        homo_src, homo_dst = [], []
        type_offset = {}
        offset = 0
        for ntype in sorted(node_type_map.keys()):
            type_offset[ntype] = offset
            offset += len(node_type_map[ntype])

        for (src_type, rel, dst_type), (src_list, dst_list) in edge_dict.items():
            s_off = type_offset[src_type]
            d_off = type_offset[dst_type]
            for s, d in zip(src_list, dst_list):
                homo_src.append(s + s_off)
                homo_dst.append(d + d_off)

        homo_edge_index = torch.tensor([homo_src, homo_dst], dtype=torch.long)

        homo_data = Data(
            x=torch.randn(total_nodes, emb_dim),
            edge_index=homo_edge_index,
            num_nodes=total_nodes
        )

        total_edges = homo_edge_index.size(1)
        print_stat(" ", f"{total_edges:,}")

        # ---- 5: （PNA ）----
        print_info(" （PNA Scalers ）...")

        # 
        row = homo_edge_index[1]  # 
        deg_tensor = degree(row, num_nodes=total_nodes, dtype=torch.long)
        max_deg = int(deg_tensor.max().item())
        deg_hist = torch.zeros(max_deg + 1, dtype=torch.long)
        for d in deg_tensor:
            deg_hist[d.item()] += 1

        print_stat(" ", max_deg)
        print_stat(" ", f"{deg_tensor.float().mean():.2f}")

        # ---- 6: ComplEx ----
        if kg_result is not None:
            print_info(" ComplEx ...")
            self._init_features_from_kg(
                hetero_data, homo_data,
                kg_result, node_type_map, type_offset,
                emb_dim
            )

        # ---- ----
        result = {
            'hetero_data': hetero_data,
            'homo_data': homo_data,
            'deg': deg_hist,
            'node_mapping': dict(node_type_map),
            'type_offset': type_offset,
            'global_id_map': global_id_map,
            'metadata': metadata,
            'total_nodes': total_nodes,
        }

        save_checkpoint(result, str(cache_path))
        print_success(f"PyG  ")

        return result

    def _init_features_from_kg(self, hetero_data, homo_data,
                                kg_result, node_type_map, type_offset,
                                emb_dim):
        """
 ComplEx 

 Args:
 hetero_data: 
 homo_data: 
 kg_result: PyKEEN 
 node_type_map: { : { ID: }}
 type_offset: 
 emb_dim: 
 """
        entity_to_id = kg_result.training.entity_to_id

        with torch.no_grad():
            entity_repr = kg_result.model.entity_representations[0]
            # CPU
            all_emb = entity_repr(
                indices=torch.arange(
                    kg_result.model.num_entities,
                    device=next(kg_result.model.parameters()).device
                )
            ).cpu()
            # ComplEx ， ， 
            if torch.is_complex(all_emb):
                all_emb = all_emb.real
            all_emb = all_emb.float().contiguous()

        matched = 0
        total = 0

        for ntype in sorted(node_type_map.keys()):
            nodes = node_type_map[ntype]
            offset = type_offset[ntype]
            feat = torch.randn(len(nodes), emb_dim) * 0.01  # 

            for entity_id, local_idx in nodes.items():
                total += 1
                kg_idx = entity_to_id.get(entity_id)
                if kg_idx is not None and kg_idx < all_emb.size(0):
                    # ComplEx ， 
                    emb_vec = all_emb[kg_idx]
                    if emb_vec.shape[0] != emb_dim:
                        # ComplEx 2*emb_dim（ + ）
                        emb_vec = emb_vec[:emb_dim]
                    feat[local_idx] = emb_vec
                    # 
                    homo_data.x[local_idx + offset] = emb_vec
                    matched += 1

            hetero_data[ntype].x = feat

        print_stat(" ", f"{matched}/{total} ({100*matched/max(total,1):.1f}%)")

    def compute_target_degrees(self, data: dict, graph_result: dict) -> dict:
        """
 （ ）

 Args:
 data: KG 
 graph_result: build_graphs() 

 Returns:
 { ID: } 
 """
        homo_data = graph_result['homo_data']
        global_id_map = graph_result['global_id_map']
        edge_index = homo_data.edge_index

        # （ + ）
        num_nodes = homo_data.num_nodes
        in_deg = degree(edge_index[1], num_nodes=num_nodes)
        out_deg = degree(edge_index[0], num_nodes=num_nodes)
        total_deg = in_deg + out_deg

        # ID
        degree_map = {}
        for entity_id, global_idx in global_id_map.items():
            degree_map[entity_id] = int(total_deg[global_idx].item())

        return degree_map
