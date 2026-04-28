"""
immuneKG 

 :
 KG + + GNN （ ）
 kg_score min-max fusion_score 
 GSEA （Top-1000 ， immuneKG ）

 ： data_loader.get_prediction_disease_entities() ，
 scorer 。

 :
 1. ComplEx : ( , , ) KG 
 2. : 
 3. GNN : HeteroPNA-Attn 
 4. : 
 5. : Score_final = Score_combined / log(1 + Degree(target))
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from .model import FeatureFusionNetwork, MultiSourceFusionNetwork, EnhancedScorer
from .novelty import NoveltyScorer
from .data_loader import DISEASE_GENE_RELATIONS
from .utils import (
    print_banner, print_success, print_warning, print_info,
    print_stat, print_error, checkpoint_exists
)


class TargetScorer:
    """
 

 ，
 KG 。
 """

    def __init__(self, config: dict, device: torch.device, work_dir: str):
        self.config = config
        self.device = device
        self.work_dir = Path(work_dir)
        self.top_k = config['prediction']['top_k']
        self.alpha = 0.5  # KG 

    # ================================================================
    # 
    # ================================================================

    def score_targets(self,
                      kg_result,
                      data: dict,
                      feature_result: dict,
                      fusion_model,
                      target_diseases: dict,
                      target_entities: set,
                      output_path: str = None,
                      gnn_embeddings: np.ndarray = None,
                      degree_map: dict = None,
                      graph_result: dict = None) -> pd.DataFrame:
        """
 

 data_loader.get_prediction_disease_entities() ，
 ， （PrimeKG type ）。
 kg_score min-max ， fusion_score。
 """
        print_info(" ...")
        start_time = time.time()

        model = kg_result.model
        model.eval()
        training_triples = kg_result.training
        entity_to_id = training_triples.entity_to_id
        entity_info = data['entity_info']
        num_relations = training_triples.num_relations
        relation_id_to_label = getattr(training_triples, 'relation_id_to_label', None)

        # ---- （ ，ID data_loader ）----
        all_disease_ids = set()
        for keyword, disease_ids in target_diseases.items():
            all_disease_ids.update(disease_ids)
            print_stat(f"  '{keyword}'", f"{len(disease_ids)}  ")

        print_stat(" ", len(all_disease_ids))
        print_stat(" ", f"{len(target_entities):,}")

        # ---- ----
        valid_diseases = [d for d in all_disease_ids if d in entity_to_id]
        valid_targets = [t for t in target_entities if t in entity_to_id]

        print_stat(" ", len(valid_diseases))
        print_stat(" ", f"{len(valid_targets):,}")

        if not valid_diseases:
            print_error(" ！ 。")
            return pd.DataFrame()

        # ---- ----
        kg_embeddings = None
        disease_fused_embs = None

        if fusion_model is not None:
            print_info(" KG ...")
            kg_embeddings = self._extract_embeddings(model)
            disease_fused_embs = self._compute_fused_embeddings(
                model, fusion_model, feature_result, valid_diseases, entity_to_id,
                gnn_embeddings=gnn_embeddings, graph_result=graph_result
            )

        # GNN 
        gnn_target_embs = None
        gnn_disease_embs = None
        if gnn_embeddings is not None and graph_result is not None:
            print_info(" GNN ...")
            global_id_map = graph_result.get('global_id_map', {})
            gnn_target_embs = {}
            gnn_disease_embs = {}
            for t in valid_targets:
                gidx = global_id_map.get(str(t))
                if gidx is not None and gidx < len(gnn_embeddings):
                    gnn_target_embs[t] = gnn_embeddings[gidx]
            for d in valid_diseases:
                gidx = global_id_map.get(str(d))
                if gidx is not None and gidx < len(gnn_embeddings):
                    gnn_disease_embs[d] = gnn_embeddings[gidx]

        # ---- ----
        print_info(f"  {len(valid_targets):,}  ...")

        scores_list = []

        for target in tqdm(valid_targets, desc=" ", ncols=80, unit=" "):
            target_model_idx = entity_to_id[target]

            # (1) KG 
            kg_score = self._compute_kg_score(
                model=model,
                disease_ids=valid_diseases,
                target_id=target,
                entity_to_id=entity_to_id,
                num_relations=num_relations,
                relation_id_to_label=relation_id_to_label,
            )

            # (2) 
            fusion_score = 0.0
            if disease_fused_embs is not None and kg_embeddings is not None:
                target_emb = torch.from_numpy(
                    kg_embeddings[target_model_idx:target_model_idx+1]
                ).float().to(self.device)
                target_emb_norm = F.normalize(target_emb, p=2, dim=-1)

                for i, disease_id in enumerate(valid_diseases):
                    if i < len(disease_fused_embs):
                        fused_emb = disease_fused_embs[i:i+1]
                        fused_norm = F.normalize(fused_emb, p=2, dim=-1)
                        sim = torch.mm(fused_norm, target_emb_norm.t()).item()
                        fusion_score += sim
                fusion_score /= max(len(valid_diseases), 1)

            # (2.5) GNN 
            gnn_score = 0.0
            if gnn_target_embs is not None and gnn_disease_embs:
                t_gnn = gnn_target_embs.get(target)
                if t_gnn is not None:
                    t_vec = torch.from_numpy(t_gnn.astype(np.float32)).unsqueeze(0).to(self.device)
                    t_vec = F.normalize(t_vec, p=2, dim=-1)
                    gnn_sims = []
                    for d_id in valid_diseases:
                        d_gnn = gnn_disease_embs.get(d_id)
                        if d_gnn is not None:
                            d_vec = torch.from_numpy(d_gnn.astype(np.float32)).unsqueeze(0).to(self.device)
                            d_vec = F.normalize(d_vec, p=2, dim=-1)
                            gnn_sims.append(torch.mm(d_vec, t_vec.t()).item())
                    if gnn_sims:
                        gnn_score = np.mean(gnn_sims)

            # (3) 
            has_fusion = fusion_model is not None
            has_gnn = gnn_embeddings is not None

            if has_fusion and has_gnn:
                combined_score = 0.4 * kg_score + 0.3 * fusion_score + 0.3 * gnn_score
            elif has_fusion:
                combined_score = self.alpha * kg_score + (1.0 - self.alpha) * fusion_score
            elif has_gnn:
                combined_score = 0.6 * kg_score + 0.4 * gnn_score
            else:
                combined_score = kg_score

            # 
            per_disease_scores = {}
            for keyword, disease_ids in target_diseases.items():
                kw_diseases = [d for d in disease_ids if d in entity_to_id]
                if kw_diseases:
                    kw_score = self._compute_kg_score(
                        model=model,
                        disease_ids=kw_diseases,
                        target_id=target,
                        entity_to_id=entity_to_id,
                        num_relations=num_relations,
                        relation_id_to_label=relation_id_to_label,
                    )
                    per_disease_scores[f"score_{keyword.replace(' ', '_')}"] = kw_score

            target_name = entity_info.get(target, {}).get('name', '')

            entry = {
                'target_id': target,
                'target_name': target_name,
                'combined_score': combined_score,
                'kg_score': kg_score,
                'fusion_score': fusion_score,
                'gnn_score': gnn_score,
                'num_diseases': len(valid_diseases),
                **per_disease_scores
            }
            scores_list.append(entry)

        # ---- DataFrame ----
        if not scores_list:
            print_error(" ！")
            return pd.DataFrame()

        scores_df = pd.DataFrame(scores_list)

        # ---- 【 】 kg_score min-max ----
        # kg_score ComplEx score_hrt ， fusion_score（ ）
        # combined_score 
        scores_df = self._normalize_scores(scores_df)

        # 
        scores_df = scores_df.sort_values('combined_score', ascending=False)
        scores_df.insert(0, 'rank', range(1, len(scores_df) + 1))
        scores_df = scores_df.reset_index(drop=True)

        # ---- ----
        if degree_map is not None:
            print_info(" ...")
            novelty_scorer = NoveltyScorer(self.config)
            scores_df = novelty_scorer.apply_novelty_scores(
                scores_df, degree_map, score_column='combined_score'
            )
            ap_metrics = novelty_scorer.compute_average_popularity(
                scores_df, degree_map, top_k=self.top_k
            )
            novelty_scorer.print_novelty_ranking(scores_df, top_k=20)

        # ---- ----
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scores_df.to_csv(output_path, index=False)
            print_success(f" : {output_path}")

            # 【 】 GSEA （ Top-1000）
            gsea_path = output_path.parent / 'gsea_input_top1000.rnk'
            self._save_gsea_input(scores_df, gsea_path, top_n=1000)

        duration = time.time() - start_time
        print_success(f" ！  {duration:.2f}  ，  {len(scores_df)}  ")

        self._print_top_targets(scores_df)

        return scores_df

    # ================================================================
    # 【 】 GSEA 
    # ================================================================

    def _save_gsea_input(self, scores_df: pd.DataFrame, output_path: Path,
                          top_n: int = 1000):
        """
 GSEA / preranked_gsea 

 : ， ，tab 
 

 ：
 - top_n （ immuneKG ， ）
 - combined_score_norm 
 - （ 、 ）

 Args:
 scores_df: DataFrame
 output_path: （.rnk ）
 top_n: （ 500-2000）
 """
        # ： 
        score_col = 'combined_score_norm' if 'combined_score_norm' in scores_df.columns \
                    else 'combined_score'

        gsea_df = scores_df[
            scores_df['target_name'].apply(self._is_valid_gene_name)
        ][['target_name', score_col]].copy()

        # （ ）
        gsea_df = gsea_df.drop_duplicates(subset='target_name', keep='first')

        # top_n
        gsea_df = gsea_df.head(top_n)

        print_stat(f"GSEA  （Top-{top_n}）", len(gsea_df))

        # .rnk （tab ， ）
        gsea_df.to_csv(output_path, sep='\t', index=False, header=False)
        print_success(f"GSEA  : {output_path.name}")

        # （ ） 
        full_path = output_path.parent / 'gsea_input_full.rnk'
        scores_df[
            scores_df['target_name'].apply(self._is_valid_gene_name)
        ][['target_name', score_col]].drop_duplicates(
            subset='target_name', keep='first'
        ).to_csv(full_path, sep='\t', index=False, header=False)
        print_info(f"  GSEA  （ ）: {full_path.name}")

    @staticmethod
    def _is_valid_gene_name(name: str) -> bool:
        """
 
 ： 、 （ ）、 、 
 """
        if not name or not isinstance(name, str):
            return False
        name = name.strip()
        if not name:
            return False
        # 
        if ' ' in name and len(name) > 20:
            return False
        # 
        if name.isdigit():
            return False
        # （1 ） （>20 ） 
        if len(name) < 2 or len(name) > 20:
            return False
        return True

    # ================================================================
    # 【 】 
    # ================================================================

    def _normalize_scores(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
 min-max 

 kg_score （ComplEx score_hrt ）
 fusion_score（ ，-1~1）， combined_score
 kg_score 。

 ，combined_score ， 。

 :
 kg_score_norm: KG [0, 1]
 fusion_score_norm: [0, 1]
 combined_score_norm: 
 """
        def minmax(series):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - mn) / (mx - mn)

        scores_df['kg_score_norm'] = minmax(scores_df['kg_score'])
        scores_df['fusion_score_norm'] = minmax(scores_df['fusion_score'])

        has_gnn = 'gnn_score' in scores_df.columns and \
                  scores_df['gnn_score'].abs().sum() > 0

        if has_gnn:
            scores_df['gnn_score_norm'] = minmax(scores_df['gnn_score'])
            scores_df['combined_score_norm'] = (
                0.4 * scores_df['kg_score_norm'] +
                0.3 * scores_df['fusion_score_norm'] +
                0.3 * scores_df['gnn_score_norm']
            )
        elif scores_df['fusion_score'].abs().sum() > 0:
            scores_df['combined_score_norm'] = (
                self.alpha * scores_df['kg_score_norm'] +
                (1 - self.alpha) * scores_df['fusion_score_norm']
            )
        else:
            scores_df['combined_score_norm'] = scores_df['kg_score_norm']

        # （ combined_score ）
        # ：rank ， combined_score
        scores_df['combined_score'] = scores_df['combined_score_norm']

        print_stat("combined_score_norm ",
                   f"[{scores_df['combined_score_norm'].min():.4f}, "
                   f"{scores_df['combined_score_norm'].max():.4f}]")

        return scores_df

    # ================================================================
    # KG 
    # ================================================================

    def _compute_kg_score(
        self,
        model,
        disease_ids: list,
        target_id: str,
        entity_to_id: dict,
        num_relations: int,
        relation_id_to_label=None,
    ) -> float:
        """
 immuneKG disease↔gene （P/U/D/ML/...） gene → disease。
 disease head、target tail ， 。
 ：
 - DISEASE_GENE_RELATIONS 
 - (target=head, disease=tail) 
 """
        target_idx = entity_to_id[target_id]
        all_scores = []

        # 
        if relation_id_to_label:
            rel_indices = [
                rid for rid in range(num_relations)
                if relation_id_to_label.get(rid) in DISEASE_GENE_RELATIONS
            ]
        else:
            # ： label （ gene→disease）
            rel_indices = list(range(num_relations))

        for disease_id in disease_ids:
            disease_idx = entity_to_id.get(disease_id)
            if disease_idx is None:
                continue
            for rel_idx in rel_indices:
                with torch.no_grad():
                    batch = torch.tensor(
                        [[target_idx, rel_idx, disease_idx]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    score = model.score_hrt(batch).cpu().item()
                    all_scores.append(score)

        return float(np.mean(all_scores)) if all_scores else 0.0

    # ================================================================
    # 
    # ================================================================

    def _extract_embeddings(self, model) -> np.ndarray:
        with torch.no_grad():
            entity_repr = model.entity_representations[0]
            emb = entity_repr(
                indices=torch.arange(model.num_entities, device=self.device)
            ).cpu().numpy()
        return emb

    def _compute_fused_embeddings(self, kg_model, fusion_model,
                                   feature_result: dict,
                                   disease_ids: list,
                                   entity_to_id: dict,
                                   gnn_embeddings: np.ndarray = None,
                                   graph_result: dict = None) -> Optional[torch.Tensor]:
        kg_to_row = feature_result['kg_to_row']
        feature_matrix = feature_result['feature_matrix']

        disease_kg_embs = []
        disease_features = []
        valid_count = 0

        with torch.no_grad():
            entity_repr = kg_model.entity_representations[0]

            for disease_id in disease_ids:
                kg_idx = entity_to_id.get(disease_id)
                feat_row = kg_to_row.get(disease_id)

                if kg_idx is not None:
                    emb = entity_repr(
                        indices=torch.tensor([kg_idx], device=self.device)
                    ).cpu().numpy()[0]
                    disease_kg_embs.append(emb)

                    if feat_row is not None:
                        disease_features.append(feature_matrix[feat_row])
                        valid_count += 1
                    else:
                        disease_features.append(
                            np.zeros(feature_matrix.shape[1], dtype=np.float32)
                        )

        if not disease_kg_embs:
            return None

        print_info(f"   : {valid_count}/{len(disease_ids)}  ")

        kg_emb_tensor = torch.from_numpy(
            np.array(disease_kg_embs, dtype=np.float32)
        ).to(self.device)

        feat_tensor = torch.from_numpy(
            np.array(disease_features, dtype=np.float32)
        ).to(self.device)

        fusion_model.eval()
        with torch.no_grad():
            if (isinstance(fusion_model, MultiSourceFusionNetwork) and
                    gnn_embeddings is not None and graph_result is not None):
                global_id_map = graph_result.get('global_id_map', {})
                gnn_emb_list = []
                for d_id in disease_ids:
                    gidx = global_id_map.get(str(d_id))
                    if gidx is not None and gidx < len(gnn_embeddings):
                        gnn_emb_list.append(gnn_embeddings[gidx])
                    else:
                        gnn_emb_list.append(np.zeros(gnn_embeddings.shape[1], dtype=np.float32))
                gnn_tensor = torch.from_numpy(
                    np.array(gnn_emb_list, dtype=np.float32)
                ).to(self.device)
                fused = fusion_model(kg_emb_tensor, feat_tensor, gnn_tensor)
            else:
                fused = fusion_model(kg_emb_tensor, feat_tensor)

        return fused

    # ================================================================
    # Top 
    # ================================================================

    def _print_top_targets(self, scores_df: pd.DataFrame):
        k = min(self.top_k, len(scores_df))
        has_gnn = 'gnn_score' in scores_df.columns and \
                  scores_df['gnn_score'].abs().sum() > 0

        print_info(f"═══ Top {k}   ═══")
        header = (f"\n  {'Rank':>5} │ {' ':<40} │ {' ':>10} │ "
                  f"{'KG ( )':>10} │ {' ':>10}")
        if has_gnn:
            header += f" │ {'GNN ':>10}"
        print(header)

        for _, row in scores_df.head(k).iterrows():
            name = str(row['target_name'])[:38]
            line = (f"  {row['rank']:5d} │ {name:<40} │ "
                    f"{row['combined_score']:10.6f} │ "
                    f"{row['kg_score']:10.4f} │ "
                    f"{row.get('fusion_score', 0):10.6f}")
            if has_gnn:
                line += f" │ {row.get('gnn_score', 0):10.6f}"
            print(line)



# ============================================================
# （ ）
# ============================================================

class PredictionReportGenerator:
    @staticmethod
    def generate_per_disease_report(scores_df: pd.DataFrame,
                                     target_diseases: dict,
                                     entity_info: dict,
                                     output_dir: str,
                                     top_k: int = 50) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = {}

        for keyword in target_diseases.keys():
            col_name = f"score_{keyword.replace(' ', '_')}"

            if col_name in scores_df.columns:
                disease_df = scores_df.copy()
                disease_df = disease_df.sort_values(col_name, ascending=False)
                disease_df.insert(0, f'{keyword}_rank',
                                 range(1, len(disease_df) + 1))

                # （ “ Top-K， ”）
                safe_name = keyword.replace(' ', '_')
                fullpath = output_dir / f"targets_{safe_name}_full.csv"
                disease_df.to_csv(fullpath, index=False)

                disease_top = disease_df.head(top_k)

                filepath = output_dir / f"targets_{safe_name}_top{top_k}.csv"
                disease_top.to_csv(filepath, index=False)

                reports[keyword] = disease_top

                print_info(f"  '{keyword}' Top-5  :")
                for _, row in disease_top.head(5).iterrows():
                    name = row.get('target_name', '')
                    score = row.get(col_name, 0)
                    print(f"    {row.get(f'{keyword}_rank', '?'):>3}. "
                          f"{name:<35} ({score:.6f})")

                print_success(f"  →  : {filepath.name}")
                print_success(f"  →  : {fullpath.name}")

        return reports