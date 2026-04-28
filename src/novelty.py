"""
immuneKG 

 ：
 ， （Degree） 。
 （ ） ， 。

 ：
 Score_final = Score_model(u, v) / log(1 + Degree(v))

 v 。 ， ，
 " " 。

 ：
 - AP (Average Popularity): 
 AP = (1/K) * Σ log(1 + degree(v_i)) for top-K targets
 AP ， 

 - Coverage@K: 
 - Serendipity: 
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .utils import print_info, print_stat, print_success, print_warning


class NoveltyScorer:
    """
 

 :
 1. 
 2. 
 3. AP (Average Popularity) 
 4. 
 """

    def __init__(self, config: dict):
        """
 Args:
 config: 
 """
        novelty_cfg = config.get('novelty', {})
        self.enable = novelty_cfg.get('enable', True)
        self.penalty_weight = novelty_cfg.get('penalty_weight', 1.0)
        self.degree_log_base = novelty_cfg.get('degree_log_base', 1)  # log(base + degree)
        self.min_degree = novelty_cfg.get('min_degree_threshold', 0)  # 

    # ================================================================
    # : Score_final = Score_model / log(1 + Degree)
    # ================================================================

    def compute_novelty_penalty(self, degree: int) -> float:
        """
 

 : penalty = 1 / log(base + Degree(v))

 Degree = 0 , penalty = 1/log(base) → （ ）
 Degree , penalty → 0（ ）

 Args:
 degree: 

 Returns:
 （0~1 ， ）
 """
        base = self.degree_log_base  # 1
        return 1.0 / np.log(base + 1 + max(degree, 0))

    def apply_novelty_scores(self,
                              scores_df: pd.DataFrame,
                              degree_map: Dict[str, int],
                              score_column: str = 'combined_score') -> pd.DataFrame:
        """
 DataFrame

 :
 - target_degree: 
 - novelty_penalty: 
 - novelty_score: = combined_score / log(1 + degree)
 - novelty_rank: 

 Args:
 scores_df: DataFrame
 degree_map: { ID: } 
 score_column: 

 Returns:
 DataFrame（ ）
 """
        if not self.enable:
            print_warning(" ")
            return scores_df

        print_info(" ...")

        df = scores_df.copy()

        # ---- ----
        df['target_degree'] = df['target_id'].apply(
            lambda tid: degree_map.get(str(tid), 0)
        )

        # ---- ----
        df['novelty_penalty'] = df['target_degree'].apply(self.compute_novelty_penalty)

        # ---- ----
        # Score_final = Score_model(u, v) / log(1 + Degree(v))
        # Score_model * novelty_penalty * penalty_weight
        if score_column in df.columns:
            df['novelty_score'] = (
                df[score_column] * df['novelty_penalty'] * self.penalty_weight
            )
        else:
            print_warning(f"  '{score_column}'  ，  combined_score")
            df['novelty_score'] = (
                df.get('combined_score', 0) * df['novelty_penalty'] * self.penalty_weight
            )

        # ---- ----
        df = df.sort_values('novelty_score', ascending=False)
        df['novelty_rank'] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)

        # ---- ----
        print_stat(" ", f"[{df['target_degree'].min()}, {df['target_degree'].max()}]")
        print_stat(" ", f"{df['target_degree'].mean():.1f}")
        print_stat(" ", f"[{df['novelty_penalty'].min():.4f}, "
                                   f"{df['novelty_penalty'].max():.4f}]")

        return df

    # ================================================================
    # AP (Average Popularity) 
    # ================================================================

    def compute_average_popularity(self,
                                     scores_df: pd.DataFrame,
                                     degree_map: Dict[str, int],
                                     top_k: int = 50) -> dict:
        """
 Average Popularity (AP) 

 AP " "。
 AP ， 。

 : AP@K = (1/K) * Σ_{i=1}^{K} log(1 + degree(v_i))

 Args:
 scores_df: DataFrame
 degree_map: 
 top_k: Top-K AP

 Returns:
 AP 
 """
        k = min(top_k, len(scores_df))

        # --- AP ---
        if 'rank' in scores_df.columns:
            original_top = scores_df.nsmallest(k, 'rank')
        else:
            original_top = scores_df.head(k)

        orig_degrees = original_top['target_id'].apply(
            lambda tid: degree_map.get(str(tid), 0)
        )
        ap_original = orig_degrees.apply(lambda d: np.log(1 + d)).mean()

        # --- AP ---
        ap_novelty = None
        if 'novelty_rank' in scores_df.columns:
            novelty_top = scores_df.nsmallest(k, 'novelty_rank')
            nov_degrees = novelty_top['target_id'].apply(
                lambda tid: degree_map.get(str(tid), 0)
            )
            ap_novelty = nov_degrees.apply(lambda d: np.log(1 + d)).mean()

        # --- ---
        # ( < )
        median_degree = np.median(list(degree_map.values()))
        low_deg_original = (orig_degrees < median_degree).sum() / k
        low_deg_novelty = None
        if 'novelty_rank' in scores_df.columns:
            low_deg_novelty = (nov_degrees < median_degree).sum() / k

        result = {
            'ap_original': float(ap_original),
            'ap_novelty': float(ap_novelty) if ap_novelty is not None else None,
            'ap_improvement': (
                float(ap_original - ap_novelty) if ap_novelty is not None else None
            ),
            'low_degree_ratio_original': float(low_deg_original),
            'low_degree_ratio_novelty': float(low_deg_novelty) if low_deg_novelty is not None else None,
            'median_degree': float(median_degree),
            'top_k': k,
        }

        # 
        print_info(f"═══   (Top-{k}) ═══")
        print_stat("AP ( )", f"{ap_original:.4f} ( )")
        if ap_novelty is not None:
            print_stat("AP ( )", f"{ap_novelty:.4f}")
            print_stat("AP ", f"{ap_original - ap_novelty:.4f}")
        print_stat(" ( )", f"{low_deg_original:.1%}")
        if low_deg_novelty is not None:
            print_stat(" ( )", f"{low_deg_novelty:.1%}")

        return result

    # ================================================================
    # 
    # ================================================================

    def print_novelty_ranking(self, scores_df: pd.DataFrame, top_k: int = 20):
        """
 

 Args:
 scores_df: DataFrame
 top_k: 
 """
        if 'novelty_rank' not in scores_df.columns:
            print_warning("DataFrame ")
            return

        k = min(top_k, len(scores_df))
        top = scores_df.nsmallest(k, 'novelty_rank')

        print_info(f"\n═══   Top-{k}   ═══")
        print(f"\n  {'N.Rank':>6} │ {' ':<35} │ {' ':>10} │ "
              f"{' ':>10} │ {' ':>6} │ {' ':>8} │ {' ':>6}")
        print(f"  {'─'*6}─┼─{'─'*35}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*6}")

        for _, row in top.iterrows():
            name = str(row.get('target_name', ''))[:33]
            orig_rank = row.get('rank', '?')
            print(f"  {row['novelty_rank']:6d} │ {name:<35} │ "
                  f"{row['novelty_score']:10.6f} │ "
                  f"{row.get('combined_score', 0):10.6f} │ "
                  f"{row['target_degree']:6d} │ "
                  f"{row['novelty_penalty']:8.4f} │ "
                  f"{orig_rank:>6}")

