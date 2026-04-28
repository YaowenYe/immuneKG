#!/usr/bin/env python3
"""
GSEA — immuneKG 

 :
 python gsea_enrichment.py # 
 python gsea_enrichment.py --scores results/target_scores.csv
 python gsea_enrichment.py --top-n 1000 # Top-N 
 python gsea_enrichment.py --geneset data/ibd_genes.gmt

 :
 - target_scores.csv: 
 - IBD （GMT txt ）

 :
 - results/gsea_input_top1000.rnk GSEA 
 - results/gsea_enrichment_result.csv 
 - results/gsea_plot.png （ immuneKG 2）
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# IBD （ DisGeNET + OMIM + ClinVar IBD ）
# immuneKG Enrichment Score
# ============================================================

IBD_KNOWN_GENES = {
    # （NOD2/IL23R/ATG16L1 Crohn's GWAS ）
    "NOD2", "IL23R", "ATG16L1", "IRGM", "IL10", "IL10RA", "IL10RB",
    "CARD9", "STAT3", "JAK2", "TYK2", "IL6", "TNF", "TNFRSF1A",
    # 
    "MUC1", "MUC2", "MUC5B", "CDH1", "CLDN1", "CLDN7", "TJP1",
    # 
    "TLR4", "TLR2", "TLR9", "NOD1", "NLRP3", "CASP1",
    # UC 
    "HLA-DRA", "HLA-DRB1", "LAMB1", "ECM1",
    # 
    "IL2", "IL7R", "IL12B", "IL18", "IFNG", "TNFSF15",
    # IBD （ ）
    "ITGA4", "ITGB7",  # vedolizumab 
    "IL12B", "IL23A",  # ustekinumab 
    "JAK1", "JAK3",    # tofacitinib 
    "TNF",             # infliximab/adalimumab 
    "IL6ST",           # 6-CSI 
    # 
    "LRRK2", "PARK7", "PINK1",
    # GWAS 
    "PTPN22", "PTPN2", "SMAD3", "NKX2-3", "CDKAL1",
    "GPR35", "FCGR2A", "BST1", "LSP1",
    # Crohn 
    "IL27", "ICOSLG", "TNFSF8", "REL", "NFKB1",
    "MST1", "RIPK2", "XIAP",
    # 
    "CYLD", "LUBAC", "HOIL1L", "SHARPIN",
    "SLC22A4", "SLC22A5",
    "IL17A", "IL17F", "IL21", "IL22", "IL33", "ST2",
    "S100A8", "S100A9", "S100A12",
    "MMP3", "MMP9", "MMP12",
    # IBD （ ）
    "CD59", "FADD", "CFLAR", "SQSTM1", "HDAC3",
    "EZH1", "EZH2", "CDK4", "CDK6", "CFL1",
    "SIGMAR1", "IQGAP1",
}


def load_scores(scores_path: str) -> pd.DataFrame:
    """ """
    df = pd.read_csv(scores_path)
    print(f"  →   {len(df)}  ")

    # 
    if 'combined_score_norm' in df.columns:
        score_col = 'combined_score_norm'
        print(f"  →   (combined_score_norm)")
    elif 'combined_score' in df.columns:
        score_col = 'combined_score'
        print(f"  →   (combined_score)")
    else:
        raise ValueError(" ！")

    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return df, score_col


def load_geneset(geneset_path: str = None) -> set:
    """
 

 :
 - GMT （ : gene_set_name \\t description \\t gene1 \\t gene2 ...）
 - TXT （ ）
 - None： IBD 
 """
    if geneset_path is None:
        print(f"  →   IBD   ({len(IBD_KNOWN_GENES)}  )")
        return IBD_KNOWN_GENES

    path = Path(geneset_path)
    if not path.exists():
        print(f"  ⚠  : {geneset_path}， ")
        return IBD_KNOWN_GENES

    genes = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                # GMT 
                genes.update(g.strip().upper() for g in parts[2:] if g.strip())
            else:
                # TXT 
                genes.add(line.upper())

    print(f"  →   {geneset_path}   {len(genes)}  ")
    return genes


def filter_genes(df: pd.DataFrame, score_col: str,
                 top_n: int = None) -> pd.DataFrame:
    """
 

 Args:
 df: DataFrame
 score_col: 
 top_n: Top-N（None ， 1000）
 """
    # 
    def is_valid(name):
        if not name or not isinstance(name, str):
            return False
        name = name.strip()
        if not name or name.isdigit() or len(name) < 2:
            return False
        # 
        if len(name) > 25 and ' ' in name:
            return False
        return True

    df = df[df['target_name'].apply(is_valid)].copy()
    df = df.drop_duplicates(subset='target_name', keep='first')

    print(f"  →  : {len(df)}  ")

    if top_n is not None:
        df = df.head(top_n)
        print(f"  →   Top-{top_n}")

    return df


def compute_gsea_es(ranked_genes: list, gene_scores: np.ndarray,
                    gene_set: set) -> tuple:
    """
 GSEA Enrichment Score（ES）

 GSEA ：
 - +sqrt((N-NH)/NH * |score|)
 - -1/(N-NH)

 Args:
 ranked_genes: 
 gene_scores: （ ， ）
 gene_set: （ ）

 Returns:
 (es, peak_position, running_es, hits_mask)
 """
    N = len(ranked_genes)
    NH = sum(1 for g in ranked_genes if g.upper() in gene_set)

    if NH == 0:
        print(" ⚠ ！ES = 0")
        return 0.0, 0, np.zeros(N), np.zeros(N, dtype=bool)

    print(f"  →  : {NH} / {N} ({100*NH/N:.1f}%)")

    # / 
    # （ GSEA |score|^p，p=1）
    abs_scores = np.abs(gene_scores)
    sum_hit_scores = sum(
        abs_scores[i] for i, g in enumerate(ranked_genes)
        if g.upper() in gene_set
    )

    running_es = []
    hits_mask = []
    cumulative = 0.0
    miss_penalty = 1.0 / (N - NH)

    for i, gene in enumerate(ranked_genes):
        is_hit = gene.upper() in gene_set
        if is_hit:
            # ： 
            w = abs_scores[i] / sum_hit_scores if sum_hit_scores > 0 else 1.0 / NH
            cumulative += w
        else:
            cumulative -= miss_penalty
        running_es.append(cumulative)
        hits_mask.append(is_hit)

    running_es = np.array(running_es)
    hits_mask = np.array(hits_mask, dtype=bool)

    # ES = 
    if abs(running_es.max()) >= abs(running_es.min()):
        es = running_es.max()
        peak_pos = running_es.argmax()
    else:
        es = running_es.min()
        peak_pos = running_es.argmin()

    # leading edge
    leading_edge_n = hits_mask[:peak_pos+1].sum()

    return es, peak_pos, running_es, hits_mask, leading_edge_n


def compute_permutation_pvalue(ranked_genes: list, gene_scores: np.ndarray,
                                gene_set: set, observed_es: float,
                                n_perm: int = 1000) -> float:
    """ p-value"""
    perm_es = []
    rng = np.random.RandomState(42)
    NH = sum(1 for g in ranked_genes if g.upper() in gene_set)
    N = len(ranked_genes)

    for _ in range(n_perm):
        perm_set = set(rng.choice(ranked_genes, size=NH, replace=False))
        es_perm, _, _, _, _ = compute_gsea_es(
            ranked_genes, gene_scores, perm_set
        )
        perm_es.append(es_perm)

    perm_es = np.array(perm_es)
    if observed_es >= 0:
        pvalue = (perm_es >= observed_es).mean()
    else:
        pvalue = (perm_es <= observed_es).mean()

    return max(pvalue, 1.0 / n_perm)  # p = 1/n_perm


def plot_enrichment(ranked_genes: list, gene_scores: np.ndarray,
                    running_es: np.ndarray, hits_mask: np.ndarray,
                    es: float, peak_pos: int, leading_edge_n: int,
                    pvalue: float, n_geneset: int,
                    output_path: str, title: str = "IBD Enrichment"):
    """
 GSEA （ immuneKG 2 ）
 """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 0.5, 5]},
        sharex=True
    )
    fig.patch.set_facecolor('white')

    N = len(ranked_genes)
    x = np.arange(N)

    # ---- ： ----
    ax1.fill_between(x, 0, gene_scores,
                     where=(gene_scores >= 0), color='#d62728', alpha=0.8)
    ax1.fill_between(x, 0, gene_scores,
                     where=(gene_scores < 0), color='#1f77b4', alpha=0.8)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Input gene score', fontsize=10)
    ax1.set_title(
        f'Ranked genes (in a decreasing order) with {n_geneset} in geneset',
        fontsize=11
    )
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- ： ----
    hit_positions = np.where(hits_mask)[0]
    ax2.vlines(hit_positions, 0, 1, color='#d62728', linewidth=0.8, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # ---- ：Running ES ----
    ax3.fill_between(x, 0, running_es, alpha=0.3, color='#d62728')
    ax3.plot(x, running_es, color='black', linewidth=1.5)
    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    # peak
    ax3.axvline(x=peak_pos, color='blue', linewidth=1.5, linestyle='--', alpha=0.7)
    ax3.scatter([peak_pos], [es], color='blue', s=60, zorder=5)

    # 
    info_x = peak_pos + N * 0.02
    info_y = es * 0.85
    info_text = (
        f"Peak at rank={peak_pos}\n"
        f"Leading edge number={leading_edge_n}\n"
        f"p-value: ~{pvalue:.0e}\n"
        f"adjusted p-value: ~{pvalue:.0e}"
    )
    ax3.annotate(
        info_text,
        xy=(peak_pos, es),
        xytext=(info_x, info_y),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='gray')
    )

    # ES 
    ax3.text(0.02, 0.92, f'ES = {es:.3f}',
             transform=ax3.transAxes,
             fontsize=12, fontweight='bold', color='black')

    ax3.set_xlabel(f'Ranked genes (in a decreasing order)', fontsize=10)
    ax3.set_ylabel('Running enrichment score', fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  →  : {output_path}")


def save_rnk(df: pd.DataFrame, score_col: str, output_path: str):
    """ .rnk （ GSEA ）"""
    rnk_df = df[['target_name', score_col]].copy()
    rnk_df.columns = ['Gene', 'Score']
    rnk_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"  → .rnk  : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='IBD GSEA ')
    parser.add_argument('--scores', type=str, default='results/target_scores.csv',
                        help=' ( : results/target_scores.csv)')
    parser.add_argument('--geneset', type=str, default=None,
                        help=' (GMT/TXT， IBD )')
    parser.add_argument('--top-n', type=int, default=1000,
                        help=' Top-N ( : 1000， immuneKG)')
    parser.add_argument('--no-truncate', action='store_true',
                        help=' （ ， ES）')
    parser.add_argument('--n-perm', type=int, default=1000,
                        help=' ( : 1000)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help=' ( : results)')
    parser.add_argument('--title', type=str, default='IBD Target Enrichment',
                        help=' ')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n══ GSEA (IBD) ══\n")

    # ---- ----
    print("1. ...")
    if not Path(args.scores).exists():
        print(f"  ✗  : {args.scores}")
        sys.exit(1)
    df, score_col = load_scores(args.scores)

    print("\n2. ...")
    gene_set = load_geneset(args.geneset)

    print("\n3. ...")
    top_n = None if args.no_truncate else args.top_n
    filtered_df = filter_genes(df, score_col, top_n=top_n)

    # .rnk 
    rnk_path = output_dir / f'gsea_input_top{top_n or "all"}.rnk'
    save_rnk(filtered_df, score_col, str(rnk_path))

    # ---- ES ----
    print("\n4. Enrichment Score...")
    ranked_genes = filtered_df['target_name'].tolist()
    gene_scores = filtered_df[score_col].values.astype(float)

    es, peak_pos, running_es, hits_mask, leading_edge_n = compute_gsea_es(
        ranked_genes, gene_scores, gene_set
    )
    print(f"  → ES = {es:.4f}")
    print(f"  → Peak at rank = {peak_pos}")
    print(f"  → Leading edge genes = {leading_edge_n}")

    # ---- ----
    print(f"\n5.   (n_perm={args.n_perm})...")
    pvalue = compute_permutation_pvalue(
        ranked_genes, gene_scores, gene_set, es,
        n_perm=args.n_perm
    )
    print(f"  → p-value = {pvalue:.2e}")
    print(f"  → adjusted p-value ≈ {pvalue:.2e}")

    # ---- ----
    print("\n6. ...")
    n_geneset = len(gene_set & {g.upper() for g in ranked_genes})
    plot_path = output_dir / 'gsea_enrichment_plot.png'
    plot_enrichment(
        ranked_genes, gene_scores, running_es, hits_mask,
        es, peak_pos, leading_edge_n, pvalue, n_geneset,
        str(plot_path), title=args.title
    )

    # ---- ----
    result_df = pd.DataFrame({
        'metric': ['ES', 'peak_rank', 'leading_edge_n', 'p_value', 'adj_p_value',
                   'n_genes_ranked', 'n_geneset_hits', 'top_n_used'],
        'value': [f'{es:.4f}', peak_pos, leading_edge_n,
                  f'{pvalue:.2e}', f'{pvalue:.2e}',
                  len(ranked_genes), n_geneset, top_n or 'all']
    })
    result_path = output_dir / 'gsea_enrichment_result.csv'
    result_df.to_csv(result_path, index=False)
    print(f"\n  →  : {result_path}")

    # ---- ----
    print(f"""
══   ══

  ES           = {es:.4f}   (immuneKG IBD = 0.705)
  Peak rank    = {peak_pos}
  Leading edge = {leading_edge_n}  
  p-value      = {pvalue:.2e}
       = {len(ranked_genes)} (Top-{top_n or 'all'})
      = {n_geneset}  

   :   ES < 0.5， :
    1.   target_scores.csv   scorer.py  
    2.   --top-n（  500~1000）
    3.  （--geneset  ）
""")


if __name__ == '__main__':
    main()
