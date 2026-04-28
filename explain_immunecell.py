#!/usr/bin/env python3
"""
immuneKG — Immune Cell Interpretability Analysis
=================================================
Quantifies and visualises the structural contribution of immune_cell nodes
to target scoring predictions.

Two complementary bridging paths are analysed:

  Path A  (IcDv bridge):  Gene ──IcDv──> ImmuneCell
    A gene ranked highly for a disease may also drive the differentiation
    of an immune cell subtype.  The cell is "implicated" via the shared gene.

  Path B  (IcE bridge):  ImmuneCell ──IcE──> Gene
    A high-ranking gene may be a marker of an immune cell subtype.
    The cell is "implicated" because its defining markers overlap with
    the predicted target set.

Neither path is a causal link prediction — they are post-hoc structural
explanations of *why* certain genes rank highly, surfacing the immune
cell biology encoded in the graph topology.

Usage:
  python explain_immunecell.py \\
      --disease "inflammatory bowel diseases" "colitis" \\
      --results results/targets_inflammatory_bowel_full.csv \\
      --top-k 50 \\
      --output results/immunecell_interpretability/

  python explain_immunecell.py --disease "rheumatoid arthritis" --top-k 100
"""

import argparse
import sys
import pickle
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_device, load_checkpoint, print_info, print_stat


# ── Cell lineage colours ──────────────────────────────────────────────
LINEAGE_COLOURS = {
    "th17 cell":                     "#E63946",
    "treg cell":                     "#457B9D",
    "th1 cell":                      "#E76F51",
    "th2 cell":                      "#F4A261",
    "tfh cell":                      "#2A9D8F",
    "naive cd4 t cell":              "#A8DADC",
    "cd8 effector t cell":           "#C77DFF",
    "exhausted t cell":              "#7B2D8B",
    "gamma delta t cell":            "#FF6B6B",
    "mait cell":                     "#FFB347",
    "nkt cell":                      "#B5838D",
    "nk cell":                       "#6D6875",
    "ilc1":                          "#A2D9A5",
    "ilc2":                          "#74C69D",
    "ilc3":                          "#40916C",
    "m1 macrophage":                 "#D62828",
    "m2 macrophage":                 "#F77F00",
    "plasmacytoid dendritic cell":   "#FCBF49",
    "conventional dendritic cell 1": "#EAE2B7",
    "conventional dendritic cell 2": "#CDB4DB",
    "classical monocyte":            "#BDE0FE",
    "non-classical monocyte":        "#A2D2FF",
    "neutrophil":                    "#CCD5AE",
    "mast cell":                     "#E9C46A",
    "basophil":                      "#F4D35E",
    "eosinophil":                    "#EE9B00",
    "regulatory b cell":             "#94D2BD",
    "plasmablast":                   "#0A9396",
}

LINEAGE_GROUP = {
    "th17 cell": "CD4 T helper", "treg cell": "CD4 T helper",
    "th1 cell": "CD4 T helper", "th2 cell": "CD4 T helper",
    "tfh cell": "CD4 T helper", "naive cd4 t cell": "CD4 T helper",
    "cd8 effector t cell": "CD8 T", "exhausted t cell": "CD8 T",
    "gamma delta t cell": "Innate-like T", "mait cell": "Innate-like T",
    "nkt cell": "Innate-like T",
    "nk cell": "Innate", "ilc1": "Innate", "ilc2": "Innate", "ilc3": "Innate",
    "m1 macrophage": "Myeloid", "m2 macrophage": "Myeloid",
    "plasmacytoid dendritic cell": "Myeloid",
    "conventional dendritic cell 1": "Myeloid",
    "conventional dendritic cell 2": "Myeloid",
    "classical monocyte": "Myeloid", "non-classical monocyte": "Myeloid",
    "neutrophil": "Myeloid", "mast cell": "Myeloid",
    "basophil": "Myeloid", "eosinophil": "Myeloid",
    "regulatory b cell": "B cell", "plasmablast": "B cell",
}


# ══════════════════════════════════════════════════════════════════════
# Graph index builder
# ══════════════════════════════════════════════════════════════════════

def build_graph_index(train_tsv: Path) -> dict:
    """Parse train.tsv and index all immune-cell-relevant edges."""
    cell_to_marker_genes  = collections.defaultdict(set)   # IcE
    gene_to_cells         = collections.defaultdict(set)   # IcDv
    cell_to_diseases      = collections.defaultdict(set)   # IcIm
    drug_to_cells         = collections.defaultdict(set)   # DrIc
    disease_to_genes      = collections.defaultdict(set)   # ML/P/U/X/D/Te
    gene_to_diseases      = collections.defaultdict(set)
    all_immune_cells      = set()

    DISEASE_GENE_RELS = {"ML", "P", "U", "X", "D", "Te", "I"}

    with open(train_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r == "IcE":
                all_immune_cells.add(h)
                cell_to_marker_genes[h].add(t)
            elif r == "IcDv":
                all_immune_cells.add(t)
                gene_to_cells[h].add(t)
            elif r == "IcIm":
                all_immune_cells.add(h)
                cell_to_diseases[h].add(t)
            elif r == "DrIc":
                all_immune_cells.add(t)
                drug_to_cells[h].add(t)
            elif r in DISEASE_GENE_RELS:
                disease_to_genes[h].add(t)
                gene_to_diseases[t].add(h)

    return {
        "cell_to_marker_genes": dict(cell_to_marker_genes),
        "gene_to_cells":        dict(gene_to_cells),
        "cell_to_diseases":     dict(cell_to_diseases),
        "drug_to_cells":        dict(drug_to_cells),
        "disease_to_genes":     dict(disease_to_genes),
        "gene_to_diseases":     dict(gene_to_diseases),
        "all_immune_cells":     all_immune_cells,
    }


# ══════════════════════════════════════════════════════════════════════
# Contribution scoring
# ══════════════════════════════════════════════════════════════════════

def compute_cell_contributions(
    top_genes: list,
    gene_scores: dict,
    graph: dict,
    target_disease_genes: set,
) -> pd.DataFrame:
    """
    For each immune cell, compute two contribution metrics:

    icdv_score:
        Sum of prediction scores for top-ranked genes that also appear as
        IcDv heads pointing to this cell.
        Interpretation: high-scoring genes whose eQTL signal is concentrated
        in this cell type.

    ice_score:
        Number of top-ranked genes that are also IcE marker genes for this
        cell, weighted by rank score.
        Interpretation: the cell's defining markers overlap with the predicted
        target set.

    combined:
        Harmonic mean of icdv_score and ice_score (handles zeros gracefully).
    """
    rows = []
    gene_to_cells    = graph["gene_to_cells"]
    cell_to_markers  = graph["cell_to_marker_genes"]
    top_gene_set     = set(top_genes)

    for cell in graph["all_immune_cells"]:
        # ── Path A: IcDv ─────────────────────────────────────────────
        icdv_genes = {g for g, cells in gene_to_cells.items()
                      if cell in cells and g in top_gene_set}
        icdv_score = sum(gene_scores.get(g, 0.0) for g in icdv_genes)
        icdv_count = len(icdv_genes)

        # ── Path B: IcE ──────────────────────────────────────────────
        marker_genes = cell_to_markers.get(cell, set())
        ice_genes    = marker_genes & top_gene_set
        ice_score    = sum(gene_scores.get(g, 0.0) for g in ice_genes)
        ice_count    = len(ice_genes)

        # ── Combined ─────────────────────────────────────────────────
        if icdv_score > 0 and ice_score > 0:
            combined = 2 * icdv_score * ice_score / (icdv_score + ice_score)
        else:
            combined = max(icdv_score, ice_score)

        rows.append({
            "cell":        cell,
            "lineage":     LINEAGE_GROUP.get(cell, "Other"),
            "icdv_score":  icdv_score,
            "icdv_count":  icdv_count,
            "icdv_genes":  ", ".join(sorted(icdv_genes)[:8]),
            "ice_score":   ice_score,
            "ice_count":   ice_count,
            "ice_genes":   ", ".join(sorted(ice_genes)[:8]),
            "combined":    combined,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("combined", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ══════════════════════════════════════════════════════════════════════
# Embedding-space proximity analysis
# ══════════════════════════════════════════════════════════════════════

def compute_embedding_proximity(
    top_genes: list,
    kg_result,
    device: torch.device,
    graph: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    For each immune cell node, compute the mean cosine similarity to the
    top-ranked gene embeddings.  Cells that are geometrically close to
    the predicted target cluster are likely involved in the biology.
    """
    entity_to_id = kg_result.training.entity_to_id
    model = kg_result.model

    with torch.no_grad():
        all_emb = model.entity_representations[0](
            indices=torch.arange(model.num_entities, device=device)
        ).cpu()
    all_norm = F.normalize(all_emb.float(), p=2, dim=-1)

    # Centroid of top-gene embeddings
    gene_indices = [entity_to_id[g] for g in top_genes if g in entity_to_id]
    if not gene_indices:
        return pd.DataFrame()

    gene_embs  = all_norm[gene_indices]                # (N_genes, d)
    centroid   = gene_embs.mean(dim=0, keepdim=True)   # (1, d)
    centroid   = F.normalize(centroid, p=2, dim=-1)

    rows = []
    for cell in graph["all_immune_cells"]:
        cidx = entity_to_id.get(cell)
        if cidx is None:
            continue
        cell_emb  = all_norm[cidx:cidx+1]             # (1, d)
        sim_cent  = torch.mm(centroid, cell_emb.t()).item()

        # Per-gene similarities
        sims      = torch.mm(gene_embs, cell_emb.t()).squeeze(-1).numpy()
        mean_sim  = float(sims.mean())
        top_sim   = float(sims.max())

        rows.append({
            "cell":              cell,
            "lineage":           LINEAGE_GROUP.get(cell, "Other"),
            "centroid_sim":      sim_cent,
            "mean_gene_sim":     mean_sim,
            "max_gene_sim":      top_sim,
        })

    df = pd.DataFrame(rows).sort_values("centroid_sim", ascending=False)
    df.insert(0, "emb_rank", range(1, len(df) + 1))
    return df


# ══════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════

def _clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#444444")
    ax.set_facecolor("#FAFAFA")
    if grid:
        ax.xaxis.grid(True, linestyle="--", alpha=0.45, color="#DDDDDD")
        ax.set_axisbelow(True)


def _short_label(cell: str) -> str:
    MAP = {
        "th17 cell": "Th17", "treg cell": "Treg", "th1 cell": "Th1",
        "th2 cell": "Th2", "tfh cell": "Tfh", "naive cd4 t cell": "Naive CD4 T",
        "cd8 effector t cell": "CD8 Eff.", "exhausted t cell": "Exhausted T",
        "gamma delta t cell": "γδ T", "mait cell": "MAIT", "nkt cell": "NKT",
        "nk cell": "NK", "ilc1": "ILC1", "ilc2": "ILC2", "ilc3": "ILC3",
        "m1 macrophage": "M1 Macro", "m2 macrophage": "M2 Macro",
        "plasmacytoid dendritic cell": "pDC",
        "conventional dendritic cell 1": "cDC1",
        "conventional dendritic cell 2": "cDC2",
        "classical monocyte": "Classical Mono",
        "non-classical monocyte": "Non-cl. Mono",
        "neutrophil": "Neutrophil", "mast cell": "Mast",
        "basophil": "Basophil", "eosinophil": "Eosinophil",
        "regulatory b cell": "Breg", "plasmablast": "Plasmablast",
    }
    return MAP.get(cell, cell.title())


def plot_interpretability(
    contrib_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    disease_label: str,
    top_k: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.42,
                  left=0.08, right=0.97, top=0.95, bottom=0.12)

    show = contrib_df.head(14).copy()
    colours = [LINEAGE_COLOURS.get(c, "#888888") for c in show["cell"]]

    # ── Panel A: horizontal bar ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _clean_axes(ax1)
    bars = ax1.barh(range(len(show)), show["combined"], color=colours,
                    height=0.62, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(show)))
    ax1.set_yticklabels([_short_label(c) for c in show["cell"]], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Combined Structural Contribution Score", fontsize=10.5)
    ax1.tick_params(axis="x", labelsize=9.5)

    xmax = show["combined"].max()
    ax1.set_xlim(0, xmax * 1.35)
    for i, (bar, row) in enumerate(zip(bars, show.itertuples())):
        w = bar.get_width()
        offset = xmax * 0.025
        if row.icdv_count > 0:
            ax1.text(w + offset, i - 0.20, f"IcDv: {row.icdv_count}",
                     va="center", ha="left", fontsize=8,
                     color="#C0392B", fontweight="semibold")
        if row.ice_count > 0:
            ax1.text(w + offset, i + 0.24, f"IcE: {row.ice_count}",
                     va="center", ha="left", fontsize=8,
                     color="#2980B9", fontweight="semibold")

    seen_lineages = {}
    for c in show["cell"]:
        lin = LINEAGE_GROUP.get(c, "Other")
        if lin not in seen_lineages:
            seen_lineages[lin] = LINEAGE_COLOURS.get(c, "#888")
    patches = [mpatches.Patch(fc=v, ec="none", alpha=0.88, label=k)
               for k, v in seen_lineages.items()]
    ax1.legend(handles=patches, loc="lower right", fontsize=8.5,
               frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
               title="Lineage", title_fontsize=8.5)

    # ── Panel B: IcDv vs IcE scatter ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _clean_axes(ax2, grid=True)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.45, color="#DDDDDD")

    top14 = contrib_df.head(14)
    sc_c  = [LINEAGE_COLOURS.get(c, "#888") for c in top14["cell"]]
    ax2.scatter(top14["icdv_score"], top14["ice_score"],
                c=sc_c, s=110, alpha=0.92,
                edgecolors="#333333", linewidths=0.6, zorder=3)

    try:
        from adjustText import adjust_text
        texts = [ax2.text(row["icdv_score"], row["ice_score"],
                          _short_label(row["cell"]), fontsize=8, color="#1A1A2E")
                 for _, row in top14.iterrows()]
        adjust_text(texts, ax=ax2,
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6),
                    expand=(1.3, 1.5), force_text=(0.4, 0.6))
    except ImportError:
        for _, row in top14.iterrows():
            ax2.annotate(_short_label(row["cell"]),
                         (row["icdv_score"], row["ice_score"]),
                         fontsize=8, color="#1A1A2E",
                         xytext=(4, 4), textcoords="offset points")

    ax2.set_xlabel("IcDv Score  (eQTL-gene bridge)", fontsize=9.5)
    ax2.set_ylabel("IcE Score  (marker-gene bridge)", fontsize=9.5)
    ax2.tick_params(labelsize=9)

    # ── Save ─────────────────────────────────────────────────────────
    safe = disease_label.replace(" ", "_").replace("/", "_")
    stem = f"Immune_Cell_Structural_Contribution_{safe}_Top{top_k}_Targets"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_png = output_dir / f"{stem}.png"
    print(f"  Saved: {out_png}")
    return out_png


def plot_gene_cell_heatmap(
    contrib_df: pd.DataFrame,
    gene_scores: dict,
    graph: dict,
    disease_label: str,
    output_dir: Path,
    top_genes: int = 20,
    top_cells: int = 12,
):
    """
    Heatmap: top immune cells (rows) x top predicted genes (cols).
    Cell (i, j) = weighted score if cell i is linked to gene j via IcE or IcDv.
    White background, no title (filename carries the title).
    """
    top_cell_names = list(contrib_df.head(top_cells)["cell"])
    ranked_genes   = sorted(gene_scores, key=gene_scores.get, reverse=True)[:top_genes]

    cell_to_markers = graph["cell_to_marker_genes"]
    gene_to_cells   = graph["gene_to_cells"]

    matrix = np.zeros((len(top_cell_names), len(ranked_genes)))
    link_type = np.zeros_like(matrix)   # 1=IcE, 2=IcDv, 3=both
    for i, cell in enumerate(top_cell_names):
        markers = cell_to_markers.get(cell, set())
        driven  = {g for g, cells in gene_to_cells.items() if cell in cells}
        for j, gene in enumerate(ranked_genes):
            has_ice  = gene in markers
            has_icdv = gene in driven
            if has_ice and has_icdv:
                matrix[i, j] = gene_scores[gene]
                link_type[i, j] = 3
            elif has_ice:
                matrix[i, j] = gene_scores[gene] * 0.85
                link_type[i, j] = 1
            elif has_icdv:
                matrix[i, j] = gene_scores[gene] * 0.55
                link_type[i, j] = 2

    cell_height = max(0.42, 5.5 / max(top_cells, 1))
    gene_width  = max(0.40, 9.0 / max(top_genes, 1))
    fig_h = top_cells  * cell_height + 1.5
    fig_w = top_genes  * gene_width  + 3.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Custom colormap: white → light blue → deep blue
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ice_heat", ["#FFFFFF", "#AED6F1", "#1A5276"], N=256)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=0, vmax=1.0)

    # Mark "both" cells with a dot
    for i in range(len(top_cell_names)):
        for j in range(len(ranked_genes)):
            if link_type[i, j] == 3:
                ax.plot(j, i, "o", color="#E74C3C", ms=4, zorder=4)
            elif link_type[i, j] == 2:
                ax.plot(j, i, "s", color="#C0392B", ms=3.5,
                        alpha=0.6, zorder=4)

    ax.set_xticks(range(len(ranked_genes)))
    ax.set_xticklabels([g.upper() for g in ranked_genes],
                       rotation=50, ha="right", fontsize=8.5, color="#1A1A2E")
    ax.set_yticks(range(len(top_cell_names)))
    ax.set_yticklabels([_short_label(c) for c in top_cell_names],
                       fontsize=9.5, color="#1A1A2E")

    ax.spines[:].set_color("#CCCCCC")
    ax.tick_params(colors="#444444")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(colors="#444444", labelsize=8)
    cbar.set_label("Weighted prediction score", color="#444444", fontsize=9)
    cbar.outline.set_edgecolor("#CCCCCC")

    # Compact legend for marker symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#E74C3C",
               ms=7, label="IcE + IcDv (both)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#C0392B",
               ms=6, alpha=0.7, label="IcDv only"),
        mpatches.Patch(fc="#AED6F1", ec="none", label="IcE only"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8.5, frameon=True, framealpha=0.92,
              edgecolor="#CCCCCC", bbox_to_anchor=(1.0, -0.18),
              ncol=3)

    fig.tight_layout(pad=1.2)

    safe = disease_label.replace(" ", "_").replace("/", "_")
    stem = f"Gene_Cell_Structural_Link_Heatmap_{safe}"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_png = output_dir / f"{stem}.png"
    print(f"  Saved: {out_png}")
    return out_png


# ══════════════════════════════════════════════════════════════════════
# Figure: Embedding-Space Proximity bar
# ══════════════════════════════════════════════════════════════════════

def plot_embedding_proximity(
    emb_df: pd.DataFrame,
    disease_label: str,
    output_dir: Path,
):
    """
    Horizontal bar chart: cosine similarity of each immune cell embedding
    to the centroid of top-ranked gene embeddings.
    White background, no figure title (filename carries the title).
    """
    if emb_df is None or emb_df.empty:
        print("  Embedding proximity skipped (no data).")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    show = emb_df.copy().sort_values("centroid_sim", ascending=True)

    colours = [LINEAGE_COLOURS.get(c, "#888888") for c in show["cell"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(show) * 0.40)))
    fig.patch.set_facecolor("white")
    _clean_axes(ax, grid=True)

    ax.barh(range(len(show)), show["centroid_sim"],
            color=colours, height=0.68, alpha=0.9,
            edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([_short_label(c) for c in show["cell"]], fontsize=10)
    ax.set_xlabel("Cosine Similarity to Top-Gene Centroid", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.axvline(0, color="#888888", lw=0.9, ls="--", zorder=0)

    # Lineage legend
    seen = {}
    for c in show["cell"]:
        lin = LINEAGE_GROUP.get(c, "Other")
        if lin not in seen:
            seen[lin] = LINEAGE_COLOURS.get(c, "#888")
    patches = [mpatches.Patch(fc=v, ec="none", alpha=0.88, label=k)
               for k, v in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8.5,
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
              title="Lineage", title_fontsize=8.5,
              ncol=max(1, len(seen) // 3))

    fig.tight_layout(pad=1.2)

    safe = disease_label.replace(" ", "_").replace("/", "_")
    stem = f"Embedding_Space_Proximity_to_Predicted_Target_Cluster_{safe}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_png = output_dir / f"{stem}.png"
    print(f"  Saved: {out_png}")
    return out_png


# ══════════════════════════════════════════════════════════════════════
# Figure: Lineage-Level Summary bar
# ══════════════════════════════════════════════════════════════════════

def plot_lineage_summary(
    contrib_df: pd.DataFrame,
    disease_label: str,
    output_dir: Path,
):
    """
    Horizontal bar chart: total combined contribution score summed per lineage.
    White background, no figure title (filename carries the title).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_contrib = (
        contrib_df.groupby("lineage")["combined"]
        .sum()
        .sort_values(ascending=True)
    )

    # Pick one representative colour per lineage
    lineage_rep_colour = {}
    for cell, lin in LINEAGE_GROUP.items():
        if lin not in lineage_rep_colour:
            lineage_rep_colour[lin] = LINEAGE_COLOURS.get(cell, "#888888")

    labels  = list(lineage_contrib.index)
    values  = list(lineage_contrib.values)
    colours = [lineage_rep_colour.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.55)))
    fig.patch.set_facecolor("white")
    _clean_axes(ax, grid=True)

    bars = ax.barh(range(len(labels)), values,
                   color=colours, height=0.65, alpha=0.9,
                   edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Total Contribution Score", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9.5)

    # Value labels on bars
    xmax = max(values) if values else 1
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + xmax * 0.02, i,
                f"{val:.1f}", va="center", ha="left",
                fontsize=8.5, color="#444444")
    ax.set_xlim(0, xmax * 1.18)

    fig.tight_layout(pad=1.2)

    safe = disease_label.replace(" ", "_").replace("/", "_")
    stem = f"Lineage_Level_Contribution_Summary_{safe}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_png = output_dir / f"{stem}.png"
    print(f"  Saved: {out_png}")
    return out_png


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="immuneKG immune cell interpretability analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--disease", nargs="+", required=True,
        help='Disease keyword(s), e.g. "inflammatory bowel diseases" "colitis"',
    )
    p.add_argument(
        "--results", type=str, default=None,
        help="Path to pre-computed full target scoring CSV. "
             "If omitted the script loads the model and scores on the fly.",
    )
    p.add_argument("--top-k",    type=int, default=50,
                   help="Number of top-ranked genes to analyse (default: 50)")
    p.add_argument("--output",   type=str, default="results/immunecell_interpretability",
                   help="Output directory (default: results/immunecell_interpretability)")
    p.add_argument("--config",   type=str, default="configs/default.yaml")
    p.add_argument("--no-embedding", action="store_true",
                   help="Skip embedding proximity analysis (faster, no model needed)")
    return p.parse_args()


def main():
    args = parse_args()
    config   = load_config(args.config)
    work_dir = Path(config["output"]["work_dir"])
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    disease_label = " + ".join(args.disease)
    print(f"\n  Disease(s): {disease_label}")
    print(f"  Top-K:      {args.top_k}")
    print(f"  Output:     {out_dir.resolve()}\n")

    # ── Build graph index ────────────────────────────────────────────
    train_tsv = Path("data/train.tsv")
    if not train_tsv.exists():
        print(f"ERROR: {train_tsv} not found"); sys.exit(1)

    print("Building graph index...")
    graph = build_graph_index(train_tsv)
    print(f"  Immune cells in graph: {len(graph['all_immune_cells'])}")

    # ── Load or use pre-computed target scores ───────────────────────
    if args.results and Path(args.results).exists():
        print(f"Loading pre-computed scores: {args.results}")
        scores_df = pd.read_csv(args.results)
        # Normalise column names — accept target_name or target_id
        name_col = "target_name" if "target_name" in scores_df.columns else "target_id"
        score_col = "combined_score_norm" if "combined_score_norm" in scores_df.columns \
                    else "combined_score"
        scores_df = scores_df[[name_col, score_col]].rename(
            columns={name_col: "gene", score_col: "score"}
        )
    else:
        # Minimal inline scoring (KG only, no GNN/fusion)
        print("No pre-computed results found — running KG-only scoring...")
        device = setup_device(config)
        filenames = config["output"]["filenames"]
        kg_path   = work_dir / filenames["kg_model"]
        data_path = work_dir / filenames["processed_data"]
        if not kg_path.exists() or not data_path.exists():
            print("ERROR: trained model not found. Run train.py first.")
            sys.exit(1)
        with open(kg_path,   "rb") as f: kg_result = pickle.load(f)
        with open(data_path, "rb") as f: data      = pickle.load(f)
        kg_result.model = kg_result.model.to(device)
        kg_result.model.eval()

        entity_to_id   = kg_result.training.entity_to_id
        num_relations  = kg_result.training.num_relations
        df_kg          = data["dataframe"]
        entity_info    = data["entity_info"]

        # Resolve disease entities
        disease_ids = set()
        for kw in args.disease:
            mask = df_kg["x_id"].str.contains(kw, case=False, na=False)
            if "x_type" in df_kg.columns:
                mask &= df_kg["x_type"] == "disease"
            disease_ids.update(df_kg[mask]["x_id"].unique())
            mask2 = df_kg["y_id"].str.contains(kw, case=False, na=False)
            if "y_type" in df_kg.columns:
                mask2 &= df_kg["y_type"] == "disease"
            disease_ids.update(df_kg[mask2]["y_id"].unique())

        if not disease_ids:
            print(f"ERROR: no disease entities found for {args.disease}")
            sys.exit(1)
        print(f"  Resolved disease entities: {sorted(disease_ids)}")

        # Candidate genes
        if "x_type" in df_kg.columns:
            gene_pool = set(df_kg[df_kg["x_type"] == "gene/protein"]["x_id"].unique())
            gene_pool |= set(df_kg[df_kg["y_type"] == "gene/protein"]["y_id"].unique())
        else:
            gene_pool = set(df_kg["x_id"].unique()) | set(df_kg["y_id"].unique())

        valid_diseases = [d for d in disease_ids if d in entity_to_id]
        valid_genes    = [g for g in gene_pool    if g in entity_to_id]

        from tqdm import tqdm
        import torch
        rows_sc = []
        for gene in tqdm(valid_genes, desc="Scoring", ncols=70):
            g_idx  = entity_to_id[gene]
            best   = -1e9
            for d in valid_diseases:
                d_idx = entity_to_id[d]
                for r in range(num_relations):
                    with torch.no_grad():
                        s = kg_result.model.score_hrt(
                            torch.tensor([[g_idx, r, d_idx]],
                                         device=device, dtype=torch.long)
                        ).cpu().item()
                    if s > best:
                        best = s
            rows_sc.append({"gene": gene, "score": best})

        scores_df = pd.DataFrame(rows_sc).sort_values("score", ascending=False)

    # ── Extract top-K genes ──────────────────────────────────────────
    top_genes  = list(scores_df["gene"].head(args.top_k))
    gene_scores = dict(zip(scores_df["gene"], scores_df["score"]))

    # Normalise scores to [0, 1] for display
    vals = np.array([gene_scores[g] for g in top_genes], dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        for g in top_genes:
            gene_scores[g] = (gene_scores[g] - vmin) / (vmax - vmin)
    print(f"  Top-{args.top_k} genes extracted.")

    # ── Build disease gene set ───────────────────────────────────────
    target_disease_genes = set()
    for kw in args.disease:
        for d, genes in graph["disease_to_genes"].items():
            if kw.lower() in d.lower():
                target_disease_genes |= genes

    # ── Contribution scoring ─────────────────────────────────────────
    print("Computing immune cell contributions...")
    contrib_df = compute_cell_contributions(
        top_genes, gene_scores, graph, target_disease_genes
    )

    contrib_csv = out_dir / f"contributions_{disease_label.replace(' ', '_')}.csv"
    contrib_df.to_csv(contrib_csv, index=False)
    print(f"  Saved: {contrib_csv}")

    print("\n  Top-10 immune cells by structural contribution:")
    for _, row in contrib_df.head(10).iterrows():
        print(f"    {row['rank']:2d}. {row['cell']:<35} "
              f"combined={row['combined']:.4f}  "
              f"IcDv={row['icdv_count']}  IcE={row['ice_count']}")

    # ── Embedding proximity ──────────────────────────────────────────
    emb_df = pd.DataFrame()
    if not args.no_embedding:
        try:
            if "kg_result" not in dir():
                device = setup_device(config)
                kg_path = work_dir / config["output"]["filenames"]["kg_model"]
                with open(kg_path, "rb") as f:
                    kg_result = pickle.load(f)
                kg_result.model = kg_result.model.to(device)
                kg_result.model.eval()
            print("Computing embedding proximity...")
            emb_df = compute_embedding_proximity(
                top_genes, kg_result, device, graph
            )
            emb_csv = out_dir / f"embedding_proximity_{disease_label.replace(' ', '_')}.csv"
            emb_df.to_csv(emb_csv, index=False)
            print(f"  Saved: {emb_csv}")
        except Exception as e:
            print(f"  Embedding proximity skipped: {e}")

    # ── Plots ────────────────────────────────────────────────────────
    print("Generating visualisations...")
    plot_interpretability(contrib_df, emb_df, disease_label, args.top_k, out_dir)
    plot_gene_cell_heatmap(contrib_df, gene_scores, graph, disease_label, out_dir)
    plot_embedding_proximity(emb_df, disease_label, out_dir)
    plot_lineage_summary(contrib_df, disease_label, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
