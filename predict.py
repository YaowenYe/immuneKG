#!/usr/bin/env python3
"""
immuneKG Prediction Script

Three independent modes:

  1. target_scoring  — given disease keyword(s), rank all gene/protein targets
                       using the full trained pipeline (ComplEx + fusion + GNN).

  2. link_prediction — given a head entity + relation, predict the most likely
                       tail entities (tail prediction); or given a tail entity +
                       relation, predict the most likely head entities (head
                       prediction).

  3. similarity      — find the K most similar entities to a query entity based
                       on cosine similarity in embedding space.

Relation reference (32 types in the KG):
  Gene–Gene:
    GG   Gene–gene interaction (generic)
    E    Co-expression
    Rg   Gene regulates gene
    Ra   Gene activates gene
    Q    Quantitative association

  Disease–Gene (head=disease, tail=gene):
    ML   Marker / mechanism of disease
    X    Disease overexpresses gene
    U    Mutation alters disease risk
    D    Gene downregulated in disease

  Gene–Disease (head=gene, tail=disease):
    P    Gene role in pathogenesis / promotes progression
    Te   Therapeutic target

  Drug–Disease (head=drug, tail=disease):
    T    Treatment / therapy
    Sa   Side effect / adverse event
    C    Contraindication
    J    Drug role in pathogenesis
    Pr   Drug prevents / alleviates disease

  Disease–Disease:
    An   Disease is ancestor of another disease
    As   Disease associated with disease

  Drug/Chemical–Gene (head=drug, tail=gene):
    N    Drug inhibits gene
    A    Drug activates gene
    B    Chemical binds gene / protein
    I    Interaction (mixed)

  Gene–Drug (head=gene, tail=drug):
    K    Gene metabolises chemical
    Z    Gene transports chemical
    O    Gene transports drug

  Drug–Drug:
    CC   Chemical–chemical interaction

  Mixed / pathway:
    Iw   Entity involved in pathway or disease process
    Mp   Disease biomarker of progression (head=disease, tail=drug)

  Novel immuneKG relations (★):
    IcE  immunecell_expresses_marker_gene   (head=immunecell, tail=gene/protein)
    IcIm immunecell_implicated_in_disease   (head=immunecell, tail=disease)
    IcDv gene_drives_immunecell_differentiation  (head=gene/protein, tail=immunecell)
    DrIc drug_modulates_immunecell          (head=drug, tail=immunecell)

Usage examples:
  # Mode 1 — disease target scoring
  python predict.py --mode target_scoring \\
      --keywords "colitis" "crohn disease" \\
      --top-k 50

  # Mode 2 — tail prediction (head + relation -> tail)
  python predict.py --mode link_prediction \\
      --head "tnf" --relation "P" --top-k 20

  # Mode 2 — head prediction (tail + relation -> head)
  python predict.py --mode link_prediction \\
      --tail "colitis" --relation "T" --top-k 20

  # Mode 2 — novel immuneKG queries
  python predict.py --mode link_prediction \\
      --head "th17 cell" --relation "IcE" --top-k 20
      # -> which marker genes does Th17 express?

  python predict.py --mode link_prediction \\
      --head "th17 cell" --relation "IcIm" --top-k 20
      # -> which diseases is Th17 implicated in?

  python predict.py --mode link_prediction \\
      --tail "th17 cell" --relation "IcDv" --top-k 20
      # -> which genes drive Th17 differentiation?

  python predict.py --mode link_prediction \\
      --tail "th17 cell" --relation "DrIc" --top-k 20
      # -> which drugs modulate Th17?

  python predict.py --mode link_prediction \\
      --tail "crohn disease" --relation "IcIm" --top-k 20
      # -> which immune cells are implicated in Crohn's disease?

  python predict.py --mode link_prediction \\
      --tail "m1 macrophage" --relation "DrIc" --top-k 20
      # -> which drugs modulate M1 macrophages?

  # Mode 3 — embedding similarity
  python predict.py --mode similarity \\
      --entity "il6" --top-k 20

  # List all available relations and entity types
  python predict.py --list-relations
  python predict.py --list-entities --entity-type disease
  python predict.py --list-entities --entity-type immune_cell
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_device, Timer,
    print_banner, print_stage, print_success, print_warning,
    print_info, print_stat, print_error,
    checkpoint_exists, load_checkpoint
)
from src.model import load_fusion_model


# ── Relation catalogue ────────────────────────────────────────────────
# Built from inspection of the actual training data (train.tsv).
# Novel relations unique to immuneKG are marked with ★.

RELATION_DESCRIPTIONS = {
    # Gene–Gene
    "GG":   "gene_gene_interaction — generic gene–gene interaction",
    "E":    "gene_coexpression — genes are co-expressed",
    "Rg":   "gene_regulates_gene — one gene regulates another",
    "Ra":   "gene_activates_gene — one gene activates another",
    "Q":    "gene_quantitative_association — quantitative association between genes",

    # Disease–Gene  (head=disease, tail=gene)
    "ML":   "disease_gene_marker — gene is a biomarker or mechanistic driver of disease",
    "X":    "disease_overexpresses_gene — gene is overexpressed in disease",
    "U":    "gene_mutation_alters_risk — mutation/polymorphism in gene alters disease risk",
    "D":    "gene_downregulated_in_disease — gene is downregulated in disease context",

    # Gene–Disease  (head=gene, tail=disease)
    "P":    "gene_pathogenesis_role — gene plays a role in disease pathogenesis",
    "Te":   "gene_therapeutic_target — gene is a therapeutic target for disease",

    # Drug–Disease  (head=drug, tail=disease)
    "T":    "drug_treats_disease — drug is used as treatment for disease",
    "Sa":   "drug_side_effect — drug causes adverse event or side effect in disease context",
    "C":    "drug_contraindication — drug is contraindicated in disease",
    "J":    "drug_pathogenesis_role — drug plays a role in disease pathogenesis",
    "Pr":   "drug_prevents_disease — drug prevents, suppresses, or alleviates disease",

    # Disease–Disease
    "An":   "disease_ancestor_of — one disease is an ancestor of another in ontology",
    "As":   "disease_associated_with — diseases are clinically associated",

    # Drug–Gene  (head=drug, tail=gene)
    "N":    "drug_inhibits_gene — drug inhibits gene product (protein/enzyme)",
    "A":    "drug_activates_gene — drug activates gene product",
    "B":    "chemical_binds_gene — chemical/drug binds to gene product",
    "I":    "entity_interacts_with — interaction between entities (mixed types)",

    # Gene–Drug  (head=gene, tail=drug)
    "K":    "gene_metabolises_chemical — gene product metabolises the chemical",
    "Z":    "gene_transports_chemical — gene product transports the chemical",
    "O":    "gene_transports_drug — gene product is a transporter channel for drug",

    # Drug–Drug
    "CC":   "chemical_chemical_interaction — interaction between two chemicals/drugs",

    # Mixed / pathway
    "Iw":   "entity_involved_in — entity is involved in pathway or disease process",
    "Mp":   "disease_progression_biomarker — biomarker of disease progression",

    # Novel immuneKG relations ★
    "IcE":  "★ immunecell_expresses_marker_gene — immune cell subtype specifically expresses a "
            "marker gene (transcription factor, cytokine, surface receptor) that defines or "
            "characterises its identity or function  [head=immunecell, tail=gene/protein]  "
            "[Source: CellMarker 2.0]",

    "IcIm": "★ immunecell_implicated_in_disease — immune cell subtype plays a documented "
            "pathogenic or regulatory role in an autoimmune disease, including tissue damage, "
            "aberrant expansion, or autoantibody production  [head=immunecell, tail=disease]  "
            "[Source: manual curation]",

    "IcDv": "★ gene_drives_immunecell_differentiation — a gene (transcription factor, cytokine, "
            "or signalling molecule) is required for the differentiation, maintenance, or "
            "functional polarisation of the immune cell subtype  [head=gene/protein, tail=immunecell]  "
            "[Source: DICE eQTL]",

    "DrIc": "★ drug_modulates_immunecell — a drug significantly alters the abundance, activation "
            "state, cytokine secretion, or survival of the immune cell subtype via its known "
            "mechanism of action  [head=drug, tail=immunecell]  [Source: manual curation]",
}

# For each relation: which type is expected at head and tail position.
# Used for automatic candidate filtering in link prediction.
RELATION_TYPE_CONSTRAINTS = {
    # Gene–Gene
    "GG":   ("gene/protein", "gene/protein"),
    "E":    ("gene/protein", "gene/protein"),
    "Rg":   ("gene/protein", "gene/protein"),
    "Ra":   ("gene/protein", "gene/protein"),
    "Q":    ("gene/protein", "gene/protein"),
    # Disease–Gene
    "ML":   ("disease",      "gene/protein"),
    "X":    ("disease",      "gene/protein"),
    "U":    ("gene/protein", "disease"),
    "D":    ("gene/protein", "disease"),
    # Gene–Disease
    "P":    ("gene/protein", "disease"),
    "Te":   ("gene/protein", "disease"),
    # Drug–Disease
    "T":    ("drug",         "disease"),
    "Sa":   ("drug",         "disease"),
    "C":    ("drug",         "disease"),
    "J":    ("drug",         "disease"),
    "Pr":   ("drug",         "disease"),
    # Disease–Disease
    "An":   ("disease",      "disease"),
    "As":   ("disease",      "disease"),
    # Drug–Gene
    "N":    ("drug",         "gene/protein"),
    "A":    ("drug",         "gene/protein"),
    "B":    (None,           "gene/protein"),
    "I":    (None,           None),
    # Gene–Drug
    "K":    ("gene/protein", "drug"),
    "Z":    ("gene/protein", "drug"),
    "O":    ("gene/protein", "drug"),
    # Drug–Drug
    "CC":   ("drug",         "drug"),
    # Mixed
    "Iw":   (None,           None),
    "Mp":   ("disease",      "drug"),
    # Novel immuneKG relations
    "IcE":  ("immune_cell",  "gene/protein"),
    "IcIm": ("immune_cell",  "disease"),
    "IcDv": ("gene/protein", "immune_cell"),
    "DrIc": ("drug",         "immune_cell"),
}


# ══════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="immuneKG - flexible KG prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--mode", type=str,
        choices=["target_scoring", "link_prediction", "similarity"],
        default="target_scoring",
        help="Prediction mode (default: target_scoring)",
    )

    grp1 = parser.add_argument_group("target_scoring options")
    grp1.add_argument(
        "--keywords", nargs="+", type=str,
        help='Disease keyword(s) to search, e.g. "colitis" "crohn disease"',
    )
    grp1.add_argument(
        "--relations", nargs="+", type=str, default=None,
        help="Restrict scoring to these relation types (default: all)",
    )

    grp2 = parser.add_argument_group("link_prediction options")
    grp2.add_argument("--head", type=str, help="Head entity for tail prediction")
    grp2.add_argument("--tail", type=str, help="Tail entity for head prediction")
    grp2.add_argument("--relation", type=str, help='Relation type, e.g. "P", "IcE"')
    grp2.add_argument(
        "--target-type", type=str, default=None,
        help="Override automatic type filter: disease / gene/protein / drug / immune_cell",
    )

    grp3 = parser.add_argument_group("similarity options")
    grp3.add_argument("--entity", type=str, help="Query entity name")
    grp3.add_argument("--filter-type", type=str, default=None,
                      help="Only return entities of this type")

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kg-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--list-relations", action="store_true")
    parser.add_argument("--list-entities", action="store_true")
    parser.add_argument("--entity-type", type=str, default=None)

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

def load_models(config: dict, device: torch.device, kg_only: bool = False):
    work_dir = Path(config["output"]["work_dir"])
    filenames = config["output"]["filenames"]

    kg_path = work_dir / filenames["kg_model"]
    if not kg_path.exists():
        print_error(f"KG model not found: {kg_path}")
        print_info("Run train.py first.")
        sys.exit(1)

    print_info(f"Loading KG model: {kg_path.name}")
    with open(kg_path, "rb") as f:
        kg_result = pickle.load(f)
    kg_result.model = kg_result.model.to(device)
    kg_result.model.eval()
    print_stat("Entities", f"{kg_result.model.num_entities:,}")
    print_stat("Relations", kg_result.model.num_relations)

    data_path = work_dir / filenames["processed_data"]
    if not data_path.exists():
        print_error(f"Processed data not found: {data_path}")
        sys.exit(1)
    print_info(f"Loading KG data: {data_path.name}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    fusion_model = feature_result = None
    if not kg_only:
        fusion_path = work_dir / filenames["fusion_model"]
        feat_path   = work_dir / filenames.get("disease_features", "disease_features.pkl")
        if fusion_path.exists() and feat_path.exists():
            print_info(f"Loading fusion model: {fusion_path.name}")
            fusion_model = load_fusion_model(str(fusion_path), device)
            fusion_model.eval()
            with open(feat_path, "rb") as f:
                feature_result = pickle.load(f)
            print_stat("Feature dim", feature_result["total_feature_dim"])
        else:
            print_warning("Fusion model not found - using KG embeddings only")

    gnn_embeddings = degree_map = graph_result = None
    gnn_path   = work_dir / "gnn_embeddings.pkl"
    graph_path = work_dir / "pyg_graphs.pkl"
    if not kg_only and gnn_path.exists():
        print_info(f"Loading GNN embeddings: {gnn_path.name}")
        gnn_embeddings = load_checkpoint(str(gnn_path))
        print_stat("GNN shape", gnn_embeddings.shape)
        if graph_path.exists():
            graph_result = load_checkpoint(str(graph_path))
            try:
                from src.graph_builder import HeteroGraphBuilder
                gb = HeteroGraphBuilder(config, str(work_dir))
                degree_map = gb.compute_target_degrees(data, graph_result)
            except Exception:
                pass

    return kg_result, fusion_model, data, feature_result, gnn_embeddings, degree_map, graph_result


# ══════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════

def resolve_entity(name: str, data: dict):
    name_lower = name.strip().lower()
    entities = data["entities"]
    if name_lower in entities:
        return name_lower
    matches = [e for e in entities if name_lower in e]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print_warning(f"'{name}' is ambiguous - {len(matches)} matches found:")
        for m in matches[:10]:
            print(f"    {m}")
        if len(matches) > 10:
            print(f"    ... and {len(matches)-10} more")
        print_info("Use a more specific name.")
        return None
    print_error(f"Entity '{name}' not found in the KG.")
    print_info("Run --list-entities to browse available entities.")
    return None


def resolve_relation(rel: str, kg_result):
    """
    Return the forward relation index only.
    Does NOT fall back to _inverse — using the inverse relation
    silently reverses prediction direction and produces meaningless results.
    """
    relation_to_id = kg_result.training.relation_to_id
    if rel in relation_to_id:
        return relation_to_id[rel]
    print_error(f"Relation '{rel}' not found in model.")
    print_info("Run --list-relations to see all available relation types.")
    print_info(
        "Note: PyKEEN stores inverse relations as '<rel>_inverse'. "
        "Always use the forward relation name."
    )
    return None


def extract_all_embeddings(kg_result, device: torch.device) -> np.ndarray:
    model = kg_result.model
    with torch.no_grad():
        emb = model.entity_representations[0](
            indices=torch.arange(model.num_entities, device=device)
        ).cpu().numpy()
    return emb


def get_type_filtered_candidates(rel: str, is_tail_pred: bool,
                                 data: dict, entity_to_id: dict,
                                 override_type: str = None) -> list:
    """
    Return candidate entity IDs filtered to the expected type for this
    relation and prediction direction.

    For tail prediction (head + rel -> ?):  use tail type constraint.
    For head prediction (? + rel -> tail):  use head type constraint.
    """
    constraints = RELATION_TYPE_CONSTRAINTS.get(rel, (None, None))
    head_type, tail_type = constraints

    if override_type:
        target_type = override_type.lower()
    elif is_tail_pred:
        target_type = tail_type
    else:
        target_type = head_type

    df = data["dataframe"]

    if target_type is None or "x_type" not in df.columns:
        # No constraint — return all entities
        return list(entity_to_id.keys())

    pool_x = set(df[df["x_type"] == target_type]["x_id"].unique())
    pool_y = set(df[df["y_type"] == target_type]["y_id"].unique())
    candidates = sorted(pool_x | pool_y)

    if not candidates:
        print_warning(
            f"Type filter '{target_type}' returned 0 candidates. "
            "Falling back to all entities."
        )
        return list(entity_to_id.keys())

    return candidates


# ══════════════════════════════════════════════════════════════════════
# Mode 1 — Target scoring
# ══════════════════════════════════════════════════════════════════════

def find_disease_entities(keywords: list, data: dict) -> dict:
    df = data["dataframe"]
    result = {}

    for kw in keywords:
        found = set()
        kw_lower = kw.lower()

        mask_x = df["x_id"].str.contains(kw_lower, case=False, na=False)
        if "x_type" in df.columns:
            mask_x &= df["x_type"] == "disease"
        found.update(df[mask_x]["x_id"].unique())

        mask_y = df["y_id"].str.contains(kw_lower, case=False, na=False)
        if "y_type" in df.columns:
            mask_y &= df["y_type"] == "disease"
        found.update(df[mask_y]["y_id"].unique())

        result[kw] = sorted(found)
        if found:
            for eid in list(found)[:5]:
                print_stat(f"  {kw}", f"-> {eid}")
            if len(found) > 5:
                print_info(f"    ... and {len(found)-5} more")
        else:
            print_warning(f"No disease entities found for '{kw}'")

    return result


def mode_target_scoring(args, kg_result, fusion_model, data,
                         feature_result, gnn_embeddings, degree_map,
                         graph_result, device):
    keywords = args.keywords or ["colitis", "inflammatory bowel diseases"]
    print_info(f"Disease keywords: {keywords}")
    target_diseases = find_disease_entities(keywords, data)

    all_disease_ids = sorted({eid for ids in target_diseases.values() for eid in ids})
    if not all_disease_ids:
        print_error("No disease entities resolved. Exiting.")
        return pd.DataFrame()

    config = load_config(args.config)
    target_types = config["prediction"].get("target_entity_types", ["gene/protein"])
    df = data["dataframe"]
    target_entities = set()
    for tt in target_types:
        if "x_type" in df.columns:
            target_entities |= set(df[df["x_type"] == tt]["x_id"].unique())
            target_entities |= set(df[df["y_type"] == tt]["y_id"].unique())
        else:
            target_entities = (
                set(df["x_id"].unique()) | set(df["y_id"].unique())
            ) - set(data["disease_entities"])
    print_stat("Candidate targets", f"{len(target_entities):,}")

    model = kg_result.model
    entity_to_id = kg_result.training.entity_to_id
    relation_to_id = kg_result.training.relation_to_id
    num_relations = kg_result.training.num_relations
    entity_info = data["entity_info"]

    valid_diseases = [d for d in all_disease_ids if d in entity_to_id]
    valid_targets  = [t for t in target_entities if t in entity_to_id]
    print_stat("Diseases in model", len(valid_diseases))
    print_stat("Targets in model",  f"{len(valid_targets):,}")

    if args.relations:
        rel_ids = set()
        for r in args.relations:
            rid = relation_to_id.get(r)
            if rid is not None:
                rel_ids.add(rid)
            else:
                print_warning(f"Relation '{r}' not in model, skipping")
        if not rel_ids:
            print_error("None of the specified relations are in the model.")
            return pd.DataFrame()
        relation_range = sorted(rel_ids)
        print_stat("Using relations", args.relations)
    else:
        relation_range = list(range(num_relations))

    kg_embeddings = disease_fused = None
    if fusion_model is not None and feature_result is not None:
        print_info("Computing fused disease embeddings...")
        kg_embeddings = extract_all_embeddings(kg_result, device)
        kg_to_row = feature_result["kg_to_row"]
        feature_matrix = feature_result["feature_matrix"]

        d_kg = np.array([kg_embeddings[entity_to_id[d]] for d in valid_diseases], dtype=np.float32)
        d_feat = np.array([
            feature_matrix[kg_to_row[d]] if d in kg_to_row
            else np.zeros(feature_matrix.shape[1], dtype=np.float32)
            for d in valid_diseases
        ], dtype=np.float32)

        with torch.no_grad():
            disease_fused = fusion_model(
                torch.from_numpy(d_kg).to(device),
                torch.from_numpy(d_feat).to(device),
            )

    gnn_target_embs = gnn_disease_embs = None
    if gnn_embeddings is not None and graph_result is not None:
        gid_map = graph_result.get("global_id_map", {})
        gnn_target_embs  = {t: gnn_embeddings[gid_map[str(t)]]
                            for t in valid_targets if str(t) in gid_map}
        gnn_disease_embs = {d: gnn_embeddings[gid_map[str(d)]]
                            for d in valid_diseases if str(d) in gid_map}

    print_info(f"Scoring {len(valid_targets):,} targets...")
    scores_list = []

    for target in tqdm(valid_targets, desc="Scoring", ncols=80):
        t_idx = entity_to_id[target]

        with torch.no_grad():
            d_indices = [entity_to_id[d] for d in valid_diseases]
            best_kg = -1e9
            for d_idx in d_indices:
                for r_idx in relation_range:
                    batch = torch.tensor([[d_idx, r_idx, t_idx]],
                                         device=device, dtype=torch.long)
                    s = model.score_hrt(batch).cpu().item()
                    if s > best_kg:
                        best_kg = s
        kg_score = best_kg

        fusion_score = 0.0
        if disease_fused is not None and kg_embeddings is not None:
            t_emb = torch.from_numpy(
                kg_embeddings[t_idx:t_idx+1].astype(np.float32)
            ).to(device)
            t_norm = F.normalize(t_emb, p=2, dim=-1)
            d_norm = F.normalize(disease_fused, p=2, dim=-1)
            fusion_score = torch.mm(d_norm, t_norm.t()).mean().item()

        gnn_score = 0.0
        if gnn_target_embs and gnn_disease_embs:
            t_gnn = gnn_target_embs.get(target)
            if t_gnn is not None:
                t_v = torch.from_numpy(t_gnn.astype(np.float32)).unsqueeze(0).to(device)
                t_v = F.normalize(t_v, p=2, dim=-1)
                sims = []
                for d in valid_diseases:
                    d_gnn = gnn_disease_embs.get(d)
                    if d_gnn is not None:
                        d_v = torch.from_numpy(d_gnn.astype(np.float32)).unsqueeze(0).to(device)
                        d_v = F.normalize(d_v, p=2, dim=-1)
                        sims.append(torch.mm(d_v, t_v.t()).item())
                if sims:
                    gnn_score = float(np.mean(sims))

        if fusion_model is not None and gnn_embeddings is not None:
            combined = 0.4 * kg_score + 0.3 * fusion_score + 0.3 * gnn_score
        elif fusion_model is not None:
            combined = args.alpha * kg_score + (1 - args.alpha) * fusion_score
        else:
            combined = kg_score

        scores_list.append({
            "target_id":      target,
            "target_name":    entity_info.get(target, {}).get("name", target),
            "combined_score": combined,
            "kg_score":       kg_score,
            "fusion_score":   fusion_score,
            "gnn_score":      gnn_score,
        })

    result_df = pd.DataFrame(scores_list)
    result_df = result_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


# ══════════════════════════════════════════════════════════════════════
# Mode 2 — Link prediction
# ══════════════════════════════════════════════════════════════════════

def mode_link_prediction(args, kg_result, data, device):
    if not args.relation:
        print_error("--relation is required for link_prediction mode.")
        print_info("Run --list-relations to see all available relation types.")
        sys.exit(1)

    if not args.head and not args.tail:
        print_error("Provide either --head (tail prediction) or --tail (head prediction).")
        sys.exit(1)

    model = kg_result.model
    entity_to_id = kg_result.training.entity_to_id
    entity_info   = data["entity_info"]

    rel_id = resolve_relation(args.relation, kg_result)
    if rel_id is None:
        sys.exit(1)

    is_tail_pred = args.head is not None
    anchor_name  = args.head if is_tail_pred else args.tail
    anchor_id    = resolve_entity(anchor_name, data)
    if anchor_id is None:
        sys.exit(1)

    anchor_idx = entity_to_id.get(anchor_id)
    if anchor_idx is None:
        print_error(f"'{anchor_id}' not in KG model entity list.")
        sys.exit(1)

    direction = "tail" if is_tail_pred else "head"
    rel_desc  = RELATION_DESCRIPTIONS.get(args.relation, args.relation)
    print_info(
        f"Link prediction: {direction} prediction | "
        f"{'head' if is_tail_pred else 'tail'}='{anchor_id}' | "
        f"relation='{args.relation}' ({rel_desc})"
    )

    # Type-aware candidate filtering
    candidate_ids = get_type_filtered_candidates(
        rel=args.relation,
        is_tail_pred=is_tail_pred,
        data=data,
        entity_to_id=entity_to_id,
        override_type=args.target_type,
    )
    print_stat(f"Candidate pool", f"{len(candidate_ids):,}")

    # Score all candidates in batches
    print_info(f"Scoring {len(candidate_ids):,} candidate entities...")
    scores_list = []

    model.eval()
    batch_size  = 512
    cand_indices = [entity_to_id[e] for e in candidate_ids if e in entity_to_id]
    cand_names   = [e for e in candidate_ids if e in entity_to_id]

    for start in tqdm(range(0, len(cand_indices), batch_size),
                      desc="Scoring", ncols=80):
        batch_ids   = cand_indices[start:start + batch_size]
        batch_names = cand_names[start:start + batch_size]

        if is_tail_pred:
            triples = torch.tensor(
                [[anchor_idx, rel_id, c] for c in batch_ids],
                device=device, dtype=torch.long
            )
        else:
            triples = torch.tensor(
                [[c, rel_id, anchor_idx] for c in batch_ids],
                device=device, dtype=torch.long
            )

        with torch.no_grad():
            batch_scores = model.score_hrt(triples).cpu().numpy()

        for name, score in zip(batch_names, batch_scores):
            if name == anchor_id:
                continue
            scores_list.append({
                "entity":      name,
                "entity_type": entity_info.get(name, {}).get("type", "unknown"),
                "score":       float(score),
            })

    result_df = pd.DataFrame(scores_list)
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df.head(args.top_k * 5)


# ══════════════════════════════════════════════════════════════════════
# Mode 3 — Embedding similarity
# ══════════════════════════════════════════════════════════════════════

def mode_similarity(args, kg_result, data, device):
    if not args.entity:
        print_error("--entity is required for similarity mode.")
        sys.exit(1)

    query_id = resolve_entity(args.entity, data)
    if query_id is None:
        sys.exit(1)

    entity_to_id = kg_result.training.entity_to_id
    query_idx = entity_to_id.get(query_id)
    if query_idx is None:
        print_error(f"'{query_id}' not found in model entity list.")
        sys.exit(1)

    print_info(f"Computing similarity for: '{query_id}'")
    entity_info = data["entity_info"]

    print_info("Extracting embedding matrix...")
    all_emb   = extract_all_embeddings(kg_result, device)
    all_emb_t = torch.from_numpy(all_emb.astype(np.float32)).to(device)
    all_norm  = F.normalize(all_emb_t, p=2, dim=-1)

    query_emb = all_norm[query_idx:query_idx + 1]
    sims = torch.mm(query_emb, all_norm.t()).squeeze(0).cpu().numpy()

    id_to_entity = {v: k for k, v in entity_to_id.items()}
    candidates = []
    for idx, sim in enumerate(sims):
        eid = id_to_entity.get(idx, "")
        if eid == query_id:
            continue
        etype = entity_info.get(eid, {}).get("type", "unknown")
        if args.filter_type and etype.lower() != args.filter_type.lower():
            continue
        candidates.append({"entity": eid, "entity_type": etype, "similarity": float(sim)})

    result_df = pd.DataFrame(candidates)
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


# ══════════════════════════════════════════════════════════════════════
# Info commands
# ══════════════════════════════════════════════════════════════════════

def cmd_list_relations(data: dict):
    df = data["dataframe"]
    rel_counts = df["relation"].value_counts()
    print_banner("Available Relations")
    print(f"\n  {'Code':<8} {'Description':<60} {'Triples':>10}")
    print(f"  {'-'*8}   {'-'*60}   {'-'*10}")
    for rel, count in rel_counts.items():
        desc = RELATION_DESCRIPTIONS.get(rel, "(no description)")
        # Truncate for display
        desc_short = desc[:57] + "..." if len(desc) > 60 else desc
        print(f"  {rel:<8}   {desc_short:<60}   {count:>10,}")
    print()


def cmd_list_entities(data: dict, filter_type: str = None):
    df = data["dataframe"]
    if "x_type" not in df.columns:
        print_info("No type information available in this dataset.")
        return

    all_ents = pd.DataFrame({
        "entity": pd.concat([df["x_id"], df["y_id"]]),
        "type":   pd.concat([df["x_type"], df["y_type"]]),
    }).drop_duplicates("entity")

    if filter_type:
        all_ents = all_ents[all_ents["type"].str.lower() == filter_type.lower()]
        print_banner(f"Entities of type: {filter_type} ({len(all_ents):,} total)")
        for _, row in all_ents.sort_values("entity").head(100).iterrows():
            print(f"    {row['entity']}")
        if len(all_ents) > 100:
            print(f"    ... and {len(all_ents)-100} more")
    else:
        print_banner("Entity Type Distribution")
        for etype, grp in all_ents.groupby("type"):
            print_stat(etype, f"{len(grp):,}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════

def print_results_table(df: pd.DataFrame, top_k: int, title: str):
    print_banner(title)
    show = df.head(top_k)
    print(show.to_string(index=False))
    print()


def save_results(df: pd.DataFrame, output_path, default_name: str, work_dir: str):
    if output_path is None:
        output_path = str(Path(work_dir) / default_name)
    df.to_csv(output_path, index=False)
    print_success(f"Results saved to: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print_banner("immuneKG Prediction")
    print_info(f"Mode : {args.mode}")
    print_info(f"Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config   = load_config(args.config)
    device   = setup_device(config)
    work_dir = config["output"]["work_dir"]

    if args.list_relations or args.list_entities:
        _, _, data, *_ = load_models(config, device, kg_only=True)
        if args.list_relations:
            cmd_list_relations(data)
        if args.list_entities:
            cmd_list_entities(data, args.entity_type)
        return

    (kg_result, fusion_model, data,
     feature_result, gnn_embeddings,
     degree_map, graph_result) = load_models(config, device, args.kg_only)

    if args.mode == "target_scoring":
        print_stage(1, 1, "Target scoring")
        result_df = mode_target_scoring(
            args, kg_result, fusion_model, data,
            feature_result, gnn_embeddings,
            degree_map, graph_result, device
        )
        if len(result_df) == 0:
            return
        print_results_table(
            result_df[["rank", "target_name", "combined_score", "kg_score",
                        "fusion_score", "gnn_score"]],
            top_k=args.top_k,
            title=f"Top {args.top_k} Targets",
        )
        save_results(result_df, args.output,
                     "prediction_target_scoring.csv", work_dir)

    elif args.mode == "link_prediction":
        direction = "tail" if args.head else "head"
        anchor    = args.head or args.tail
        rel_desc  = RELATION_DESCRIPTIONS.get(args.relation, args.relation)
        print_stage(1, 1,
                    f"Link prediction ({direction}) — {anchor} "
                    f"—[{args.relation}: {rel_desc}]-> ?")

        result_df = mode_link_prediction(args, kg_result, data, device)
        if len(result_df) == 0:
            return
        show_df = result_df.head(args.top_k)
        print_results_table(
            show_df, top_k=args.top_k,
            title=f"Top {args.top_k} Predicted {'Tails' if direction == 'tail' else 'Heads'}"
        )
        save_results(show_df, args.output,
                     f"prediction_link_{anchor}_{args.relation}.csv", work_dir)

    elif args.mode == "similarity":
        print_stage(1, 1, f"Embedding similarity — query: '{args.entity}'")
        result_df = mode_similarity(args, kg_result, data, device)
        if len(result_df) == 0:
            return
        show_df = result_df.head(args.top_k)
        print_results_table(
            show_df, top_k=args.top_k,
            title=f"Top {args.top_k} Similar Entities to '{args.entity}'"
        )
        save_results(show_df, args.output,
                     f"prediction_similarity_{args.entity}.csv", work_dir)

    print_success("Done.")


if __name__ == "__main__":
    main()
