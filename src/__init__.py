"""
immuneKG — Immune Disease Knowledge Graph Embedding Target Discovery System v2

Modules:
    utils:           Utility functions (config, logging, GPU, timer)
    data_loader:     KG data loading and preprocessing
    feature_encoder: Five-dimensional disease feature encoding
    model:           Fusion network definitions (FFN + MultiSourceFFN)
    trainer:         KG embedding + fusion network training
    scorer:          Target scoring and ranking
    graph_builder:   PyG graph construction (heterogeneous + homogeneous)
    gnn_module:      HeteroPNA-Attn GNN (PNA aggregation + heterogeneous attention)
    novelty:         Novelty scoring (degree penalty + AP metric)
"""
