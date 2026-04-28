"""
immuneKG Data Loader

Loads the underlying KG from pre-split train/valid/test TSV files.

File format:
    train.tsv / valid.tsv / test.tsv
    Each line: head<TAB>relation<TAB>tail (no header)
    Entities: lowercase English (e.g., "colitis", "tnf", "methotrexate")
    Relations: letter codes (P/U/D/ML/GG/T/I etc., 28 types)

Relation semantics (disease<->gene relevant):
    P  = gene causes phenotype/disease
    U  = gene upregulates disease
    D  = gene downregulates disease
    ML = molecule expressed in disease context
    I  = inhibits
    T  = treats (drug->disease)
"""

import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pykeen.triples import TriplesFactory

from .utils import (
    Timer, print_banner, print_stage, print_success, print_warning,
    print_info, print_stat, print_error, checkpoint_exists,
    save_checkpoint, load_checkpoint
)


# IBD-related entity sets for disease matching
IBD_ENTITIES = {
    "colitis",
    "inflammatory bowel diseases",
    "enterocolitis",
    "crohn disease",
    "ulcerative colitis",
}

IBD_GROUPS = {
    "ulcerative colitis":  ["ulcerative colitis", "colitis"],
    "crohn":               ["crohn disease", "enterocolitis"],
    "inflammatory bowel":  ["inflammatory bowel diseases"],
}

# Disease<->gene relevant relation types
DISEASE_GENE_RELATIONS = {
    "P", "U", "D", "ML", "I", "C", "O", "A", "K", "Pr", "N", "Q",
}


class KGDataLoader:
    """
    Knowledge graph data loader for pre-split TSV files.

    Reads train/valid/test.tsv, merges into a unified graph,
    infers entity types, and identifies IBD disease nodes.
    """

    def __init__(self, config: dict, work_dir: str):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = config.get('data', {})
        self.train_path = data_cfg.get('train_path', 'data/train.tsv')
        self.valid_path = data_cfg.get('valid_path', 'data/valid.tsv')
        self.test_path  = data_cfg.get('test_path',  'data/test.tsv')

        if 'kg_path' in data_cfg and not Path(self.train_path).exists():
            print_warning("train.tsv not found, inferring from kg_path...")
            kg_dir = Path(data_cfg['kg_path']).parent
            self.train_path = str(kg_dir / 'train.tsv')
            self.valid_path = str(kg_dir / 'valid.tsv')
            self.test_path  = str(kg_dir / 'test.tsv')

        self.autoimmune_keywords = config['prediction'].get('autoimmune_keywords', [])
        self.target_entity_types = config['prediction'].get('target_entity_types', ['gene/protein'])

        filenames = config['output']['filenames']
        self.processed_path = self.work_dir / filenames['processed_data']
        self.triples_path   = self.work_dir / filenames['triples_factory']

    def load_and_process(self) -> dict:
        """
        Load train/valid/test TSV files and merge into a complete graph.

        Returns:
            dict with keys: dataframe, entities, relations, disease_entities,
                            entity_info, split_info
        """
        if checkpoint_exists(self.processed_path):
            print_success("Found processed KG checkpoint, loading...")
            data = load_checkpoint(self.processed_path, "KG preprocessed data")
            self._print_data_summary(data)
            return data

        print_info("Loading KG from TSV files...")
        start_time = time.time()

        for p in [self.train_path, self.valid_path, self.test_path]:
            if not Path(p).exists():
                print_error(f"File not found: {p}")
                print_info("Please place train.tsv / valid.tsv / test.tsv in the data/ directory")
                raise FileNotFoundError(p)

        cols = ['x_id', 'relation', 'y_id']
        train_df = pd.read_csv(self.train_path, sep='\t', header=None, names=cols)
        valid_df = pd.read_csv(self.valid_path, sep='\t', header=None, names=cols)
        test_df  = pd.read_csv(self.test_path,  sep='\t', header=None, names=cols)

        print_stat("Train triples", f"{len(train_df):,}")
        print_stat("Valid triples", f"{len(valid_df):,}")
        print_stat("Test triples",  f"{len(test_df):,}")

        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        df['x_id']     = df['x_id'].astype(str).str.strip().str.lower()
        df['y_id']     = df['y_id'].astype(str).str.strip().str.lower()
        df['relation'] = df['relation'].astype(str).str.strip()

        print_stat("Total triples (merged)", f"{len(df):,}")

        entities  = sorted(set(df['x_id'].unique()) | set(df['y_id'].unique()))
        relations = sorted(df['relation'].unique().tolist())
        print_stat("Total entities", f"{len(entities):,}")
        print_stat("Relation types", f"{len(relations)}")

        print_info("Inferring entity types...")
        entity_type_map = self._infer_entity_types(entities, df)

        df['x_name'] = df['x_id']
        df['y_name'] = df['y_id']
        df['x_type'] = df['x_id'].map(entity_type_map).fillna('unknown')
        df['y_type'] = df['y_id'].map(entity_type_map).fillna('unknown')

        type_counts = pd.Series(entity_type_map).value_counts()
        print_info("Entity type distribution:")
        for t, c in type_counts.items():
            print_stat(f"  {t}", f"{c:,}")

        print_info("Identifying IBD disease entities...")
        disease_entities = self._identify_autoimmune_diseases(df, entity_type_map)

        entity_info = {
            eid: {'name': eid, 'type': entity_type_map.get(eid, 'unknown'), 'source': 'KG'}
            for eid in entities
        }

        for split_df in [train_df, valid_df, test_df]:
            split_df['x_id'] = split_df['x_id'].str.lower()
            split_df['y_id'] = split_df['y_id'].str.lower()

        data = {
            'dataframe':        df,
            'entities':         entities,
            'relations':        relations,
            'disease_entities': disease_entities,
            'entity_info':      entity_info,
            'split_info': {
                'train': train_df,
                'valid': valid_df,
                'test':  test_df,
            }
        }

        save_checkpoint(data, self.processed_path, "KG preprocessed data")
        print_success(f"Data preprocessing complete ({time.time()-start_time:.2f}s)")
        self._print_data_summary(data)
        return data

    def build_triples_factory(self, data: dict):
        """
        Build PyKEEN TriplesFactory from the merged graph.

        Uses the pre-split train/valid/test partitions.
        """
        if checkpoint_exists(self.triples_path):
            print_success("Found TriplesFactory checkpoint, loading...")
            tf = load_checkpoint(self.triples_path, "TriplesFactory")
            print_stat("Triples", f"{tf.num_triples:,}")
            return tf

        print_info("Building PyKEEN TriplesFactory (using pre-split data)...")
        start_time = time.time()

        df = data['dataframe']
        triples = df[['x_id', 'relation', 'y_id']].values

        all_entities  = sorted(set(triples[:, 0]) | set(triples[:, 2]))
        all_relations = sorted(set(triples[:, 1]))

        entity_to_id   = {e: i for i, e in enumerate(all_entities)}
        relation_to_id = {r: i for i, r in enumerate(all_relations)}

        kg_cfg = self.config.get('training', {}).get('kg', {})
        create_inverse_triples = bool(kg_cfg.get('create_inverse_triples', True))

        tf = TriplesFactory.from_labeled_triples(
            triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=create_inverse_triples,
        )

        print_stat("TriplesFactory triples",   f"{tf.num_triples:,}")
        print_stat("TriplesFactory entities",  f"{tf.num_entities:,}")
        print_stat("TriplesFactory relations", f"{tf.num_relations}")

        save_checkpoint(tf, self.triples_path, "TriplesFactory")
        print_success(f"TriplesFactory built ({time.time()-start_time:.2f}s)")
        return tf

    def build_split_triples_factories(self, data: dict, triples_factory):
        """Build separate TriplesFactory objects for train/valid/test splits."""
        split_info = data.get('split_info', {})
        if not split_info:
            print_warning("No split info found, using random 80/10/10 split")
            return triples_factory.split([0.8, 0.1, 0.1])

        entity_to_id   = triples_factory.entity_to_id
        relation_to_id = triples_factory.relation_to_id

        kg_cfg = self.config.get('training', {}).get('kg', {})
        create_inverse = bool(kg_cfg.get('create_inverse_triples', True))

        factories = {}
        for split_name, split_df in split_info.items():
            triples = split_df[['x_id', 'relation', 'y_id']].values
            valid_mask = (
                pd.Series(triples[:, 0]).isin(entity_to_id) &
                pd.Series(triples[:, 2]).isin(entity_to_id) &
                pd.Series(triples[:, 1]).isin(relation_to_id)
            ).values
            triples = triples[valid_mask]

            if len(triples) == 0:
                print_warning(f"Split '{split_name}' has no valid triples after filtering")
                continue

            factories[split_name] = TriplesFactory.from_labeled_triples(
                triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
                create_inverse_triples=create_inverse,
            )
            print_stat(f"{split_name} triples", f"{len(triples):,}")

        train_tf = factories.get('train')
        valid_tf = factories.get('valid')
        test_tf  = factories.get('test')

        # Warn on cold-start entities
        if valid_tf and test_tf:
            train_ents = set(train_tf.entity_to_id.keys())
            valid_unseen = sum(1 for e in valid_tf.entity_to_id if e not in train_ents)
            test_unseen  = sum(1 for e in test_tf.entity_to_id  if e not in train_ents)
            if valid_unseen or test_unseen:
                print_warning(
                    f"Cold-start entities detected: "
                    f"valid={valid_unseen}, test={test_unseen} "
                    "(may lower Hits@1/MRR)"
                )

        return train_tf, valid_tf, test_tf

    def get_target_entities(self, data: dict) -> set:
        """Return all candidate target (gene/protein) entities."""
        df = data['dataframe']
        targets = set()
        if 'x_type' in df.columns:
            for t_type in self.target_entity_types:
                found = (
                    set(df[df['x_type'] == t_type]['x_id'].unique()) |
                    set(df[df['y_type'] == t_type]['y_id'].unique())
                )
                targets.update(found)
                if found:
                    print_stat(f"Type '{t_type}'", f"{len(found):,} targets")
        else:
            disease_set = set(data['disease_entities'])
            all_ent = set(df['x_id'].unique()) | set(df['y_id'].unique())
            targets = all_ent - disease_set

        print_stat("Total candidate targets", f"{len(targets):,}")
        return targets

    def get_prediction_disease_entities(self, data: dict) -> dict:
        """Return IBD-related disease nodes by exact matching."""
        entities_set = set(data['entities'])
        entity_info  = data['entity_info']

        print_info("=== IBD node exact matching ===")
        result = {}
        for group_name, entity_list in IBD_GROUPS.items():
            valid = []
            for eid in entity_list:
                eid_lower = eid.lower()
                if eid_lower in entities_set:
                    valid.append(eid_lower)
                    name = entity_info.get(eid_lower, {}).get('name', eid_lower)
                    print(f"    [OK] {name}  (id:{eid_lower})")
                else:
                    print_warning(f"  '{eid_lower}' not in graph, skipping")
            result[group_name] = valid
            print_stat(f"  '{group_name}'", f"{len(valid)} nodes")

        total = len(set(eid for ids in result.values() for eid in ids))
        print_success(f"Confirmed {total} IBD disease nodes")
        return result

    def _infer_entity_types(self, entities: list, df: pd.DataFrame) -> dict:
        """Infer entity types from relation structure."""
        entity_type = {}

        # Gene/protein from GG relation
        for eid in list(df[df['relation'] == 'GG']['x_id'].unique()) + \
                   list(df[df['relation'] == 'GG']['y_id'].unique()):
            entity_type[eid] = 'gene/protein'

        # Gene/protein from Rg relation
        for eid in list(df[df['relation'] == 'Rg']['x_id'].unique()) + \
                   list(df[df['relation'] == 'Rg']['y_id'].unique()):
            if eid not in entity_type:
                entity_type[eid] = 'gene/protein'

        # Drug from CC relation
        for eid in list(df[df['relation'] == 'CC']['x_id'].unique()) + \
                   list(df[df['relation'] == 'CC']['y_id'].unique()):
            if eid not in entity_type:
                entity_type[eid] = 'drug'

        # Disease from T (treats) tail
        for eid in df[df['relation'] == 'T']['y_id'].unique():
            if eid not in entity_type:
                entity_type[eid] = 'disease'

        # Disease/gene from P (phenotype) relation
        for eid in df[df['relation'] == 'P']['y_id'].unique():
            if eid not in entity_type:
                entity_type[eid] = 'disease'
        for eid in df[df['relation'] == 'P']['x_id'].unique():
            if eid not in entity_type:
                entity_type[eid] = 'gene/protein'

        # Gene/disease from U/D relations
        for rel in ['U', 'D']:
            rel_df = df[df['relation'] == rel]
            for eid in rel_df['x_id'].unique():
                if eid not in entity_type:
                    entity_type[eid] = 'gene/protein'
            for eid in rel_df['y_id'].unique():
                if eid not in entity_type:
                    entity_type[eid] = 'disease'

        # Gene from ML relation
        for eid in df[df['relation'] == 'ML']['x_id'].unique():
            if eid not in entity_type:
                entity_type[eid] = 'gene/protein'

        # Fallback heuristics
        for eid in entities:
            if eid not in entity_type:
                if ' ' in eid and len(eid) > 15:
                    entity_type[eid] = 'disease'
                elif len(eid) <= 8 and eid.isalpha():
                    entity_type[eid] = 'gene/protein'
                else:
                    entity_type[eid] = 'drug'

        return entity_type

    def _identify_autoimmune_diseases(self, df: pd.DataFrame,
                                       entity_type_map: dict) -> list:
        """Identify autoimmune disease nodes for training."""
        disease_entities = set()
        all_entities = set(df['x_id'].unique()) | set(df['y_id'].unique())

        for eid in IBD_ENTITIES:
            if eid.lower() in all_entities:
                disease_entities.add(eid.lower())

        autoimmune_keywords = self.autoimmune_keywords or [
            'lupus', 'arthritis', 'sclerosis', 'psoriasis',
            'colitis', 'crohn', 'bowel', 'autoimmune',
            'diabetes', 'celiac', 'myasthenia'
        ]
        for eid in all_entities:
            for kw in autoimmune_keywords:
                if kw.lower() in eid.lower():
                    disease_entities.add(eid)
                    break

        result = sorted(disease_entities)
        print_success(f"Identified {len(result)} autoimmune disease entities")
        for eid in result:
            print(f"    |  {eid}")
        return result

    def _print_data_summary(self, data: dict):
        print_info("=== KG Data Summary ===")
        print_stat("Total entities",     f"{len(data['entities']):,}")
        print_stat("Relation types",     f"{len(data['relations'])}")
        print_stat("Disease entities",   f"{len(data['disease_entities'])}")
        if data['disease_entities']:
            print_info(f"All disease entities ({len(data['disease_entities'])} total):")
            for eid in data['disease_entities']:
                print(f"    |  {eid}")
