"""
immuneKG 

 、 、 :
 1. GWAS (gwas_genetic_features.csv)
 2. HPO (hpo_organ_features.csv)
 3. HPO (hpo_phenotype_stats.csv)
 4. IEDB (iedb_onehot_features.csv)
 5. IEDB (iedb_statistical_features.csv)

 mondo_id ， 。
"""

import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from .utils import (
    print_banner, print_success, print_warning, print_info,
    print_stat, print_error, checkpoint_exists,
    save_checkpoint, load_checkpoint
)


class DiseaseFeatureEncoder:
    """
 
 
 :
 1. 5 CSV 
 2. mondo_id 
 3. 
 4. 
 5. mondo_id ↔ KG ID 
 
 Attributes:
 feature_matrix: np.ndarray, shape=(num_diseases, total_feature_dim)
 feature_dims: dict, { : }
 mondo_to_kg_map: dict, {mondo_id: kg_entity_id}
 disease_ids: list, KG ID 
 """
    
    def __init__(self, config: dict, work_dir: str):
        """
 Args:
 config: 
 work_dir: 
 """
        self.config = config
        self.work_dir = Path(work_dir)
        self.feature_dir = Path(config['data']['feature_dir'])
        self.feature_files = config['data']['feature_files']
        self.mondo_prefix = config['data'].get('mondo_id_prefix', '')
        
        # 
        self.features_path = self.work_dir / config['output']['filenames']['disease_features']
        
        # （ ）
        self.feature_dims = {}
        self.total_feature_dim = 0
        self.scalers = {}  # 
    
    # ================================================================
    # ： 
    # ================================================================
    
    def encode_all_features(self, data: dict) -> dict:
        """
 
 
 Args:
 data: KGDataLoader.load_and_process() 
 
 Returns:
 :
 - feature_matrix: np.ndarray, 
 - feature_dims: dict, { : }
 - total_feature_dim: int, 
 - disease_kg_ids: list, KG ID（ ）
 - mondo_to_kg: dict, mondo_id → KG ID 
 - kg_to_row: dict, KG ID → 
 - feature_names: list, 
 - scalers: dict, Scaler 
 """
        # === ===
        if checkpoint_exists(self.features_path):
            print_success(" ， ...")
            result = load_checkpoint(self.features_path, " ")
            self._print_feature_summary(result)
            return result
        
        print_info(" ...")
        start_time = time.time()
        
        # ---- 1: mondo_id ↔ KG ID ----
        mondo_to_kg = self._build_mondo_mapping(data)
        
        # ---- 2: ----
        feature_groups = {}
        
        # (a) GWAS 
        print_info(" GWAS ...")
        gwas_df = self._load_feature_file(
            self.feature_files['gwas_genetic'],
            numeric_cols=None,  # 
            name="GWAS "
        )
        if gwas_df is not None:
            feature_groups['gwas_genetic'] = gwas_df
        
        # (b) HPO 
        print_info(" HPO ...")
        hpo_organ_df = self._load_feature_file(
            self.feature_files['hpo_organ'],
            numeric_cols=None,
            name="HPO "
        )
        if hpo_organ_df is not None:
            feature_groups['hpo_organ'] = hpo_organ_df
        
        # (c) HPO 
        print_info(" HPO ...")
        hpo_pheno_df = self._load_feature_file(
            self.feature_files['hpo_phenotype'],
            numeric_cols=None,
            name="HPO "
        )
        if hpo_pheno_df is not None:
            feature_groups['hpo_phenotype'] = hpo_pheno_df
        
        # (d) IEDB 
        print_info(" IEDB ...")
        iedb_oh_df = self._load_feature_file(
            self.feature_files['iedb_onehot'],
            numeric_cols=None,
            name="IEDB "
        )
        if iedb_oh_df is not None:
            feature_groups['iedb_onehot'] = iedb_oh_df
        
        # (e) IEDB 
        print_info(" IEDB ...")
        iedb_stat_df = self._load_feature_file(
            self.feature_files['iedb_statistical'],
            numeric_cols=None,
            name="IEDB "
        )
        if iedb_stat_df is not None:
            feature_groups['iedb_statistical'] = iedb_stat_df
        
        if not feature_groups:
            print_error(" ！ 。")
            return self._empty_result()
        
        # ---- 3: mondo_id ----
        print_info(" ...")
        aligned_features, feature_names, all_mondo_ids = self._align_and_normalize(
            feature_groups, mondo_to_kg
        )
        
        # ---- 4: ----
        disease_kg_ids = []
        kg_to_row = {}
        
        for row_idx, mondo_id in enumerate(all_mondo_ids):
            kg_id = mondo_to_kg.get(str(mondo_id))
            if kg_id is not None:
                disease_kg_ids.append(kg_id)
                kg_to_row[kg_id] = row_idx
        
        # KG mondo_id， （ ）
        print_stat(" KG ", f"{len(disease_kg_ids)}")
        print_stat(" ", f"{len(all_mondo_ids)}")
        
        # ---- 5: ----
        result = {
            'feature_matrix': aligned_features,
            'feature_dims': dict(self.feature_dims),
            'total_feature_dim': self.total_feature_dim,
            'disease_kg_ids': disease_kg_ids,
            'mondo_to_kg': mondo_to_kg,
            'kg_to_row': kg_to_row,
            'feature_names': feature_names,
            'all_mondo_ids': all_mondo_ids,
            'scalers': self.scalers,
        }
        
        save_checkpoint(result, self.features_path, " ")
        
        duration = time.time() - start_time
        print_success(f" ，  {duration:.2f}  ")
        
        self._print_feature_summary(result)
        
        return result
    
    # ================================================================
    # 
    # ================================================================
    
    def _load_feature_file(self, filename: str, numeric_cols: list = None,
                           name: str = "") -> Optional[pd.DataFrame]:
        """
 CSV 
 
 mondo_id ， 。
 
 Args:
 filename: 
 numeric_cols: （None ）
 name: （ ）
 
 Returns:
 mondo_id DataFrame， None（ ）
 """
        filepath = self.feature_dir / filename
        
        if not filepath.exists():
            print_warning(f" : {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print_error(f"  {filename}  : {e}")
            return None
        
        print_stat(f"  {name}  ", f"{len(df)}")
        print_stat(f"  {name}  ", f"{len(df.columns)}")
        
        # mondo_id 
        if 'mondo_id' not in df.columns:
            print_warning(f"  {filename}   mondo_id  ， : {list(df.columns)[:10]}")
            return None
        
        # mondo_id 
        df['mondo_id'] = df['mondo_id'].astype(str).str.strip()
        df = df.set_index('mondo_id')
        
        # （ disease_name ）
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                text_cols.append(col)
        
        if text_cols:
            print_info(f"   : {text_cols}")
            df = df.drop(columns=text_cols)
        
        # 
        if numeric_cols is not None:
            available = [c for c in numeric_cols if c in df.columns]
            df = df[available]
        
        # 0
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print_info(f"    {nan_count}   0")
            df = df.fillna(0)
        
        # float
        df = df.astype(float)
        
        print_stat(f"  {name}  ", f"{df.shape[0]}   × {df.shape[1]}  ")
        
        return df
    
    # ================================================================
    # 
    # ================================================================
    
    def _align_and_normalize(self, feature_groups: dict, mondo_to_kg: dict
                              ) -> Tuple[np.ndarray, list, list]:
        """
 mondo_id 、 ， 
 
 :
 - mondo_id 
 - 0 
 - StandardScaler 
 - （ HPO 、IEDB ） 
 
 Args:
 feature_groups: { : DataFrame} 
 mondo_to_kg: mondo_id → KG ID 
 
 Returns:
 ( , , mondo_id )
 """
        # mondo_id 
        all_mondo_ids = set()
        for group_name, df in feature_groups.items():
            all_mondo_ids.update(df.index.tolist())
        all_mondo_ids = sorted(all_mondo_ids, key=lambda x: int(x) if x.isdigit() else x)
        
        print_stat(" mondo_id ", f"{len(all_mondo_ids)}")
        
        # 
        aligned_parts = []
        all_feature_names = []
        
        # （ ）
        binary_groups = {'hpo_organ', 'iedb_onehot'}
        
        for group_name, df in feature_groups.items():
            print_info(f"   : {group_name} ({df.shape[1]}  )")
            
            # mondo_id
            aligned_df = df.reindex(all_mondo_ids, fill_value=0.0)
            
            feature_values = aligned_df.values.astype(np.float32)
            col_names = [f"{group_name}__{col}" for col in aligned_df.columns]
            
            # （ StandardScaler）
            if group_name not in binary_groups:
                scaler = StandardScaler()
                feature_values = scaler.fit_transform(feature_values)
                self.scalers[group_name] = scaler
                print_info(f"    → StandardScaler  ")
            else:
                print_info(f"    →  ， ")
            
            # 
            self.feature_dims[group_name] = feature_values.shape[1]
            
            aligned_parts.append(feature_values)
            all_feature_names.extend(col_names)
        
        # 
        feature_matrix = np.hstack(aligned_parts).astype(np.float32)
        self.total_feature_dim = feature_matrix.shape[1]
        
        print_info(f"   : {feature_matrix.shape[0]}   × {feature_matrix.shape[1]}  ")
        
        # 
        print_info(" :")
        for gname, gdim in self.feature_dims.items():
            print_stat(f"    {gname}", f"{gdim}  ")
        
        return feature_matrix, all_feature_names, all_mondo_ids
    
    # ================================================================
    # MONDO / （immuneKG ↔ disease_name）
    # ================================================================

    @staticmethod
    def _mondo_id_variants(mondo_str: str) -> List[str]:
        """ MONDO CSV 5101 / 0005101 ， 。"""
        s = str(mondo_str).strip()
        if not s:
            return []
        out = [s]
        if s.isdigit():
            no0 = s.lstrip('0') or '0'
            if no0 not in out:
                out.append(no0)
            z7 = s.zfill(7)
            if z7 not in out:
                out.append(z7)
        return list(dict.fromkeys(out))

    @staticmethod
    def _normalize_disease_label(text: str) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        s = str(text).strip().lower()
        s = s.replace(",", " ")
        return " ".join(s.split())

    @classmethod
    def _disease_label_token_set(cls, text: str) -> frozenset:
        lab = cls._normalize_disease_label(text)
        if not lab:
            return frozenset()
        return frozenset(lab.split())

    def _register_mondo_to_kg(
        self, mondo_to_kg: dict, mondo_id_str: str, kg_entity_id: str, source: str
    ) -> bool:
        """
 mondo KG ； False。
 """
        for key in self._mondo_id_variants(mondo_id_str):
            existing = mondo_to_kg.get(key)
            if existing is not None and existing != kg_entity_id:
                print_warning(
                    f"  MONDO   [{source}]: '{key}'   → '{existing}'，"
                    f"  → '{kg_entity_id}'"
                )
                return False
        for key in self._mondo_id_variants(mondo_id_str):
            mondo_to_kg[key] = kg_entity_id
        return True

    def _extend_mondo_mapping_from_feature_names(
        self, mondo_to_kg: dict, kg_disease_ids: Set[str]
    ) -> int:
        """
 immuneKG ID ， mondo_id 。
 immunekg_to_mondo ， disease_name（ ）
 ** ** ， mondo → KG 。
 """
        # ： 、token → ID（ ）
        norm_to_kg: Dict[str, str] = {}
        tokens_to_kgs: Dict[frozenset, List[str]] = {}
        for kg_id in kg_disease_ids:
            n = self._normalize_disease_label(kg_id)
            if n and n not in norm_to_kg:
                norm_to_kg[n] = kg_id
            ts = self._disease_label_token_set(kg_id)
            if ts:
                tokens_to_kgs.setdefault(ts, []).append(kg_id)

        added = 0
        seen_pairs = set()

        for fname in self.feature_files.values():
            filepath = self.feature_dir / fname
            if not filepath.exists():
                continue
            try:
                header = pd.read_csv(filepath, nrows=0)
            except Exception:
                continue
            cols = list(header.columns)
            if "mondo_id" not in cols:
                continue
            name_col = None
            if "disease_name" in cols:
                name_col = "disease_name"
            elif "mondo_disease_name" in cols:
                name_col = "mondo_disease_name"
            if not name_col:
                continue

            try:
                df = pd.read_csv(filepath, usecols=["mondo_id", name_col], dtype=str)
            except Exception:
                df = pd.read_csv(filepath, dtype=str)
                if "mondo_id" not in df.columns or name_col not in df.columns:
                    continue

            for _, row in df.iterrows():
                mid = str(row["mondo_id"]).strip()
                label = row[name_col]
                if label is None or (isinstance(label, float) and np.isnan(label)):
                    continue
                label_s = str(label).strip()
                if not mid or not label_s:
                    continue

                kg_match = None
                nlab = self._normalize_disease_label(label_s)
                if nlab in norm_to_kg:
                    kg_match = norm_to_kg[nlab]
                else:
                    ts = self._disease_label_token_set(label_s)
                    if ts and ts in tokens_to_kgs:
                        cands = tokens_to_kgs[ts]
                        if len(cands) == 1:
                            kg_match = cands[0]

                if kg_match is None:
                    continue

                pair = (mid, kg_match)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if self._register_mondo_to_kg(mondo_to_kg, mid, kg_match, fname):
                    added += 1

        if added > 0:
            print_stat(" → immuneKG （ ）", f"{added}  ")
        return added

    # ================================================================
    # mondo_id ↔ KG ID 
    # ================================================================
    
    def _build_mondo_mapping(self, data: dict) -> dict:
        """
 mondo_id KG ID 

 ：
 - PrimeKG ： ID ， mondo_id 
 - immuneKG ： ID ， config 
 immunekg_to_mondo →mondo_id→KG 

 Args:
 data: KG 

 Returns:
 {mondo_id_str: kg_entity_id} 
 """
        print_info(" mondo_id ↔ KG ID ...")

        entity_info = data['entity_info']
        entities    = data['entities']
        kg_disease_ids: Set[str] = {
            eid for eid, inf in entity_info.items()
            if inf.get('type') == 'disease'
        }

        mondo_to_kg = {}

        # ── immuneKG ： immunekg_to_mondo ──────────────
        immunekg_to_mondo = self.config.get('data', {}).get('immunekg_to_mondo', {})
        if immunekg_to_mondo:
            print_info("immuneKG ： immunekg_to_mondo ...")
            entity_set = set(entities)
            immunekg_count = 0
            for kg_name, mondo_id in immunekg_to_mondo.items():
                kg_name_lower = kg_name.lower()
                if kg_name_lower in entity_set:
                    ok = self._register_mondo_to_kg(
                        mondo_to_kg, str(mondo_id), kg_name_lower, "immunekg_to_mondo"
                    )
                    if ok:
                        immunekg_count += 1
                        print(f"    ✓ '{kg_name_lower}' → mondo_id={mondo_id}")
                else:
                    print_warning(f"  immuneKG   '{kg_name_lower}'  ， ")
            print_stat("immuneKG→MONDO ", f"{immunekg_count}")

        # immuneKG： ； 
        # （ immunekg return， mondo→ ）
        if kg_disease_ids:
            print_info(
                "immuneKG： CSV disease_name / mondo_disease_name "
                " ， mondo→KG ..."
            )
            self._extend_mondo_mapping_from_feature_names(mondo_to_kg, kg_disease_ids)

        # ── PrimeKG ： ID ， ─────────────────
        direct_count = 0
        for eid in entities:
            clean_id = eid.strip()
            if clean_id.isdigit():
                mondo_to_kg[clean_id] = eid
                direct_count += 1
            elif clean_id.upper().startswith('MONDO:'):
                mondo_num = clean_id.split(':')[-1].lstrip('0') or '0'
                mondo_to_kg[mondo_num] = eid
                direct_count += 1

        if direct_count > 0:
            print_stat(" / ", f"{direct_count}  ")

        df = data['dataframe']
        if 'x_type' in df.columns:
            disease_df = pd.concat([
                df[df['x_type'] == 'disease'][['x_id', 'x_name']].rename(
                    columns={'x_id': 'id', 'x_name': 'name'}),
                df[df['y_type'] == 'disease'][['y_id', 'y_name']].rename(
                    columns={'y_id': 'id', 'y_name': 'name'})
            ]).drop_duplicates(subset='id')
            for _, row in disease_df.iterrows():
                eid = str(row['id']).strip()
                if eid.isdigit() and eid not in mondo_to_kg:
                    mondo_to_kg[eid] = eid

        print_stat(" ", f"{len(mondo_to_kg)}")
        return mondo_to_kg
    
    # ================================================================
    # KG 
    # ================================================================
    
    def get_entity_features(self, feature_result: dict, kg_entity_ids: list) -> torch.Tensor:
        """
 KG 
 
 ， 。
 
 Args:
 feature_result: encode_all_features() 
 kg_entity_ids: KG ID 
 
 Returns:
 , shape=(len(kg_entity_ids), total_feature_dim)
 """
        feature_matrix = feature_result['feature_matrix']
        kg_to_row = feature_result['kg_to_row']
        total_dim = feature_result['total_feature_dim']
        
        result = np.zeros((len(kg_entity_ids), total_dim), dtype=np.float32)
        
        matched = 0
        for i, eid in enumerate(kg_entity_ids):
            if eid in kg_to_row:
                row_idx = kg_to_row[eid]
                result[i] = feature_matrix[row_idx]
                matched += 1
        
        print_info(f" : {matched}/{len(kg_entity_ids)}  ")
        
        return torch.from_numpy(result)
    
    # ================================================================
    # （ ）
    # ================================================================
    
    def _empty_result(self) -> dict:
        """ """
        return {
            'feature_matrix': np.zeros((0, 0), dtype=np.float32),
            'feature_dims': {},
            'total_feature_dim': 0,
            'disease_kg_ids': [],
            'mondo_to_kg': {},
            'kg_to_row': {},
            'feature_names': [],
            'all_mondo_ids': [],
            'scalers': {},
        }
    
    # ================================================================
    # 
    # ================================================================
    
    def _print_feature_summary(self, result: dict):
        """ """
        print_info("═══ ═══")
        print_stat(" ", result['total_feature_dim'])
        print_stat(" （mondo_id）", len(result['all_mondo_ids']))
        print_stat(" KG ", len(result['disease_kg_ids']))
        
        print_info(" :")
        for gname, gdim in result['feature_dims'].items():
            print_stat(f"  {gname}", f"{gdim}  ")
