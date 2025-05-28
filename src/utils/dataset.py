"""
This module contains the necessary datasets for the project.
More specifically, it has the main DrugProtein dataset, which contains specific DrugMolecules.
Note that DrugMolecules now include optional 3D atomic coordinates as node features.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from typing import Any, List, Optional, Tuple # Added Tuple
import pandas as pd
from functools import lru_cache
import os
import re
import ast
import hashlib # Keep for _sanitize_prot_id if used for filename generation

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import Data # For type hinting loaded protein graph
from tqdm import tqdm

try:
    from .functional_groups import *
    # from .embed_proteins import ProteinGraphBuilder # embed_proteins is a script, not usually imported directly like this
                                                  # We might need helper functions from it if they exist and are importable
except ImportError:
    from src.utils.functional_groups import *
    # from src.utils.embed_proteins import ProteinGraphBuilder

ATOM_PAD_ID = 0          # 0 = "dummy / padded node"

# from sklearn.model_selection import train_test_split # Not used directly in this file after load_data moved to train.py
# from tdc.multi_pred import DTI # Not used directly in this file


# ----------------------------------------------------------------------------
# Collation helper
# ----------------------------------------------------------------------------
def pad_to(x: torch.Tensor, shape: tuple):
    pad = []
    for cur, tgt in zip(reversed(x.shape), reversed(shape)):
        pad += [0, tgt - cur]
    return F.pad(x, pad, mode='constant', value=0)


def collate_drug_prot(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Data, Any]], # Modified type hint for protein part
        hard_limit: int = 64, # Max nodes for padding
        drug_edge_feats: int = 17, # Number of drug edge features
        prot_edge_feats: int = 1   # Number of protein edge features (distance)
        ):

    drug_zs, drug_xs, drug_es, drug_as = [], [], [], []
    prot_zs, prot_xs, prot_es, prot_as = [], [], [], [] # Restored protein graph components
    labels = []

    H = hard_limit # Max nodes (atoms for drugs, residues for proteins)

    # for d_x, d_z, d_e, d_a, protein_embedding, label in batch: # Old custom embedding version
    for d_x, d_z, d_e, d_a, protein_graph_data, label in batch: # New: protein_graph_data is a torch_geometric.data.Data object
        # ───────────── drug ─────────────
        drug_zs.append(pad_to(d_z, (H,)))
        drug_xs.append(pad_to(d_x, (H, d_x.size(-1))))
        drug_es.append(pad_to(d_e, (H, H, drug_edge_feats)))
        drug_as.append(pad_to(d_a, (H, H)))

        # ─────────── protein ────────────
        # protein_graph_data is expected to be a torch_geometric.data.Data object
        # It should have: p_x (node features), p_edge_index, p_edge_attr
        
        p_x = protein_graph_data.x # Node features [N_prot_nodes, F_prot_node] (F_prot_node=24)
        p_edge_index = protein_graph_data.edge_index # Edge indices [2, num_edges]
        p_edge_attr = protein_graph_data.edge_attr # Edge attributes [num_edges, F_prot_edge] (F_prot_edge=1)

        N_prot_nodes = p_x.size(0)

        if N_prot_nodes > H: # truncate to hard_limit
            # Simple truncation: keep first H nodes and their induced subgraph
            node_mask = torch.arange(H)
            p_x = p_x[node_mask]
            
            # Filter edges: both source and target nodes must be within the first H nodes
            edge_mask = (p_edge_index[0] < H) & (p_edge_index[1] < H)
            p_edge_index = p_edge_index[:, edge_mask]
            p_edge_attr = p_edge_attr[edge_mask]
            N_prot_nodes = H
        
        # (1) residue-type integer IDs (derive from first 20 dims of one-hot node features)
        # Node features in .pt are: one_hot_AA(20) + charge(1) + coords(3) = 24 dims
        # The FeaturePrep in the model will use these IDs to create embeddings.
        res_ids = torch.argmax(p_x[:, :20], dim=1).long() + 1 # 1-20, 0=pad
        prot_zs.append(pad_to(res_ids, (H,)))

        # (2) protein node features (the part FeaturePrep concatenates: charge + coords)
        # The FeaturePrep module expects the raw node features (excluding IDs part if it embeds them)
        # For proteins, prot_prep = FeaturePrep(num_res_types, z_emb_dim)
        # It will embed res_ids. The remaining features from p_x (charge, coords) should be passed.
        # p_x here is [N, 24]. We pass p_x[:, 20:] which are charge (1) and coords (3) = 4 features.
        # So, prot_in_features for the model's GAT part will be z_emb_dim (from FeaturePrep) + 4.
        # This means prot_in_features argument to DualGraphAttentionNetwork's __init__ should be 4.
        prot_node_dense_feats = p_x[:, 20:] # [N_prot_nodes, 4] (charge, x, y, z)
        prot_xs.append(pad_to(prot_node_dense_feats, (H, prot_node_dense_feats.size(1))))


        # (3) dense edge-attributes & adjacency matrix for proteins
        adj_prot = torch.zeros((H, H), dtype=torch.float32)
        edge_attr_prot_dense = torch.zeros((H, H, prot_edge_feats), dtype=torch.float32)

        if p_edge_index.numel() > 0: # Check if there are any edges after potential truncation
            row, col = p_edge_index[0], p_edge_index[1]
            adj_prot[row, col] = 1
            adj_prot[col, row] = 1 # Assuming undirected, though original GAT handles directedness via attention
            # Populate edge attributes, ensure it's [H,H,F_prot_edge]
            # This assumes p_edge_attr corresponds to edges in p_edge_index
            edge_attr_prot_dense[row, col] = p_edge_attr 
            # If edges are undirected and attributes are symmetric, or if GAT sums/averages edge features for both directions:
            edge_attr_prot_dense[col, row] = p_edge_attr # Or handle as per model's expectation

        prot_as.append(adj_prot)
        prot_es.append(edge_attr_prot_dense)

        labels.append(label)

    return (
        torch.stack(drug_zs),    # [B, H]          (long) atomic numbers
        torch.stack(drug_xs),    # [B, H, F_d_node] drug node features (continuous part)
        torch.stack(drug_es),    # [B, H, H, F_d_edge] drug edge features
        torch.stack(drug_as),    # [B, H, H]       drug adjacency
        torch.stack(prot_zs),    # [B, H]          (long) protein residue type IDs
        torch.stack(prot_xs),    # [B, H, F_p_node_dense] protein node dense features (charge, coords)
        torch.stack(prot_es),    # [B, H, H, F_p_edge] protein edge features (distance)
        torch.stack(prot_as),    # [B, H, H]       protein adjacency (from graph)
        torch.tensor(labels, dtype=torch.float32)
    )


def _sanitize_prot_id(pid_input):
    """
    Sanitize protein ID input, handling various input types.
    (This function seems fine, keeping it as is for now if it's used for filename generation)
    """
    if pid_input is None: return "unknown"
    pid_str = str(pid_input)
    if "Name: Target_ID" in pid_str or "dtype: object" in pid_str:
        try:
            matches = re.findall(r'(\\d+\\s+[A-Z0-9]+(?:\\([^)]+\\))?)', pid_str)
            if matches:
                parts = matches[0].strip().split()
                if len(parts) >= 2: return parts[-1].strip()
            protein_pattern = r'\\b[A-Z0-9]{3,10}\\b'
            protein_matches = re.findall(protein_pattern, pid_str)
            if protein_matches:
                filtered = [p for p in protein_matches if p not in ['NAME', 'TARGET', 'TYPE', 'DTYPE', 'OBJECT']]
                if filtered: return filtered[0]
            if len(pid_str) > 50:
                hash_obj = hashlib.md5(pid_str.encode())
                return f"seq-{hash_obj.hexdigest()}"
        except Exception as e: print(f"Error parsing complex protein ID: {e}")
    if pid_str.startswith('[') and pid_str.endswith(']'):
        try:
            literal = ast.literal_eval(pid_str)
            if isinstance(literal, list) and literal: return str(literal[0])
        except:
            match = re.search(r"'([^']+)'", pid_str)
            if match: return match.group(1)
    if hasattr(pid_input, 'iloc') and hasattr(pid_input, 'values'):
        try: return str(pid_input.iloc[0])
        except: pass
    elif isinstance(pid_input, list) and len(pid_input) > 0:
        return str(pid_input[0])
    return pid_str.strip() # Added strip


class DrugMolecule:
    """
    Represents a drug molecule and its molecular graph structure, optionally with 3D coords.

    Node features: one-hot atom type, formal charge, degree, hybridization, aromatic, plus (x,y,z) coords if available.
    Edge features: one-hot bond type, conjugation, ring flags.
    """
    def __init__(self, smiles_str: str, max_nodes: int = 50, include_3d: bool = True):
        self.include_3d = include_3d

        # Parse base molecule
        mol0 = Chem.MolFromSmiles(smiles_str)
        if mol0 is None:
            raise ValueError(f"Invalid SMILES: {smiles_str}")

        # Prepare hydrogenated mol for optional 3D embedding
        mol_h = Chem.AddHs(mol0)

        # Identify heavy atom indices for coordinate fallback
        heavy_atom_indices = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() != 1]

        if include_3d:
            # Embed with ETKDG, fixed seed
            params = AllChem.ETKDGv3()
            params.randomSeed = 0
            res = AllChem.EmbedMolecule(mol_h)
            # 2) then optimize with MMFF94
            if res == 0 and mol_h.GetNumConformers() > 0:
                opt = AllChem.MMFFOptimizeMolecule(mol_h)   # <— run MMFF94
                conf = mol_h.GetConformer()
                heavy_coords = [[*conf.GetAtomPosition(idx)] for idx in heavy_atom_indices]
            else:
                heavy_coords = [[0.0, 0.0, 0.0] for _ in heavy_atom_indices]
            self.atom_coords = heavy_coords

        # Remove hydrogens to build heavy-only graph
        mol = Chem.RemoveHs(mol_h)

        # Extract graph structure
        self.mol, self.node_feats_list, self.edge_feats_dict, self.adjacency_list, self.neighbours = \
            self._construct_molecular_graph(mol)
        self.num_nodes = len(self.node_feats_list)
        self.max_nodes = max_nodes
        self.node_features_tensor, self.edge_features_tensor, self.adjacency_tensor, self.atomic_id_tensor = self._tensor_preprocess()

    def _construct_molecular_graph(self, mol: Chem.Mol):
        node_feats_list, adjacency_list, edge_feats_dict, neighbours = [], [], {}, []
        for atom_idx, atom in enumerate(mol.GetAtoms()): # Use enumerate for clarity if idx is needed elsewhere
            feats = {
                "atomic_num": atom.GetAtomicNum(), # Used for atomic_id_tensor (d_z)
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "hybridization": str(atom.GetHybridization()),
                "aromatic": int(atom.GetIsAromatic())
            }
            # If include_3d and self.atom_coords is populated correctly for heavy atoms:
            # if self.include_3d and hasattr(self, 'atom_coords') and atom_idx < len(self.atom_coords):
            #    feats["coords"] = self.atom_coords[atom_idx] # This would require _process_node_features to handle "coords"
            node_feats_list.append(feats)
            neighbours.append([])
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = {"bond_type": str(bond.GetBondType()),
                  "conjugated": int(bond.GetIsConjugated()),
                  "ring": int(bond.IsInRing())}
            edge_feats_dict[(i, j)], edge_feats_dict[(j, i)] = bf, bf
            adjacency_list += [(i, j), (j, i)]
            # neighbours[i].append(j) # This was not used later
            # neighbours[j].append(i)
        return mol, node_feats_list, edge_feats_dict, adjacency_list, neighbours # Removed neighbours from return as it's not used

    def _tensor_preprocess(self):
        proc_node_features = [] # This will be d_x
        proc_atomic_ids  = [] # This will be d_z
        
        for idx, feats in enumerate(self.node_feats_list):
            proc_node_features.append(self._process_node_features(feats))
            proc_atomic_ids.append(feats["atomic_num"])

        node_features_tensor = torch.tensor(proc_node_features, dtype=torch.float32) # d_x
        node_features_tensor = self._pad(node_features_tensor, (self.max_nodes, node_features_tensor.size(1) if self.num_nodes > 0 else 29)) # ensure consistent feature dim for empty graphs

        atomic_id_tensor = torch.tensor(proc_atomic_ids, dtype=torch.long) # d_z
        atomic_id_tensor = F.pad(atomic_id_tensor, (0, self.max_nodes - atomic_id_tensor.size(0)), value=ATOM_PAD_ID)

        if len(self.edge_feats_dict) > 0:
            num_edge_feats = len(self._process_edge_features(next(iter(self.edge_feats_dict.values()))))
        else: # Handle molecules with no bonds (e.g. single atoms)
            num_edge_feats = len(self._process_edge_features({"bond_type":"UNSPECIFIED","conjugated":0,"ring":0})) # Default edge feature count (17)
        
        edge_features_tensor = torch.zeros((self.max_nodes, self.max_nodes, num_edge_feats), dtype=torch.float32) # d_e
        adjacency_tensor = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32) # d_a
        for (i, j), bf in self.edge_feats_dict.items():
            if i < self.max_nodes and j < self.max_nodes:
                edge_features_tensor[i, j] = torch.tensor(self._process_edge_features(bf), dtype=torch.float32)
                adjacency_tensor[i, j] = 1 # Adjacency for GAT
            
        return node_features_tensor, edge_features_tensor, adjacency_tensor, atomic_id_tensor

    def _pad(self, t: torch.Tensor, shape: tuple):
        pad = []
        for cur, tgt in zip(reversed(t.shape), reversed(shape)):
            pad_val = tgt - cur
            if pad_val < 0: # Should not happen if max_nodes is respected
                pad_val = 0 
            pad += [0, pad_val]
        return F.pad(t, pad, mode='constant', value=0)

    def _process_node_features(self, features: dict[str, Any]) -> list[float]:
        hyb = {"UNSPECIFIED":0,"S":1,"SP":2,"SP2":3,"SP3":4,"SP2D":5,"SP3D":6,"SP3D2":7,"OTHER":8} # len 9
        atoms = [1,3,5,6,7,8,9,11,14,15,16,17,19,30,34,35,53] # len 17
        out: list[float] = []
        
        # Order: atomic_num_onehot (17), formal_charge (1), degree (1), hybridization_onehot (9), aromatic (1)
        # Total = 17 + 1 + 1 + 9 + 1 = 29 features. This matches drug_in_features=29.
        
        # Atomic num one-hot
        one_hot_atom = [0.0]*len(atoms)
        if features["atomic_num"] in atoms: one_hot_atom[atoms.index(features["atomic_num"])] = 1.0
        out += one_hot_atom
        
        out.append(float(features["formal_charge"]))
        out.append(float(features["degree"]))
        
        # Hybridization one-hot
        one_hot_hyb = [0.0]*len(hyb)
        one_hot_hyb[hyb.get(str(features["hybridization"]), 0)] = 1.0 # Ensure key is string
        out += one_hot_hyb
        
        out.append(float(features["aromatic"]))
        return out

    def _process_edge_features(self, features: dict[str, Any]) -> list[float]:
        bond_enc = {"UNSPECIFIED":0,"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4, # len 5
                    "IONIC":5,"HYDROGEN":6,"THREECENTER":7,"DATIVEONE":8,"DATIVE":9, # len 5
                    "DATIVEL":10,"DATIVER":11,"OTHER":12,"ZERO":13} # len 4. Total = 14?
                    # Original code implies 17 edge features. Let's check.
                    # Original: bond_type_onehot (14) + conjugated (1) + ring (1) = 16.
                    # One more feature? Let's assume 17 is correct and there's one more binary flag or similar.
                    # For now, sticking to original feature processing if it yielded 17.
                    # The provided snippet for process_edge_features only explicitly handles bond_type.

        # Re-implementing based on common GAT-CD4C features if the snippet was partial
        out: list[float] = []
        # Bond type one-hot (14 values)
        bond_type_one_hot = [0.0] * len(bond_enc)
        bond_type_one_hot[bond_enc.get(str(features.get("bond_type", "UNSPECIFIED")), 0)] = 1.0
        out += bond_type_one_hot

        # Conjugated flag (1 value)
        out.append(float(features.get("conjugated", 0)))

        # Ring flag (1 value)
        out.append(float(features.get("ring", 0)))
        
        # If total is 16, and model expects 17, we need one more placeholder or identify the missing feature.
        # For now, pad to 17 if needed, or assume these 16 are what's used if drug_edge_features=16.
        # The train.py sets drug_edge_features=17.
        if len(out) < 17: # Pad to 17 if current is 16
            out.append(0.0) # Placeholder for the 17th feature
        return out[:17] # Ensure always 17

    def to_tensors(self): # d_x (node features), d_z (atomic numbers), d_e (edge features), d_a (adjacency)
        return self.node_features_tensor, self.atomic_id_tensor, self.edge_features_tensor, self.adjacency_tensor


class DrugProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, graph_dir: str, # Changed args
                 max_nodes: int = 64, use_half: bool = False, include_3d_drug: bool = False): # include_3d for drugs
        
        if RDLogger is not None:
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)

        self.df = df
        self.graph_dir = graph_dir # Directory for .pt protein graph files
        self.max_nodes = max_nodes # Max nodes for padding (applies to both drug and protein)
        self.use_half = use_half # For half precision, not currently used in tensor creation
        self.include_3d_drug = include_3d_drug # For drug 3D coordinates

        # Ensure protein graph directory exists
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Precompute smiles to DrugMolecule
        self.smiles_to_drug = {
            smiles: self.load_drug(smiles)
            for smiles in tqdm(df['Drug'].unique(), desc="Processing drugs")
        }
        
        self._validate_protein_graph_paths()


    def _generate_protein_graph_filename(self, identifier: str) -> str:
        """
        Generates a safe filename for a protein graph. Consistent with embed_proteins.py.
        Hashes identifiers that are too long or contain disallowed characters.
        """
        MAX_IDENTIFIER_LEN_BEFORE_HASH = 100
        # Corrected regex: Allow A-Z, a-z, 0-9, underscore, hyphen, period.
        # Original flawed regex: r"^[A-Za-z0-9_\\-\\.]+$" which incorrectly required literal backslashes.
        ALLOWED_CHARS_REGEX = r"^[A-Za-z0-9_.-]+$" # Corrected regex
        is_too_long = len(identifier) > MAX_IDENTIFIER_LEN_BEFORE_HASH
        has_disallowed_chars = not re.match(ALLOWED_CHARS_REGEX, identifier)

        if is_too_long or has_disallowed_chars:
            hashed_identifier = hashlib.md5(identifier.encode()).hexdigest()
            return f"seq-{hashed_identifier}.pt"
        else:
            return f"{identifier}.pt"

    def _validate_protein_graph_paths(self):
        print("INFO: Validating protein graph paths...")
        missing_graphs_count = 0
        unique_prot_ids = self.df['Target_ID'].apply(_sanitize_prot_id).unique()
        # Get a list of raw unique protein IDs as well for fallback checking
        raw_unique_prot_ids = self.df['Target_ID'].unique()
        
        # Create a dictionary for quick lookup of raw_id -> sanitized_id
        # This assumes that the order and uniqueness of apply(_sanitize_prot_id).unique()
        # can be mapped back to original IDs if needed, but it's safer to iterate through
        # the original IDs from the dataframe that will be used in __getitem__
        
        checked_raw_ids = set() # To avoid redundant checks if raw IDs map to same sanitized ID

        print(f"INFO: Checking {len(raw_unique_prot_ids)} unique Target_IDs from the DataFrame.")

        for i, original_pid_from_df in enumerate(raw_unique_prot_ids):
            if original_pid_from_df in checked_raw_ids:
                continue
            checked_raw_ids.add(original_pid_from_df)

            sanitized_pid = _sanitize_prot_id(original_pid_from_df)
            
            # Attempt 1: Filename from sanitized_pid
            graph_filename_sanitized = self._generate_protein_graph_filename(sanitized_pid)
            graph_path_sanitized = os.path.join(self.graph_dir, graph_filename_sanitized)
            
            # Attempt 2: Filename from original_pid_from_df (if different from sanitized_pid)
            graph_filename_original = None
            graph_path_original = None
            if original_pid_from_df != sanitized_pid:
                graph_filename_original = self._generate_protein_graph_filename(original_pid_from_df)
                graph_path_original = os.path.join(self.graph_dir, graph_filename_original)

            log_prefix = f"  Processed ID {i+1}/{len(raw_unique_prot_ids)}: original='{original_pid_from_df}', sanitized='{sanitized_pid}' -> "
            
            found_path = None
            if os.path.exists(graph_path_sanitized):
                found_path = graph_path_sanitized
                if i < 5 or missing_graphs_count < 2 : # Log first few finds or if we start seeing misses
                     print(f"{log_prefix}FOUND (using sanitized): '{graph_filename_sanitized}'")
            elif graph_path_original and os.path.exists(graph_path_original):
                found_path = graph_path_original
                if i < 5 or missing_graphs_count < 2:
                     print(f"{log_prefix}FOUND (using original): '{graph_filename_original}'")
            else:
                missing_graphs_count += 1
                # Log detailed info for missing ones, showing both attempts
                log_msg = f"{log_prefix}MISSING. Attempted: "
                log_msg += f"1. Using sanitized ('{sanitized_pid}') -> '{graph_filename_sanitized}' (Path: {graph_path_sanitized})"
                if graph_filename_original:
                    log_msg += f", 2. Using original ('{original_pid_from_df}') -> '{graph_filename_original}' (Path: {graph_path_original})"
                else: # original was same as sanitized
                    log_msg += f" (Original ID was same as sanitized, so only one distinct filename generated: '{graph_filename_sanitized}')"
                
                # Log more frequently for missing files to catch problematic IDs
                if missing_graphs_count <= 10 or (i % (len(raw_unique_prot_ids)//20 +1) == 0) :
                    print(log_msg)


        if missing_graphs_count > 0:
            print(f"WARNING: {missing_graphs_count} unique protein graphs appear to be missing from '{self.graph_dir}' out of {len(raw_unique_prot_ids)} unique protein IDs from DataFrame.")
            print("Ensure that `embed_proteins.py` has been run for this dataset and the `protein_graph_dir` argument points to the correct location.")
            print("Review the 'MISSING' logs above to see the exact filenames that were attempted.")


    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=8192) # Cache DrugMolecule object processing
    def load_drug(self, smiles: str):
        # Returns tuple: (node_features_tensor, atomic_id_tensor, edge_features_tensor, adjacency_tensor)
        return DrugMolecule(smiles_str=smiles, max_nodes=self.max_nodes, include_3d=self.include_3d_drug).to_tensors()

    @lru_cache(maxsize=1024) # Cache loaded protein graph Data objects
    def load_protein(self, protein_id_original: str) -> Data:
        # This is the ID from the DataFrame row['Target_ID']
        
        # Attempt 1: Use sanitized ID for filename generation
        sanitized_pid = _sanitize_prot_id(protein_id_original)
        graph_filename_attempt1 = self._generate_protein_graph_filename(sanitized_pid)
        graph_path_attempt1 = os.path.join(self.graph_dir, graph_filename_attempt1)

        # Attempt 2: Use original ID for filename generation (if different and first attempt failed)
        graph_filename_attempt2 = None
        graph_path_attempt2 = None

        final_path_to_load = None

        if os.path.exists(graph_path_attempt1):
            final_path_to_load = graph_path_attempt1
        elif protein_id_original != sanitized_pid: # Only try original if it's different
            graph_filename_attempt2 = self._generate_protein_graph_filename(protein_id_original)
            graph_path_attempt2 = os.path.join(self.graph_dir, graph_filename_attempt2)
            if os.path.exists(graph_path_attempt2):
                final_path_to_load = graph_path_attempt2
        
        try:
            if final_path_to_load:
                protein_graph = torch.load(final_path_to_load, map_location='cpu', weights_only=False) 
                if not isinstance(protein_graph, Data):
                    # This warning is important
                    print(f"Warning: File {final_path_to_load} (for original ID '{protein_id_original}') did not contain a PyG Data object. Got {type(protein_graph)}. Using dummy.")
                    return self._create_dummy_protein_graph(protein_id_original=protein_id_original)
                return protein_graph
            else:
                # Neither attempt found the file. This case should be logged by _validate_protein_graph_paths too.
                # The dummy creation message here will be specific to the load attempt.
                # print(f"Debug: Protein graph file not found for '{protein_id_original}' (sanitized: '{sanitized_pid}'). Tried: '{graph_filename_attempt1}' and possibly '{graph_filename_attempt2}'. Creating dummy.")
                raise FileNotFoundError(f"Graph not found for '{protein_id_original}' (sanitized: '{sanitized_pid}')")

        except FileNotFoundError:
            return self._create_dummy_protein_graph(protein_id_original=protein_id_original)
        except Exception as e:
            # Log which path was being attempted if possible, or just the general error
            path_info = f"at path {final_path_to_load}" if final_path_to_load else f"(path not resolved for ID {protein_id_original})"
            print(f"Error loading protein graph {path_info} for ID {protein_id_original}: {e}. Using dummy protein.")
            return self._create_dummy_protein_graph(protein_id_original=protein_id_original)

    def _create_dummy_protein_graph(self, protein_id_original: Optional[str] = None) -> Data:
        pid_info = f" for PID: {protein_id_original}" if protein_id_original else ""
        print(f"Warning: Creating dummy protein graph{pid_info}.")
        # Node features: 24 dims (one-hot AA(20) + charge(1) + coords(3))
        # Edge features: 1 dim (distance)
        # Create a single dummy node
        dummy_x = torch.zeros((1, 24), dtype=torch.float32)
        # No edges for a single node graph, or a self-loop if required by model downstream (GAT usually handles this)
        dummy_edge_index = torch.empty((2, 0), dtype=torch.long) # No edges
        dummy_edge_attr = torch.empty((0, 1), dtype=torch.float32) # No edge attributes

        return Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Data, Any]:
        row = self.df.iloc[i]
        drug_smiles = row['Drug']
        protein_id = row['Target_ID'] # Original ID from DataFrame
        label = row['Label']
        
        # d_x_drug, d_z_drug, d_e_drug, d_a_drug
        drug_tensors = self.load_drug(drug_smiles) 
        
        protein_graph_data = self.load_protein(protein_id)

        return (*drug_tensors, protein_graph_data, label)


# Example of how to use if this file were run (for testing)
if __name__ == '__main__':
    # Create dummy data for testing
    dummy_df = pd.DataFrame({
        'Drug': ['CCO', 'CNC'],
        'Target_ID': ['P12345', 'P67890'], # Example protein IDs
        'Label': [7.5, 8.1]
    })
    dummy_graph_dir = "../data/dummy_protein_graphs"
    os.makedirs(dummy_graph_dir, exist_ok=True)

    # Create dummy .pt files for P12345 and P67890
    # Dummy graph for P12345
    prot1_x = torch.rand(10, 24) # 10 residues, 24 features
    prot1_edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                                     [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 0, 9]], dtype=torch.long) # Example edges
    prot1_edge_attr = torch.rand(prot1_edge_index.size(1), 1) # 1 edge feature (distance)
    prot1_data = Data(x=prot1_x, edge_index=prot1_edge_index, edge_attr=prot1_edge_attr)
    torch.save(prot1_data, os.path.join(dummy_graph_dir, "P12345.pt"))
    
    # Dummy graph for P67890 (missing, to test dummy creation)
    # Not creating P67890.pt to test fallback

    print(f"Created dummy data in {dummy_graph_dir}")

    dataset = DrugProteinDataset(df=dummy_df, graph_dir=dummy_graph_dir, max_nodes=15, include_3d_drug=False)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        item0 = dataset[0] # d_x, d_z, d_e, d_a, protein_graph_data, label
        item1 = dataset[1] # Should load dummy for P67890

        print(f"Item 0 drug node features shape: {item0[0].shape}")      # d_x
        print(f"Item 0 drug atomic IDs shape: {item0[1].shape}")         # d_z
        print(f"Item 0 protein graph data (P12345): {item0[4]}")
        print(f"Item 0 label: {item0[5]}")

        print(f"Item 1 protein graph data (P67890 - should be dummy): {item1[4]}")


    # Test collation
    from functools import partial
    
    # Create a list of items for the batch
    # Ensure items are structured as expected by collate_drug_prot
    # d_x, d_z, d_e, d_a, protein_graph_data, label
    batch_data = [dataset[i] for i in range(len(dataset))]
    if batch_data:
        collate_fn_test = partial(collate_drug_prot, hard_limit=15, drug_edge_feats=17, prot_edge_feats=1)
        
        try:
            collated_batch = collate_fn_test(batch_data)
            d_z, d_x, d_e, d_a, p_z, p_x, p_e, p_a, lbls = collated_batch
            print("\nCollated batch shapes:")
            print(f"  Drug Atomic IDs (d_z): {d_z.shape}, type: {d_z.dtype}")
            print(f"  Drug Node Features (d_x): {d_x.shape}")
            print(f"  Drug Edge Features (d_e): {d_e.shape}")
            print(f"  Drug Adjacency (d_a): {d_a.shape}")
            print(f"  Protein Residue IDs (p_z): {p_z.shape}, type: {p_z.dtype}")
            print(f"  Protein Node Dense Feats (p_x): {p_x.shape}") # charge + coords
            print(f"  Protein Edge Features (p_e): {p_e.shape}")
            print(f"  Protein Adjacency (p_a): {p_a.shape}")
            print(f"  Labels (lbls): {lbls.shape}")
        except Exception as e:
            print(f"Error during collate_fn_test: {e}")
            import traceback
            traceback.print_exc()

    # Clean up dummy files and dir
    # os.remove(os.path.join(dummy_graph_dir, "P12345.pt"))
    # os.rmdir(dummy_graph_dir)
    # print("Cleaned up dummy files.")