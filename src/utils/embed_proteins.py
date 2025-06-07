"""Protein graph builder & bulk graph generator for CD4C, KIBA and DAVIS
===========================================================================

Usage
-----
1. **Single build (library use)**

   
python
   from embed_proteins import ProteinGraphBuilder
   builder = ProteinGraphBuilder(graph_dir="../data/protein_graphs")
   data = builder.build("AF‑P00533‑F1‑model_v4.pdb")
   builder.save("CHEMBL612545", data)


2. **Dataset‑wide pre‑compute**

   
bash
   python src/utils/embed_proteins.py --dataset KIBA --out-dir data/protein_graphs \\
                            --num-workers 8 [--use-local-colabfold]


   The script pulls the desired **TDC** dataset, resolves each *Target_ID* (which
   is already a UniProt accession for KIBA/DAVIS) or raw sequence, obtains a PDB
   via the **ColabFold** local pipeline (if --use-local-colabfold is set) or
   remote AlphaFold DB, builds the residue graph (one‑hot + charge + Cα coords) and
   saves it as <Target_ID>.pt.

Node features
-------------
* **one‑hot(20)** standard amino‑acid alphabet
* **charge (1)** integer {‑1,0,+1} at pH≈7
* **coords (3)** Cα x,y,z in Å
Total: **24** dims.

Edges
-----
* Peptide‑bond neighbours
* Any residue pair with Cα‑Cα distance < *cut‑off* (default **10 Å**).
Edge feature = single scalar distance (Å).
"""
from __future__ import annotations

import argparse
import os
import sys
import hashlib
import pathlib
import re
import subprocess
import time
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

import requests
import torch
import pandas as pd
from Bio.PDB import PDBParser
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Amino‑acid constants
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1}
NUM_AA = len(AA_ORDER)                                                  # 20

_AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
_UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# Regex for UniProt accession
UNIPROT_REGEX = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}|[A-N][0-9][A-Z0-9]{3}[0-9]{4}|[A-Z]{3}[0-9]{7})$"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def looks_like_uniprot(acc: str) -> bool:
    """Check if a string matches the UniProt ID format."""
    return bool(UNIPROT_REGEX.match(acc))

def uniprot_exists(acc: str) -> bool:
    """Check if a UniProt ID exists in the remote database."""
    url = _UNIPROT_FASTA.format(acc=acc)
    try:
        r = requests.head(url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False

# ---------------------------------------------------------------------------
class ProteinGraphBuilder:
    """Build protein graphs from PDB structures.

    Args:
        graph_dir: Directory to save/load protein graphs
        cutoff: Distance cutoff for spatial edges in Angstroms
        pocket_radius: If given, only keep residues within this distance of ligand
        use_colabfold: Whether to use local ColabFold for structure prediction
    """

    def __init__(
        self,
        graph_dir: str = "../data/protein_graphs",
        cutoff: float = 10.0,
        pocket_radius: float | None = None,
        use_colabfold: bool = False
    ):
        self.cutoff = cutoff              # edge build distance threshold (Å)
        self.pocket_radius = pocket_radius
        self.graph_dir = pathlib.Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDBParser(QUIET=True)
        self.use_colabfold = use_colabfold
        
        # Check if colabfold_batch is available in PATH
        if self.use_colabfold:
            try:
                # First test if colabfold_batch is in path
                subprocess.run(["which", "colabfold_batch"], 
                              check=True, 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
                
                # Also check if it actually runs without errors
                test_cmd = ["colabfold_batch", "--help"]
                test_proc = subprocess.run(test_cmd, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                
                if test_proc.returncode != 0 or "linear_util" in test_proc.stderr or "AttributeError" in test_proc.stderr:
                    print("WARNING: ColabFold has dependency issues. Found in PATH but failed basic test.")
                    print(f"Error: {test_proc.stderr[:200]}...")
                    print("Falling back to linear structure generation.")
                    self._colabfold_available = False
                else:
                    self._colabfold_available = True
            except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
                print("WARNING: colabfold_batch command not found in PATH.")
                print("Falling back to linear structure generation.")
                self._colabfold_available = False
        else:
            self._colabfold_available = False

    def _generate_protein_graph_filename(self, identifier: str) -> str:
        """
        Generates a safe filename for a protein graph.
        Hashes identifiers that are too long or contain disallowed characters.
        """
        # Maximum length for the identifier part of the filename before hashing.
        # Common filesystem limit is 255 bytes. This is conservative.
        MAX_IDENTIFIER_LEN_BEFORE_HASH = 100

        # Characters allowed in a direct filename (conservative set).
        # UniProt IDs, ChEMBL IDs, and typical sequences (A-Z) should pass this.
        ALLOWED_CHARS_REGEX = r"^[A-Za-z0-9_\-\.]+$"

        is_too_long = len(identifier) > MAX_IDENTIFIER_LEN_BEFORE_HASH
        # A typical protein sequence (all caps letters) IS matched by the regex.
        # So, has_disallowed_chars will be False for them.
        # Hashing for sequences will primarily be triggered by length.
        has_disallowed_chars = not re.match(ALLOWED_CHARS_REGEX, identifier)

        if is_too_long or has_disallowed_chars:
            # Use MD5 hash for consistent naming
            hashed_identifier = hashlib.md5(identifier.encode()).hexdigest()
            return f"seq-{hashed_identifier}.pt"
        else:
            return f"{identifier}.pt"

    def _fetch_pdb(self, acc: str) -> pathlib.Path:
        """
        Fetch PDB structure for a UniProt ID or create one from a sequence.
        
        Args:
            acc: UniProt ID or protein sequence
            
        Returns:
            Path to PDB file
        """
        # Try as UniProt ID first
        is_uniprot = looks_like_uniprot(acc) and uniprot_exists(acc)
        
        if is_uniprot:
            # For UniProt IDs, try AlphaFold DB first
            pdb_url = _AF_URL.format(acc=acc)
            pdb_path = self.graph_dir / f"AF-{acc}-F1-model_v4.pdb"
            
            if pdb_path.exists():
                return pdb_path
                
            try:
                print(f"Fetching {acc} from AlphaFold DB...")
                r = requests.get(pdb_url, timeout=30)
                r.raise_for_status()
                pdb_path.write_bytes(r.content)
                return pdb_path
            except requests.RequestException as e:
                if not self.use_colabfold:
                    print(f"Failed to download AlphaFold PDB for {acc}: {e}")
                    print("Falling back to linear structure generation.")
                    # Get sequence for UniProt ID
                    try:
                        r_fasta = requests.get(_UNIPROT_FASTA.format(acc=acc), timeout=15)
                        if r_fasta.status_code == 200:
                            fasta_content = r_fasta.text
                            fetched_seq = "".join(fasta_content.splitlines()[1:]).strip()
                            if fetched_seq:
                                return self._create_linear_structure(acc, fetched_seq)
                    except Exception as ex:
                        print(f"Failed to fetch sequence for {acc}: {ex}")
                    # If all else fails, use the ID as a placeholder sequence
                    return self._create_linear_structure(acc, "A" * 50)
                print(f"AlphaFold DB fetch failed for {acc}, trying ColabFold...")

        # If not UniProt or failed to fetch, try ColabFold
        if self.use_colabfold and self._colabfold_available:
            # For UniProt IDs, try to get sequence first
            seq = acc  # Default: assume acc is the sequence
            
            if is_uniprot:
                try:
                    r_fasta = requests.get(_UNIPROT_FASTA.format(acc=acc), timeout=15)
                    if r_fasta.status_code == 200:
                        fasta_content = r_fasta.text
                        fetched_seq = "".join(fasta_content.splitlines()[1:]).strip()
                        if fetched_seq:
                            seq = fetched_seq
                except Exception:
                    pass
            
            # Validate sequence
            if not seq or not re.match(r"^[A-Za-z]+$", seq.upper()):
                raise ValueError(f"Invalid protein sequence for {acc}")
                
            # Create a safe filename for ColabFold
            if len(acc) > 30:
                safe_name = f"seq-{hashlib.md5(acc.encode()).hexdigest()}"
            else:
                safe_name = re.sub(r'[^\w\-.]', '_', acc)
                
            # Create FASTA file for ColabFold
            fasta_path = self.graph_dir / f"{safe_name}.fasta"
            with open(fasta_path, 'w') as f:
                # Use a very short header to prevent long filenames in output
                f.write(f">seq\n{seq}\n")
                
            # Create separate subdirectory for ColabFold output to prevent long filenames
            colabfold_output_dir = self.graph_dir / f"cf_output_{hashlib.md5(acc.encode()).hexdigest()[:8]}"
            colabfold_output_dir.mkdir(exist_ok=True)
                
            # Run ColabFold
            cmd = [
                "colabfold_batch",
                "--num-recycle", "3",
                "--model-type", "auto",
                "--msa-mode", "single_sequence",  # Skip the MSA generation step
            ]

            # Add input and output paths
            cmd.extend([
                str(fasta_path),
                str(colabfold_output_dir)
            ])
            
            print(f"Running ColabFold command: {' '.join(cmd)}")
            
            try:
                print(f"Starting ColabFold for {safe_name}...")
                
                # Set specific environment variables to help with JAX compatibility
                env = os.environ.copy()
                env["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
                env["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
                
                # Check if GPU is available, add cpu-only if not
                try:
                    import torch
                    if not torch.cuda.is_available():
                        cmd.append("--cpu-only")
                        print("No GPU detected, using CPU-only mode")
                    else:
                        print(f"GPU detected: {torch.cuda.get_device_name(0)}, using GPU acceleration")
                        # Add environment variables for better GPU performance
                        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
                        env["TF_FORCE_UNIFIED_MEMORY"] = "1"
                        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Use 80% of GPU memory
                        env["TF_GPU_THREAD_MODE"] = "gpu_private"
                except ImportError:
                    cmd.append("--cpu-only")
                    print("Torch not found, defaulting to CPU-only mode")
                
                # Try to run ColabFold
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env
                )
                
                # Set a timeout (10 minutes)
                timeout = 30  # Start with a short timeout for quick error detection
                try:
                    # First check if it fails immediately due to dependency issues
                    stdout, stderr = process.communicate(timeout=timeout)
                    if process.returncode != 0:
                        print(f"ColabFold failed with return code {process.returncode}")
                        print(f"Error: {stderr[:500]}")
                        return self._create_linear_structure(acc, seq)
                except subprocess.TimeoutExpired:
                    # If it didn't fail immediately with dependency error, try longer timeout
                    try:
                        print("Still running, waiting longer...")
                        stdout, stderr = process.communicate(timeout=600)  # 10 minute full timeout
                        if process.returncode != 0:
                            print(f"ColabFold failed after longer run with return code {process.returncode}")
                            print(f"Error: {stderr[:200]}")
                            return self._create_linear_structure(acc, seq)
                    except subprocess.TimeoutExpired:
                        print("ColabFold process timed out after 10 minutes")
                        process.kill()
                        return self._create_linear_structure(acc, seq)
                    
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"Failed to run ColabFold for {acc}: {e}")
                return self._create_linear_structure(acc, seq)
                
            # Find PDB file (search for rank 1 model in the output directory)
            pdb_files = list(colabfold_output_dir.glob(f"*.pdb"))
            
            if not pdb_files:
                print(f"WARNING: ColabFold did not produce PDB for {acc}. Creating a simple linear structure.")
                return self._create_linear_structure(acc, seq)
                
            # Prefer relaxed model
            for pdb in pdb_files:
                if "relaxed" in pdb.name:
                    return pdb
            # Otherwise take first
            return pdb_files[0]
            
        else:
            # If colabfold not available or not requested, create a linear structure
            if is_uniprot:
                # Try to get sequence for UniProt ID
                try:
                    r_fasta = requests.get(_UNIPROT_FASTA.format(acc=acc), timeout=15)
                    if r_fasta.status_code == 200:
                        fasta_content = r_fasta.text
                        fetched_seq = "".join(fasta_content.splitlines()[1:]).strip()
                        if fetched_seq:
                            return self._create_linear_structure(acc, fetched_seq)
                except Exception as ex:
                    print(f"Failed to fetch sequence for {acc}: {ex}")
            
            # For non-UniProt or if fetching failed, assume acc is the sequence
            if not re.match(r"^[A-Za-z]+$", acc):
                print(f"Warning: {acc} doesn't look like a valid sequence. Using placeholder.")
                seq = "A" * 50
            else:
                seq = acc
                
            return self._create_linear_structure(acc, seq)

    def _create_linear_structure(self, acc: str, seq: str) -> pathlib.Path:
        """Create a simple linear structure for a protein sequence as fallback.
        
        This generates a PDB file with a straight-line backbone (Cα atoms only)
        when ColabFold fails to produce a proper structure.
        
        Args:
            acc: UniProt ID or sequence identifier
            seq: Protein sequence
            
        Returns:
            Path to generated PDB file
        """
        # Create a safe filename
        if len(acc) > 30:
            safe_name = f"seq-{hashlib.md5(acc.encode()).hexdigest()}"
        else:
            safe_name = re.sub(r'[^\w\-.]', '_', acc)
            
        pdb_path = self.graph_dir / f"{safe_name}_linear.pdb"
        
        # Generate a linear structure with 3.8Å between consecutive Cα atoms
        with open(pdb_path, 'w') as f:
            f.write("HEADER    LINEAR STRUCTURE (FALLBACK)              \n")
            f.write(f"TITLE     LINEAR STRUCTURE FOR {acc}              \n")
            
            for i, aa in enumerate(seq):
                x = i * 3.8  # 3.8Å is typical Cα-Cα distance
                y = 0.0
                z = 0.0
                atom_serial = i + 1
                residue_seq = i + 1
                
                # Use standard 3-letter amino acid code
                aa_code = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", 
                           "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", 
                           "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", 
                           "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}.get(aa, "GLY")
                
                # PDB ATOM record format
                f.write(f"ATOM  {atom_serial:5d}  CA  {aa_code} A{residue_seq:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
            
            f.write("END")
        
        print(f"Created linear structure at {pdb_path}")
        return pdb_path

    def _extract_ca(self, pdb_path: pathlib.Path) -> Tuple[torch.Tensor, List[str]]:
        """Extract C-alpha coordinates and residue names from PDB."""
        coords, aa3 = [], []
        structure = self.parser.get_structure("af", str(pdb_path))
        for model in structure:
            for chain in model:
                for res in chain:
                    if "CA" not in res:
                        continue
                    coords.append(torch.tensor(res["CA"].coord, dtype=torch.float32))
                    aa3.append(res.get_resname().upper())
        if not coords:
            raise RuntimeError(f"No C-alpha atoms found in {pdb_path}")
        return torch.stack(coords), aa3

    def _aa_one_hot(self, aa_list: List[str]) -> torch.Tensor:
        """Convert list of 3-letter amino acid codes to one-hot encoding."""
        one_hot = torch.zeros((len(aa_list), NUM_AA), dtype=torch.float32)
        for i, aa3 in enumerate(aa_list):
            aa1 = self._three_to_one(aa3)
            idx = AA_TO_IDX.get(aa1, None)
            if idx is not None:
                one_hot[i, idx] = 1.0
        return one_hot

    def _three_to_one(self, resname: str) -> str:
        """Convert 3‑letter residue code to 1‑letter, fallback to 'X'."""
        mapping = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        return mapping.get(resname, "X")

    def _charges(self, aa_list: List[str]) -> torch.Tensor:
        """Create charge tensor for amino acids."""
        charges = torch.zeros((len(aa_list), 1), dtype=torch.float32)
        for i, aa3 in enumerate(aa_list):
            aa1 = self._three_to_one(aa3)
            charges[i, 0] = AA_CHARGE.get(aa1, 0)
        return charges

    def _crop_to_pocket(self, coords: torch.Tensor, 
                       ligand_coords: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the subset of coords within pocket_radius of *any* ligand_coords."""
        if ligand_coords is None or self.pocket_radius is None:
            m = torch.ones(coords.size(0), dtype=torch.bool)
            return coords, m
        dist = torch.cdist(coords, ligand_coords)          # [N_res, N_lig]
        m = (dist.min(dim=1).values < self.pocket_radius)   # [N_res]
        return coords[m], m
    
    def build(
        self,
        pdb_path: str | pathlib.Path,
        seq: Optional[str] = None,
        esm: Optional[torch.Tensor] = None,
        ligand_coords: Optional[torch.Tensor] = None,
    ) -> Data:
        """Build a protein graph from a PDB file.

        Args:
            pdb_path: Path to PDB file
            seq: Optional sequence to validate against
            esm: Optional ESM embeddings [L, 1280]
            ligand_coords: Optional ligand coordinates for pocket cropping

        Returns:
            torch_geometric.data.Data object with:
                - x: Node features [N, F] (one-hot AA + charge + coords)
                - edge_index: Edge indices [2, E]
                - edge_attr: Edge features [E, 1] (distances)
        """
        coords, aa_list = self._extract_ca(pathlib.Path(pdb_path))
        
        # Sanitize coordinates
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute pairwise distances
        dist = torch.cdist(coords, coords)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=1e4)  # 1e4 Å ≫ cutoff

        # Build edges
        within = (dist > 0) & (dist < self.cutoff)
        edge_index = within.nonzero(as_tuple=False).t()
        edge_attr = dist[within].unsqueeze(-1)

        # Build node features
        node_x = torch.cat([
            self._aa_one_hot(aa_list),
            self._charges(aa_list).unsqueeze(-1),
            coords
        ], dim=-1)

        # Optional: crop to pocket
        if self.pocket_radius is not None and ligand_coords is not None:
            node_mask, edge_mask = self._crop_to_pocket(coords, ligand_coords)
            node_x = node_x[node_mask]
            edge_index = edge_index[:, edge_mask]
            edge_attr = edge_attr[edge_mask]

        # Optional: add ESM embeddings
        if esm is not None:
            if len(esm) != len(aa_list):
                raise ValueError(f"ESM embeddings length {len(esm)} does not match sequence length {len(aa_list)}")
            node_x = torch.cat([node_x, esm], dim=-1)

        return Data(
            x=node_x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

    def load(self, identifier: str) -> Data:
        """Load protein graph from file."""
        filename_str = self._generate_protein_graph_filename(identifier)
        path = self.graph_dir / filename_str
        if not path.is_file():
            raise FileNotFoundError(f"Protein graph not found: {path} (derived from identifier: {identifier})")
        
        # Try loading with various approaches
        try:
            # First try with default settings
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # weights_only parameter not available in this PyTorch version
                return torch.load(path, map_location="cpu")
            
        except (RuntimeError, AttributeError) as e:
            error_str = str(e)
            if "WeightsUnpickler error" in error_str or "Unsupported global" in error_str:
                # Try registering torch_geometric classes and load again
                if "DataEdgeAttr" in error_str or "DataTensorAttr" in error_str:
                    print(f"Encountered serialization error: {error_str}")
                    print("Attempting to register torch_geometric classes...")
                    register_torch_geometric_classes()
                    try:
                        try:
                            return torch.load(path, map_location="cpu", weights_only=False)
                        except TypeError:
                            # weights_only parameter not available in this PyTorch version
                            return torch.load(path, map_location="cpu")
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                
                # Try without weights_only parameter (older PyTorch versions)
                try:
                    return torch.load(path, map_location="cpu")
                except Exception as e3:
                    print(f"All loading attempts failed for {path}")
                    raise RuntimeError(f"Could not load protein graph: {e3}") from e
            else:
                # Some other RuntimeError
                raise

    def save(self, identifier: str, data: Data):
        """Save protein graph to file."""
        filename_str = self._generate_protein_graph_filename(identifier)
        
        # Ensure torch_geometric classes are registered before saving
        register_torch_geometric_classes()
        
        torch.save(data, self.graph_dir / filename_str)

# ---------------------------------------------------------------------------
# Single-threaded bulk processing
# ---------------------------------------------------------------------------

def register_torch_geometric_classes():
    """
    Register torch_geometric classes with PyTorch's serialization safe globals.
    This is required for PyTorch 2.6+ to allow loading and saving torch_geometric objects.
    """
    try:
        import torch.serialization
        import sys
        
        # List of modules that might contain the required classes
        modules_to_check = [
            'torch_geometric.data.data',
            'torch_geometric.data',
            'torch_geometric.data.storage',
        ]
        
        # Classes we need to register
        class_names = [
            'Data', 'HeteroData', 'DataEdgeAttr', 'DataTensorAttr',
            'BaseStorage', 'NodeStorage', 'EdgeStorage', 'GlobalStorage',
            'TensorAttr', 'EdgeAttr',
        ]
        
        # Find and collect all available classes
        classes_to_register = []
        
        for module_name in modules_to_check:
            try:
                module = sys.modules.get(module_name) or __import__(module_name)
                for class_name in class_names:
                    try:
                        cls = getattr(module, class_name, None)
                        if cls is not None and cls not in classes_to_register:
                            classes_to_register.append(cls)
                            print(f"Found {class_name} in {module_name}")
                    except (ImportError, AttributeError):
                        pass
            except ImportError:
                pass
        
        # Register classes with PyTorch's serialization
        if classes_to_register:
            # Try direct method first (newer PyTorch)
            try:
                torch.serialization.add_safe_globals(classes_to_register)
                print(f"Added {len(classes_to_register)} torch_geometric classes to PyTorch safe globals list")
                return True
            except (AttributeError, TypeError):
                # For older PyTorch that might have different API
                if hasattr(torch.serialization, '_get_safe_globals'):
                    safe_globals = torch.serialization._get_safe_globals()
                    for cls in classes_to_register:
                        key = f"{cls.__module__}.{cls.__name__}"
                        safe_globals[key] = cls
                    print(f"Added {len(classes_to_register)} torch_geometric classes to PyTorch safe globals (legacy method)")
                    return True
        
        return False
    
    except (ImportError, Exception) as e:
        print(f"Warning: Could not register torch_geometric classes: {e}")
        return False

def process_targets(targets: List[str], 
                   out_dir: str, 
                   cutoff: float = 10.0,
                   use_colabfold: bool = False):
    """
    Process a list of protein targets and save graphs.
    
    Args:
        targets: List of protein targets (UniProt IDs or sequences)
        out_dir: Output directory for protein graphs
        cutoff: Distance cutoff for edges
        use_colabfold: Whether to use local ColabFold
    """
    # Register torch_geometric classes for serialization
    register_torch_geometric_classes()

    builder = ProteinGraphBuilder(
        graph_dir=out_dir,
        cutoff=cutoff,
        use_colabfold=use_colabfold
    )
    
    if use_colabfold and not builder._colabfold_available:
        print("WARNING: ColabFold requested but not available or has dependency issues.")
        print("Will use linear structure generation as fallback.")
        print("To fix ColabFold, consider creating a fresh environment with compatible versions:")
        print("  conda create -n colabfold-fixed python=3.9")
        print("  conda activate colabfold-fixed") 
        print("  pip install 'jax==0.3.25' 'jaxlib==0.3.25'")
        print("  pip install 'colabfold[alphafold-cpu]==1.5.2'")
    
    success = 0
    for target in tqdm(targets, desc="Processing proteins"):
        try:
            # Skip if already exists
            try:
                builder.load(target)
                print(f"[✓] {target} (already exists)")
                success += 1
                continue
            except FileNotFoundError:
                pass
                
            # Process target with error handling
            try:
                pdb_path = builder._fetch_pdb(target)
                protein_graph = builder.build(pdb_path)
                builder.save(target, protein_graph)
                print(f"[✓] {target}")
                success += 1
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Exiting...")
                return success
            except Exception as e:
                print(f"[✗] Error processing {target}: {type(e).__name__} - {str(e)}")
                # If any individual target fails, continue with the next one
                continue
                
        except Exception as e:
            print(f"[✗] Error processing {target}: {type(e).__name__} - {str(e)}")
    
    print(f"\nProcessed {len(targets)} targets: {success} successful, {len(targets) - success} failed")
    return success

def load_davis_mapping():
    """Load mapping between protein IDs and sequences for DAVIS dataset."""
    try:
        from tdc.multi_pred import DTI
        data = DTI(name="DAVIS")
        df = data.get_data()
        
        processed_targets = []
        raw_targets = df["Target"].unique()

        for t_original in raw_targets:
            if isinstance(t_original, list) and len(t_original) > 0:
                processed_targets.append(str(t_original[0])) # Use the first element
            else:
                processed_targets.append(str(t_original))
        
        # Check for uniqueness of the processed simple names
        if len(processed_targets) != len(set(processed_targets)):
            print("WARNING: Non-unique simple protein names found in DAVIS after processing. This could lead to graph overwriting.")
            # Potentially add more detailed logging of duplicates here if needed
            # For now, we proceed with the (potentially non-unique) simple names

        # The original logic for uniprot_like can be kept if desired, but apply it to processed_targets
        uniprot_like = [pt for pt in set(processed_targets) if looks_like_uniprot(pt)]
        
        if uniprot_like:
            print(f"Found {len(uniprot_like)} UniProt-like simple IDs in DAVIS dataset from processed targets.")
            # Decide if you want to return only uniprot_like or all processed_targets
            # Returning all processed_targets for consistency with how train_new.py will expect them.
            return sorted(list(set(processed_targets))) 
        else:
            print("No UniProt IDs found in DAVIS dataset processed targets, using all unique processed targets.")
            return sorted(list(set(processed_targets)))
            
    except Exception as e:
        print(f"Error loading DAVIS dataset: {e}")
        return []

def main():
    """Command-line interface for protein graph generation."""
    parser = argparse.ArgumentParser(description="Protein graph builder")
    parser.add_argument("--dataset", choices=["KIBA", "DAVIS", "CD4C"], required=True,
                        help="Dataset to process")
    parser.add_argument("--out-dir", default="../data/protein_graphs",
                        help="Output directory for protein graphs")
    parser.add_argument("--cutoff", type=float, default=10.0,
                        help="Distance cutoff for edges (Å)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers (currently ignored, using single thread)")
    parser.add_argument("--use-local-colabfold", action="store_true",
                        help="Try to use local ColabFold instead of AlphaFold DB")
    
    args = parser.parse_args()
    
    # Check if colabfold is available if requested
    if args.use_local_colabfold:
        try:
            subprocess.run(["which", "colabfold_batch"], 
                          check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            print("ColabFold found in PATH.")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("\n" + "="*80)
            print("WARNING: colabfold_batch command not found in PATH.")
            print("To install ColabFold in a dedicated environment, run:")
            print("  conda create -n colabfold python=3.9")
            print("  conda activate colabfold")
            print("  pip install 'colabfold[alphafold-cpu]'")
            print("="*80 + "\n")
            print("Continuing with linear structure generation as fallback...")
    
    # Get targets based on dataset
    targets = []
    if args.dataset == "DAVIS":
        targets = load_davis_mapping()
    elif args.dataset == "KIBA":
        from tdc.multi_pred import DTI
        data = DTI(name="KIBA")
        df = data.get_data()
        raw_targets = list(df["Target"].unique())
        processed_targets_kiba = []
        for t_original in raw_targets:
            if isinstance(t_original, list) and len(t_original) > 0:
                processed_targets_kiba.append(str(t_original[0]))
            else:
                processed_targets_kiba.append(str(t_original))
        
        targets = sorted(list(set(processed_targets_kiba)))
        if len(processed_targets_kiba) != len(targets):
             print("WARNING: Non-unique simple protein names found in KIBA after processing. Using unique set.")
        print(f"Found {len(targets)} unique simple targets in KIBA dataset")
    elif args.dataset == "CD4C":
        data_path = pathlib.Path("../data/filtered_cancer_all.csv")
        if not data_path.exists():
            sys.exit(f"CD4C dataset file not found: {data_path}")
        df = pd.read_csv(data_path)
        targets = list(df["Target_ID"].unique())
        print(f"Found {len(targets)} targets in CD4C dataset")
    else:
        sys.exit(f"Unknown dataset: {args.dataset}")
    
    if not targets:
        sys.exit("No targets found to process")
    
    print(f"Processing {len(targets)} targets from {args.dataset} dataset")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process targets
    process_targets(
        targets=targets,
        out_dir=args.out_dir,
        cutoff=args.cutoff,
        use_colabfold=args.use_local_colabfold
    )

if __name__ == "__main__":
    main()