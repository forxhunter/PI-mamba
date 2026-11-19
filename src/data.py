"""
Data utilities for PI-Mamba: PDB parsing, dataset, length-bucketed batching and
PDB writing.

The model works on single-chain CA traces. Batches contain chains of one
length only, because the Rouse mode basis depends on the chain length, so no
padding or masking is needed.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .geometry import ca_to_backbone

BACKBONE_ATOMS = ("N", "CA", "C")


def read_backbone_atoms(path) -> Dict[str, np.ndarray]:
    """
    Read N, CA and C coordinates of the first chain of the first model.

    Residues are kept in file order. Alternate locations other than the first
    one are skipped. HETATM records are ignored.

    Returns:
        dict with key "CA" -> (L, 3) float32 array, and keys "N" and "C" when
        every residue has those atoms.
    """
    atoms: Dict[str, Dict[tuple, np.ndarray]] = {name: {} for name in BACKBONE_ATOMS}
    order: List[tuple] = []
    chain = None
    with open(path) as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("ENDMDL"):
                break
            if record.startswith("TER") and order:
                break
            if not record.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name not in atoms:
                continue
            if chain is None:
                chain = line[21]
            elif line[21] != chain:
                break
            key = (line[22:27],)  # residue number + insertion code
            if key in atoms[name]:
                continue  # alternate location
            atoms[name][key] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32
            )
            if name == "CA":
                order.append(key)

    out = {"CA": np.stack([atoms["CA"][k] for k in order]) if order else np.zeros((0, 3), np.float32)}
    if order and all(k in atoms["N"] and k in atoms["C"] for k in order):
        out["N"] = np.stack([atoms["N"][k] for k in order])
        out["C"] = np.stack([atoms["C"][k] for k in order])
    return out


def read_ca_coords(path) -> np.ndarray:
    """CA coordinates (L, 3) of the first chain of the first model."""
    return read_backbone_atoms(path)["CA"]


class CADataset(Dataset):
    """
    CA traces from a directory of PDB files.

    Args:
        data_dir: Directory with ``.pdb`` files (searched recursively)
        min_len: Chains shorter than this are dropped
        max_len: Chains longer than this are dropped
        max_bond: Chains with a CA-CA distance above this (chain break) are dropped
        with_backbone: Also load N and C atoms; chains without them are dropped
        files: Optional explicit list of files (overrides the directory scan)

    Each item is a dict with
        ``ca``: (L, 3) float tensor, centred, in Angstrom
        ``backbone``: (L, 3, 3) [N, CA, C], only if with_backbone
        ``index``: position in the dataset
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        min_len: int = 20,
        max_len: int = 512,
        max_bond: float = 4.8,
        with_backbone: bool = False,
        files: Optional[Sequence[str]] = None,
    ):
        if files is None:
            if data_dir is None:
                raise ValueError("data_dir or files is required")
            files = sorted(str(p) for p in Path(data_dir).rglob("*.pdb"))
            if not files:
                raise FileNotFoundError(f"No .pdb files found under {data_dir}")

        self.names: List[str] = []
        self.coords: List[torch.Tensor] = []
        self.backbones: List[torch.Tensor] = []
        self.with_backbone = with_backbone
        self.n_dropped = 0

        for path in files:
            atoms = read_backbone_atoms(path)
            ca = atoms["CA"]
            L = len(ca)
            keep = max(min_len, 2) <= L <= max_len  # a chain needs at least one bond
            if keep and max_bond is not None:
                keep = float(np.linalg.norm(ca[1:] - ca[:-1], axis=-1).max()) <= max_bond
            if keep and with_backbone:
                keep = "N" in atoms
            if not keep:
                self.n_dropped += 1
                continue
            centre = ca.mean(axis=0, keepdims=True)
            self.names.append(Path(path).stem)
            self.coords.append(torch.from_numpy(ca - centre))
            if with_backbone:
                bb = np.stack([atoms["N"], ca, atoms["C"]], axis=1) - centre[None]
                self.backbones.append(torch.from_numpy(bb))

        if not self.coords:
            raise RuntimeError("No chains left after filtering")
        self.lengths = [len(c) for c in self.coords]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        item = {"ca": self.coords[idx], "index": idx}
        if self.with_backbone:
            item["backbone"] = self.backbones[idx]
        return item

    def split(self, val_fraction: float, seed: int = 0) -> Tuple[List[int], List[int]]:
        """Deterministic train / validation index split."""
        idx = list(range(len(self)))
        random.Random(seed).shuffle(idx)
        n_val = int(round(val_fraction * len(idx)))
        return sorted(idx[n_val:]), sorted(idx[:n_val])


class LengthBatchSampler(Sampler):
    """
    Batches of chains that all have the same length.

    Args:
        lengths: Length of every chain in the dataset
        batch_size: Maximum number of chains per batch
        indices: Subset of dataset indices to draw from (default: all)
        shuffle: Shuffle chains within a length and the order of the batches
        seed: Seed of the shuffling
        max_residues: Optional cap on batch_size * length; long chains then get
            smaller batches
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        indices: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        seed: int = 0,
        max_residues: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_residues = max_residues
        self.rng = random.Random(seed)
        self.buckets: Dict[int, List[int]] = defaultdict(list)
        for i in (range(len(lengths)) if indices is None else indices):
            self.buckets[lengths[i]].append(i)

    def _size(self, length: int) -> int:
        if self.max_residues is None:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_residues // length))

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for length in sorted(self.buckets):
            idx = list(self.buckets[length])
            if self.shuffle:
                self.rng.shuffle(idx)
            size = self._size(length)
            batches.extend(idx[i:i + size] for i in range(0, len(idx), size))
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(-(-len(idx) // self._size(length)) for length, idx in self.buckets.items())


def collate_same_length(items: List[dict]) -> dict:
    """Stack items of one length into a batch."""
    batch = {
        "ca": torch.stack([it["ca"] for it in items]),
        "index": torch.tensor([it["index"] for it in items]),
    }
    if "backbone" in items[0]:
        batch["backbone"] = torch.stack([it["backbone"] for it in items])
    return batch


def _atom_line(serial: int, name: str, resseq: int, xyz, element: str, chain: str) -> str:
    return (
        f"ATOM  {serial:5d}  {name:<3s} ALA {chain}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {element}  \n"
    )


def write_ca_pdb(ca_coords, path, chain: str = "A") -> None:
    """Write a CA-only PDB file from (L, 3) coordinates in Angstrom."""
    ca = np.asarray(torch.as_tensor(ca_coords).detach().cpu(), dtype=np.float32)
    with open(path, "w") as f:
        for i, xyz in enumerate(ca):
            f.write(_atom_line(i + 1, "CA", i + 1, xyz, "C", chain))
        f.write("END\n")


def write_backbone_pdb(ca_coords, path, chain: str = "A") -> None:
    """
    Write a full backbone PDB (N, CA, C, O) from (L, 3) CA coordinates.

    N, C and O are placed with ``geometry.ca_to_backbone``; the CA positions
    are written unchanged.
    """
    ca = torch.as_tensor(ca_coords).detach().cpu().float()
    backbone = ca_to_backbone(ca).numpy()
    with open(path, "w") as f:
        f.write("REMARK  Generated with PI-Mamba, N/C/O placed around the CA trace\n")
        serial = 1
        for i, residue in enumerate(backbone):
            for name, xyz, element in zip(("N", "CA", "C", "O"), residue, ("N", "C", "C", "O")):
                f.write(_atom_line(serial, name, i + 1, xyz, element, chain))
                serial += 1
        f.write("END\n")
