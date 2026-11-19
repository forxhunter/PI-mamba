"""
Self-consistency (designability) evaluation of generated backbones.

Pipeline for one backbone:
    1. ProteinMPNN (CA-only weights) designs ``n_seqs`` sequences for the CA trace.
    2. ESMFold predicts a structure for every sequence.
    3. The predicted CA trace is compared with the generated one.
       scTM is the best TM-score and scRMSD the lowest RMSD over the sequences.

External requirements (not part of this repository):
    - a checkout of https://github.com/dauparas/ProteinMPNN with its
      ``ca_model_weights`` (pass ``proteinmpnn_dir`` or set PROTEINMPNN_DIR)
    - ESMFold through the ``transformers`` package (``facebook/esmfold_v1``,
      or a local directory with the same files). ESMFold has 3.5 B
      parameters: about 9 GB of GPU memory with the language model in half
      precision, or about 14 GB of RAM on CPU (minutes per sequence).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .data import read_ca_coords
from .geometry import kabsch_rmsd
from .metrics import tm_score


def find_proteinmpnn(proteinmpnn_dir: Optional[str] = None) -> Path:
    """Locate the ProteinMPNN checkout or raise a clear error."""
    path = proteinmpnn_dir or os.environ.get("PROTEINMPNN_DIR")
    if not path:
        raise RuntimeError(
            "ProteinMPNN is required for scTM evaluation. Clone "
            "https://github.com/dauparas/ProteinMPNN and pass --proteinmpnn_dir "
            "or set the PROTEINMPNN_DIR environment variable."
        )
    path = Path(path)
    if not (path / "protein_mpnn_run.py").is_file():
        raise FileNotFoundError(f"protein_mpnn_run.py not found in {path}")
    return path


def design_sequences(
    pdb_paths: Sequence[str],
    proteinmpnn_dir: Optional[str] = None,
    n_seqs: int = 8,
    temperature: float = 0.1,
    seed: int = 0,
    workdir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Run CA-only ProteinMPNN on a list of single-chain PDB files.

    Returns:
        dict: PDB stem -> list of ``n_seqs`` designed sequences
    """
    mpnn = find_proteinmpnn(proteinmpnn_dir)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        pdb_dir = tmp / "pdbs"
        pdb_dir.mkdir()
        for path in pdb_paths:
            (pdb_dir / Path(path).name).symlink_to(Path(path).resolve())

        parsed = tmp / "parsed.jsonl"
        common = dict(cwd=str(mpnn), check=True, capture_output=True, text=True)
        try:
            subprocess.run(
                [sys.executable, "helper_scripts/parse_multiple_chains.py",
                 f"--input_path={pdb_dir}/", f"--output_path={parsed}", "--ca_only"],
                **common,
            )
            subprocess.run(
                [sys.executable, "protein_mpnn_run.py", "--ca_only",
                 "--jsonl_path", str(parsed), "--out_folder", str(tmp / "out"),
                 "--num_seq_per_target", str(n_seqs), "--batch_size", "1",
                 "--sampling_temp", str(temperature), "--seed", str(seed)],
                **common,
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"ProteinMPNN failed:\n{err.stdout}\n{err.stderr}") from err

        designs = {}
        for fasta in sorted((tmp / "out" / "seqs").glob("*.fa")):
            records = [line.strip() for line in fasta.read_text().splitlines() if line.strip()]
            # Records alternate header / sequence; the first record is the input sequence
            seqs = [records[i + 1] for i in range(2, len(records) - 1, 2)]
            designs[fasta.stem] = seqs
    return designs


class ESMFoldPredictor:
    """ESMFold through HuggingFace ``transformers``."""

    def __init__(self, model_name: str = "facebook/esmfold_v1", device: str = "cuda"):
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForProteinFolding.from_pretrained(model_name).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.esm = self.model.esm.half()  # language model in half precision

    @torch.no_grad()
    def predict(self, sequences: Sequence[str], batch_size: int = 4):
        """
        Fold sequences of one length.

        Returns:
            ca: (N, L, 3) predicted CA coordinates in Angstrom
            plddt: (N,) mean CA pLDDT on a 0-100 scale
        """
        cas, plddts = [], []
        for i in range(0, len(sequences), batch_size):
            chunk = list(sequences[i:i + batch_size])
            tokens = self.tokenizer(chunk, return_tensors="pt", add_special_tokens=False)["input_ids"]
            out = self.model(tokens.to(self.device))
            cas.append(out.positions[-1][:, :, 1].float().cpu())  # atom14 index 1 is CA
            plddt = out.plddt[:, :, 1].float().mean(dim=-1).cpu()
            plddts.append(plddt * 100.0 if plddt.max() <= 1.0 else plddt)
        return torch.cat(cas), torch.cat(plddts)


class SelfConsistencyEvaluator:
    """
    scTM / scRMSD evaluation of backbone PDB files.

    Args:
        proteinmpnn_dir: ProteinMPNN checkout (or set PROTEINMPNN_DIR)
        esmfold_model: HuggingFace model name or local path of ESMFold
        n_seqs: ProteinMPNN sequences per backbone
        temperature: ProteinMPNN sampling temperature
        device: device for ESMFold
    """

    def __init__(
        self,
        proteinmpnn_dir: Optional[str] = None,
        esmfold_model: str = "facebook/esmfold_v1",
        n_seqs: int = 8,
        temperature: float = 0.1,
        device: str = "cuda",
        fold_batch_size: int = 4,
    ):
        self.proteinmpnn_dir = str(find_proteinmpnn(proteinmpnn_dir))
        self.n_seqs = n_seqs
        self.temperature = temperature
        self.fold_batch_size = fold_batch_size
        self.folder = ESMFoldPredictor(esmfold_model, device)

    def evaluate(self, pdb_paths: Sequence[str], seed: int = 0, verbose: bool = False) -> List[dict]:
        """
        Returns one record per PDB file with keys
        ``domain``, ``length``, ``scTM``, ``scRMSD``, ``pLDDT``, ``sequence``
        (pLDDT and sequence belong to the design with the best TM-score), and
        ``scTM_all`` with the TM-score of every design.
        """
        designs = design_sequences(
            pdb_paths, self.proteinmpnn_dir, self.n_seqs, self.temperature, seed
        )
        records = []
        for path in pdb_paths:
            name = Path(path).stem
            ca = torch.from_numpy(read_ca_coords(path))
            seqs = designs[name]
            pred, plddt = self.folder.predict(seqs, self.fold_batch_size)
            ref = ca.expand(len(seqs), -1, -1)
            tms = tm_score(pred, ref)
            rmsds = kabsch_rmsd(pred, ref)
            best = int(tms.argmax())
            record = {
                "domain": name,
                "length": len(ca),
                "scTM": tms[best].item(),
                "scRMSD": rmsds.min().item(),
                "pLDDT": plddt[best].item(),
                "sequence": seqs[best],
                "scTM_all": [round(v, 4) for v in tms.tolist()],
            }
            records.append(record)
            if verbose:
                print(f"{name}: L={record['length']} scTM={record['scTM']:.3f} "
                      f"scRMSD={record['scRMSD']:.2f} pLDDT={record['pLDDT']:.1f}", flush=True)
        return records


def summarize(records: List[dict]) -> dict:
    """Aggregate per-backbone records into the numbers usually reported."""
    if not records:
        return {"n_samples": 0}
    tm = torch.tensor([r["scTM"] for r in records])
    rmsd = torch.tensor([r["scRMSD"] for r in records])
    plddt = torch.tensor([r["pLDDT"] for r in records])
    return {
        "n_samples": len(records),
        "scTM_mean": tm.mean().item(),
        "scTM_std": tm.std().item() if len(records) > 1 else 0.0,
        "scTM_median": tm.median().item(),
        "scRMSD_mean": rmsd.mean().item(),
        "pLDDT_mean": plddt.mean().item(),
        # Common designability criteria
        "frac_scTM_above_0.5": (tm > 0.5).float().mean().item(),
        "frac_scRMSD_below_2": (rmsd < 2.0).float().mean().item(),
    }


def write_results(records: List[dict], out_dir) -> dict:
    """Write ``info.csv`` and ``summary.json`` into ``out_dir`` and return the summary."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "length", "scTM", "scRMSD", "pLDDT", "sequence"]
    with open(out_dir / "info.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
