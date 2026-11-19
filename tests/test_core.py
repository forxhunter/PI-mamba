"""
Unit tests for the PI-Mamba core. Run with either

    python tests/test_core.py
    pytest tests/
"""

import math
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backbone_pi_mamba import PIMambaBackbone, project_bonds
from src.data import CADataset, LengthBatchSampler, read_ca_coords, write_backbone_pdb, write_ca_pdb
from src.geometry import (
    CA_CA_TRANS, coords_to_torsions, kabsch_rmsd, nerf_build_backbone, random_rotations,
)
from src.losses import HydrogenBondLoss, RamachandranLoss
from src.metrics import assign_secondary_structure, HELIX, STRAND, tm_score_superposed
from src.model_io import MODEL_CHOICES, build_model, load_checkpoint, save_checkpoint
from src.pi_mamba_layer import PhysicsInformedMamba
from src.rouse_physics import RouseTransform, compute_rouse_eigenvalues, compute_rouse_eigenvectors

DATA_DIR = REPO_ROOT / "Distilled_trainset"


def test_rouse_basis_diagonalises_chain_laplacian():
    N = 37
    K = 2 * torch.eye(N) - torch.diag(torch.ones(N - 1), 1) - torch.diag(torch.ones(N - 1), -1)
    K[0, 0] = K[-1, -1] = 1
    V = compute_rouse_eigenvectors(N).double()
    lam = compute_rouse_eigenvalues(N).double()
    assert torch.allclose(V.T @ V, torch.eye(N, dtype=torch.double), atol=1e-5)
    assert torch.allclose(V.T @ K.double() @ V, torch.diag(lam), atol=1e-5)


def test_rouse_transform_round_trip_for_any_length():
    tr = RouseTransform(max_length=64)
    for L in (7, 50, 100, 300):  # includes lengths above and below max_length
        x = torch.randn(2, L, 5)
        x_back = tr.inverse_transform(tr.forward_transform(x))
        assert torch.allclose(x, x_back, atol=1e-4), L
        assert torch.allclose(tr.get_eigenvalues(L), compute_rouse_eigenvalues(L))


def test_spectrum_variants():
    d_state = 16
    for L in (50, 51, 300):
        rouse = RouseTransform(spectrum="rouse").get_state_rates(L, d_state)
        assert torch.allclose(rouse, compute_rouse_eigenvalues(L)[:d_state])
        # alpha = 1 is pure Rouse; alpha < 1 stays inside the Rouse range
        full = RouseTransform(spectrum="interpolated", alpha=1.0).get_state_rates(L, d_state)
        assert torch.allclose(full, rouse)
        for alpha in (0.0, 0.5):
            mixed = RouseTransform(spectrum="interpolated", alpha=alpha).get_state_rates(L, d_state)
            assert not torch.allclose(mixed, rouse)
            assert mixed.min() >= 0 and mixed.max() <= rouse.max() + 1e-6

    # The scrambled control draws modes from the whole spectrum and gives every
    # state channel the same rate at every chain length
    a = RouseTransform(spectrum="scrambled").get_state_rates(50, d_state)
    b = RouseTransform(spectrum="scrambled").get_state_rates(300, d_state)
    assert torch.equal(a, b) and a.max() > 1.0 and a.min() >= 0
    other_seed = RouseTransform(spectrum="scrambled", seed=7).get_state_rates(50, d_state)
    assert not torch.equal(a, other_seed)
    # The random part of the interpolation is the same draw at every length
    r50 = RouseTransform(spectrum="interpolated", alpha=0.0).get_state_rates(50, d_state)
    r300 = RouseTransform(spectrum="interpolated", alpha=0.0).get_state_rates(300, d_state)
    assert torch.allclose(r50 / r50.max(), r300 / r300.max(), atol=1e-5)

    assert RouseTransform().get_state_rates(10, 64).shape == (64,)


def test_learned_a_starts_in_the_rouse_range():
    layer = PhysicsInformedMamba(d_model=16, d_state=64, n_groups=4, use_physics=False)
    rates = torch.exp(layer.A_log)
    reference = RouseTransform().get_state_rates(128, 64)
    assert rates.max() <= reference.max() + 1e-6 and rates.min() > 0


def test_time_embedding_resolves_time():
    from src.backbone_pi_mamba import TimeEmbedding
    emb = TimeEmbedding(64)
    half = emb.d_model // 2
    t = torch.linspace(0, 1, 101)
    freqs = torch.exp(-math.log(emb.max_period) * torch.arange(half) / half)
    args = emb.time_scale * t[:, None] * freqs[None]
    raw = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # Several independent directions, not a nearly linear function of t
    sv = torch.linalg.svdvals(raw - raw.mean(0))
    assert (sv / sv[0] > 0.1).sum() >= 4
    assert torch.nn.functional.cosine_similarity(raw[0], raw[-1], dim=0) < 0.9


def test_residual_stream_is_not_renormalised():
    """Every encoder must compute h <- h + f(AdaLN(h, t)), so a zero update is a no-op."""
    for name in MODEL_CHOICES:
        model = build_model(dict(model=name, d_model=32, n_layers=2, d_state=8, n_groups=4,
                                 n_structure_layers=1, n_heads=4, dropout=0.0)).eval()
        h, t_emb = model._embed(torch.randn(2, 20, 3), torch.rand(2))
        if name == "pi_mamba":
            for layer in model.pi_mamba_layers:
                torch.nn.init.zeros_(layer.pi_mamba.out_proj.weight)
        else:
            for layer in model.attn_layers:
                for module in layer.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.zeros_(module.weight)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    if isinstance(module, torch.nn.MultiheadAttention):
                        torch.nn.init.zeros_(module.out_proj.weight)
                        torch.nn.init.zeros_(module.out_proj.bias)
        out, _ = model.encode(h, t_emb)
        assert torch.allclose(out, h, atol=1e-5), name


def test_short_chains_do_not_crash_metrics():
    for L in (3, 5, 6, 7):
        ss = assign_secondary_structure(torch.randn(2, L, 3))
        assert ss.shape == (2, L)


def _scan_inputs(layer, B, L):
    G, N = layer.n_groups, layer.d_state
    x = torch.randn(B, L, layer.d_inner, dtype=torch.double)
    u = torch.rand(B, L, G, dtype=torch.double) * 0.5
    Bv = torch.randn(B, L, N, G, dtype=torch.double)
    Cv = torch.randn(B, L, N, G, dtype=torch.double)
    return x, u, Bv, Cv


def test_chunked_scan_matches_sequential_recurrence():
    for use_physics in (True, False):
        layer = PhysicsInformedMamba(
            d_model=16, d_state=8, n_groups=4, use_physics=use_physics, scan_chunk=7
        ).double()
        for L in (5, 7, 30):  # shorter than, equal to and longer than the chunk
            x, u, Bv, Cv = _scan_inputs(layer, 2, L)
            rate = layer._decay_rates(L).double()
            A = torch.exp(-rate.view(1, 1, layer.d_state, -1) * u.unsqueeze(2))
            A = A.expand(2, L, layer.d_state, layer.n_groups)
            y_ref = layer._ssm_forward(x, A, Bv, Cv)
            y = layer._ssm_scan(x, rate, u, Bv, Cv)
            assert torch.allclose(y, y_ref, atol=1e-9), (use_physics, L)


def test_scan_is_stable_for_fast_decay():
    layer = PhysicsInformedMamba(d_model=16, d_state=8, n_groups=4, scan_chunk=8)
    x, u, Bv, Cv = (v.float() for v in _scan_inputs(layer, 1, 40))
    y = layer._ssm_scan(x, layer._decay_rates(40), u * 1e4, Bv, Cv)
    assert torch.isfinite(y).all()


def test_layer_gradients_with_and_without_checkpointing():
    torch.manual_seed(0)
    grads = []
    for ckpt in (True, False):
        torch.manual_seed(1)
        layer = PhysicsInformedMamba(d_model=16, d_state=8, n_groups=4, checkpoint_scan=ckpt).train()
        torch.manual_seed(2)
        y, info = layer(torch.randn(2, 40, 16), return_physics=True)
        y.pow(2).sum().backward()
        assert info["tau"].shape == (2, 40, 4)
        assert 0.0 <= info["A_min"] <= info["A_max"] <= 1.0
        for name, p in layer.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        grads.append(layer.tau_proj[0].weight.grad.clone())
    assert grads[0].abs().sum() > 0
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_backbone_forward_and_sample():
    for name in MODEL_CHOICES:
        model = build_model(dict(model=name, d_model=32, n_layers=2, d_state=8, n_groups=4,
                                 n_structure_layers=1, n_heads=4)).eval()
        coords, t = torch.randn(2, 24, 3), torch.rand(2)
        velocity, omega_logits = model(coords, t)
        assert velocity.shape == (2, 24, 3) and omega_logits.shape == (2, 24, 2)
        out = model(coords, t, return_physics=True, return_hidden=True)
        assert len(out) == 4 and out[3].shape == (2, 24, 32)

        samples = model.sample(length=20, n_samples=3, n_steps=4)
        assert samples.shape == (3, 20, 3)
        # The sampler relaxes bonds and clashes together, so bond lengths end
        # up near 3.8 rather than exactly on it, as in real structures
        bonds = (samples[:, 1:] - samples[:, :-1]).norm(dim=-1)
        assert (bonds - CA_CA_TRANS).abs().mean() < 0.4, bonds
        # project_bonds alone still gives exact bond lengths
        exact = model.sample(length=20, n_samples=2, n_steps=4, relax=False)
        exact_bonds = (exact[:, 1:] - exact[:, :-1]).norm(dim=-1)
        assert torch.allclose(exact_bonds, torch.full_like(exact_bonds, CA_CA_TRANS), atol=1e-3)


def test_learned_a_ablation_has_free_transition():
    model = PIMambaBackbone(d_model=32, n_layers=1, d_state=8, n_groups=4, use_physics=False)
    assert any(n.endswith("A_log") for n, _ in model.named_parameters())
    physics = PIMambaBackbone(d_model=32, n_layers=1, d_state=8, n_groups=4, use_physics=True)
    assert not any(n.endswith("A_log") for n, _ in physics.named_parameters())


def test_checkpoint_round_trip():
    config = dict(model="pi_mamba", d_model=32, n_layers=2, d_state=8, n_groups=4,
                  n_structure_layers=1, spectrum="scrambled")
    model = build_model(config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, config, step=3)
        loaded, ckpt = load_checkpoint(path)
    assert ckpt["step"] == 3
    assert loaded.pi_mamba_layers[0].pi_mamba.rouse_transform.spectrum == "scrambled"
    x, t = torch.randn(1, 16, 3), torch.rand(1)
    assert torch.allclose(model.eval()(x, t)[0], loaded(x, t)[0])


def test_relax_chain_removes_clashes_and_keeps_local_geometry():
    """The relaxation must resolve overlaps without unwinding secondary structure."""
    from src.backbone_pi_mamba import relax_chain
    from src.metrics import bond_lengths, clash_fraction

    # Two helices folded back onto each other so that they overlap
    helix = _ideal_helix(30)
    other = helix.flip(0) + torch.tensor([2.0, 1.0, 0.0])
    tangled = torch.cat([helix, other])[None]
    assert clash_fraction(tangled).item() > 0.3

    relaxed = relax_chain(tangled, n_steps=400)
    assert clash_fraction(relaxed).item() < 0.05, "overlaps should be resolved"
    # Bond lengths end up near 3.8 rather than exactly on it: the relaxation
    # balances them against the overlap term instead of enforcing them
    deviation = (bond_lengths(relaxed) - CA_CA_TRANS).abs()
    assert deviation.mean() < 0.1 and deviation.max() < 1.0, deviation
    # the helical i to i+4 distance is held by the local restraint
    before = (helix[4:] - helix[:-4]).norm(dim=-1).mean()
    after = (relaxed[0, 4:30] - relaxed[0, :26]).norm(dim=-1).mean()
    assert (after - before).abs() < 0.6, f"helix geometry drifted: {before:.2f} -> {after:.2f}"
    # the centroid is preserved
    assert torch.allclose(relaxed.mean(1), tangled.mean(1), atol=1e-3)


def test_relax_chain_leaves_a_clean_chain_alone():
    """A structure that already satisfies both constraints should barely move."""
    from src.backbone_pi_mamba import relax_chain
    from src.metrics import clash_fraction

    clean = project_bonds(_ideal_helix(40)[None])
    assert clash_fraction(clean).item() == 0.0
    relaxed = relax_chain(clean, n_steps=200)
    assert (relaxed - clean).norm(dim=-1).max() < 0.5


def test_project_bonds_keeps_directions_and_centroid():
    x = torch.randn(2, 30, 3) * 3
    y = project_bonds(x)
    bx, by = x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1]
    assert torch.allclose(by.norm(dim=-1), torch.full((2, 29), CA_CA_TRANS), atol=1e-4)
    assert torch.allclose(torch.nn.functional.cosine_similarity(bx, by, dim=-1), torch.ones(2, 29), atol=1e-5)
    assert torch.allclose(x.mean(1), y.mean(1), atol=1e-4)


def test_nerf_round_trip():
    torch.manual_seed(0)
    B, L = 3, 25
    phi, psi = (torch.rand(B, L) * 2 - 1) * math.pi, (torch.rand(B, L) * 2 - 1) * math.pi
    omega = torch.full((B, L), math.pi) + 0.1 * torch.randn(B, L)
    coords = nerf_build_backbone(phi, psi, omega)
    phi2, psi2, omega2 = coords_to_torsions(coords)

    def close(a, b):
        return torch.allclose(torch.cos(a), torch.cos(b), atol=1e-3) and \
            torch.allclose(torch.sin(a), torch.sin(b), atol=1e-3)

    assert close(phi[:, 1:], phi2[:, 1:])
    assert close(psi[:, :-1], psi2[:, :-1])
    assert close(omega[:, 1:], omega2[:, 1:])
    # Ideal bond lengths, and a trans peptide gives CA-CA close to 3.8 A
    assert torch.allclose((coords[:, :, 1] - coords[:, :, 0]).norm(dim=-1), torch.full((B, L), 1.458), atol=1e-3)
    ca = coords[:, :, 1]
    assert ((ca[:, 1:] - ca[:, :-1]).norm(dim=-1) - 3.8).abs().max() < 0.1


def _ideal_helix(L):
    # Alpha helix: 1.5 A rise and 100 degrees per residue, CA radius 2.3 A
    i = torch.arange(L, dtype=torch.float32)
    ang = torch.deg2rad(100.0 * i)
    return torch.stack([2.3 * torch.cos(ang), 2.3 * torch.sin(ang), 1.5 * i], dim=-1)


def _ideal_strand(L):
    i = torch.arange(L, dtype=torch.float32)
    return torch.stack([3.3 * i, 0.9 * (-1.0) ** i, 0.3 * torch.sin(i)], dim=-1)


def test_secondary_structure_assignment():
    helix = assign_secondary_structure(_ideal_helix(30)[None])[0]
    assert (helix[2:-3] == HELIX).all()
    strand = assign_secondary_structure(_ideal_strand(20)[None])[0]
    assert (strand[2:-3] == STRAND).all()
    walk = torch.cumsum(torch.nn.functional.normalize(torch.randn(1, 200, 3), dim=-1) * 3.8, dim=1)
    assert (assign_secondary_structure(walk) == HELIX).float().mean() < 0.1


def test_kabsch_and_tm_score():
    x = torch.randn(4, 50, 3) * 8
    R = random_rotations(4)
    y = x @ R.transpose(1, 2) + torch.randn(4, 1, 3)
    assert kabsch_rmsd(x, y).max() < 1e-3
    assert tm_score_superposed(x, y).min() > 0.999
    assert tm_score_superposed(x, torch.randn(4, 50, 3) * 8).max() < 0.5
    assert torch.allclose(torch.det(R), torch.ones(4), atol=1e-4)


def test_prior_losses_have_gradients():
    ca = torch.cumsum(torch.nn.functional.normalize(torch.randn(2, 60, 3), dim=-1) * 3.8, dim=1)
    ca.requires_grad_(True)
    for loss_fn in (HydrogenBondLoss(), RamachandranLoss()):
        loss = loss_fn(ca)
        (grad,) = torch.autograd.grad(loss, ca)
        assert loss.item() > 0 and grad.abs().sum() > 0, type(loss_fn).__name__
    helix = _ideal_helix(60)[None]
    assert HydrogenBondLoss().compute_helix_loss(helix).item() < 1e-3
    assert RamachandranLoss()(helix).item() < 1e-6


def test_dataset_and_sampler():
    files = sorted(str(p) for p in DATA_DIR.glob("*.pdb"))[:200]
    ds = CADataset(files=files, min_len=50, max_len=300)
    assert len(ds) > 100
    item = ds[0]
    assert item["ca"].shape == (ds.lengths[0], 3)
    assert item["ca"].mean(0).abs().max() < 1e-3
    train_idx, val_idx = ds.split(0.1, seed=0)
    assert not set(train_idx) & set(val_idx) and len(train_idx) + len(val_idx) == len(ds)
    sampler = LengthBatchSampler(ds.lengths, batch_size=8, indices=train_idx, max_residues=1200)
    seen = []
    for batch in sampler:
        assert len({ds.lengths[i] for i in batch}) == 1
        assert len(batch) * ds.lengths[batch[0]] <= max(1200, ds.lengths[batch[0]])
        seen.extend(batch)
    assert sorted(seen) == train_idx and len(sampler) == len(list(sampler))


def test_pdb_writers_round_trip():
    ca = _ideal_helix(12) + 5.0
    with tempfile.TemporaryDirectory() as tmp:
        write_ca_pdb(ca, Path(tmp) / "ca.pdb")
        write_backbone_pdb(ca, Path(tmp) / "bb.pdb")
        for name in ("ca.pdb", "bb.pdb"):
            back = torch.from_numpy(read_ca_coords(Path(tmp) / name))
            assert torch.allclose(back, ca, atol=2e-3), name
        n_atoms = sum(line.startswith("ATOM") for line in open(Path(tmp) / "bb.pdb"))
    assert n_atoms == 48


def test_ca_to_backbone_builds_ideal_peptides():
    from src.geometry import ca_to_backbone
    ca = project_bonds(torch.cumsum(torch.randn(1, 40, 3), dim=1))[0]
    bb = ca_to_backbone(ca)
    N, CA, C, O = bb.unbind(1)
    assert torch.equal(CA, ca)
    for dist, ideal in (((N - CA).norm(dim=-1), 1.458), ((C - CA).norm(dim=-1), 1.525),
                        ((C[:-1] - N[1:]).norm(dim=-1), 1.329), ((O - C).norm(dim=-1), 1.231)):
        assert torch.allclose(dist, torch.full_like(dist, ideal), atol=2e-3)
    omega = coords_to_torsions(bb[None, :, :3])[2][0, 1:]
    assert torch.allclose(omega.abs(), torch.full_like(omega, math.pi), atol=1e-2)
    # A straight chain has no defined plane orientation and must still give finite output
    straight = torch.zeros(10, 3)
    straight[:, 0] = 3.8 * torch.arange(10)
    assert torch.isfinite(ca_to_backbone(straight)).all()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
