"""
Geometry utilities for PI-Mamba.

This module implements:
1. Differentiable NeRF (Natural Extension Reference Frame) for backbone reconstruction.
2. Conversion between coordinates and torsion angles (phi, psi, omega).
3. Approximate full-backbone (N, CA, C, O) placement around a CA trace.
4. Kabsch superposition.

References:
    Parsons et al. (2005) "Practical conversion from torsion space to Cartesian space for in silico protein synthesis".
"""

import math
from typing import Optional, Tuple

import torch

# Ideal geometry constants (Engh & Huber, 1991)
# Bond lengths in Angstroms
L_N_CA = 1.458
L_CA_C = 1.525
L_C_N = 1.329
L_C_O = 1.231

# Interior bond angles in radians
A_N_CA_C = 1.939  # ~111.2 degrees
A_CA_C_N = 2.028  # ~116.2 degrees
A_C_N_CA = 2.124  # ~121.7 degrees

# CA-CA virtual bond lengths in Angstroms
CA_CA_TRANS = 3.8
CA_CA_CIS = 2.96


def normalize_vector(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v, dim=dim, keepdim=True) + eps)


def dihedral_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angle for atoms a-b-c-d (IUPAC sign convention).

    Args:
        a, b, c, d: (..., 3) coordinates
    Returns:
        angle: (...,) in range [-pi, pi]
    """
    u1 = normalize_vector(b - a)
    u2 = normalize_vector(c - b)
    u3 = normalize_vector(d - c)

    # Normals of the planes (a, b, c) and (b, c, d)
    n1 = normalize_vector(torch.cross(u1, u2, dim=-1))
    n2 = normalize_vector(torch.cross(u2, u3, dim=-1))

    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(torch.cross(n1, n2, dim=-1) * u2, dim=-1)

    return torch.atan2(y, x)


def ca_pseudo_dihedrals(ca: torch.Tensor) -> torch.Tensor:
    """
    Pseudo-dihedral of four consecutive CA atoms.

    Args:
        ca: (..., L, 3) CA coordinates
    Returns:
        (..., L-3) angles in radians. About +50 deg in a right-handed alpha
        helix and about -170 deg in a beta strand.
    """
    return dihedral_angle(ca[..., :-3, :], ca[..., 1:-2, :], ca[..., 2:-1, :], ca[..., 3:, :])


def coords_to_torsions(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract backbone torsion angles (phi, psi, omega) from full backbone coordinates.

    Args:
        coords: (B, L, 3, 3) backbone coordinates [N, CA, C] per residue

    Returns:
        phi: (B, L) phi angles in radians, phi[:, 0] = 0
        psi: (B, L) psi angles in radians, psi[:, -1] = 0
        omega: (B, L) omega angles in radians, omega[:, 0] = 0
               (omega_i is the torsion of the peptide bond before residue i)
    """
    N, CA, C = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
    zero = coords.new_zeros(coords.shape[0], 1)

    # Phi: C_{i-1} - N_i - CA_i - C_i
    phi = dihedral_angle(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
    # Psi: N_i - CA_i - C_i - N_{i+1}
    psi = dihedral_angle(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])
    # Omega: CA_{i-1} - C_{i-1} - N_i - CA_i
    omega = dihedral_angle(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])

    return (
        torch.cat([zero, phi], dim=1),
        torch.cat([psi, zero], dim=1),
        torch.cat([zero, omega], dim=1),
    )


def nerf_step(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bond_len: float,
    bond_angle: float,
    torsion: torch.Tensor,
) -> torch.Tensor:
    """
    Place atom d given a, b, c and internal coordinates.

    Args:
        a, b, c: (B, 3) previous atoms
        bond_len: scalar length c-d
        bond_angle: scalar interior angle b-c-d (radians)
        torsion: (B,) dihedral a-b-c-d (radians)

    Returns:
        d: (B, 3) placed atom
    """
    bc = normalize_vector(c - b)
    n = normalize_vector(torch.cross(b - a, bc, dim=-1))  # normal of the plane (a, b, c)
    m = torch.cross(n, bc, dim=-1)  # in-plane direction perpendicular to bc

    # d - c makes the angle (pi - bond_angle) with the bc direction
    along = -bond_len * math.cos(bond_angle)
    radial = bond_len * math.sin(bond_angle)

    return (
        c
        + along * bc
        + (radial * torch.cos(torsion)).unsqueeze(-1) * m
        + (radial * torch.sin(torsion)).unsqueeze(-1) * n
    )


def nerf_build_backbone(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct backbone (N, CA, C) from torsion angles using NeRF with ideal
    bond lengths and angles.

    Uses the conventions of ``coords_to_torsions``: phi[:, 0], omega[:, 0] and
    psi[:, -1] are ignored.

    Args:
        phi: (B, L)
        psi: (B, L)
        omega: (B, L)

    Returns:
        coords: (B, L, 3, 3) N, CA, C coordinates. The first residue is placed
        in the x-y plane with CA at the origin.
    """
    B, L = phi.shape
    device, dtype = phi.device, phi.dtype

    # First residue in the x-y plane
    n1 = torch.tensor([-L_N_CA, 0.0, 0.0], device=device, dtype=dtype).expand(B, 3)
    ca1 = torch.zeros(B, 3, device=device, dtype=dtype)
    c1 = torch.tensor(
        [L_CA_C * math.cos(math.pi - A_N_CA_C), L_CA_C * math.sin(math.pi - A_N_CA_C), 0.0],
        device=device, dtype=dtype,
    ).expand(B, 3)

    residues = [torch.stack([n1, ca1, c1], dim=1)]
    prev_n, prev_ca, prev_c = n1, ca1, c1

    for i in range(1, L):
        # N_i: torsion psi_{i-1} = N_{i-1} - CA_{i-1} - C_{i-1} - N_i
        n_curr = nerf_step(prev_n, prev_ca, prev_c, L_C_N, A_CA_C_N, psi[:, i - 1])
        # CA_i: torsion omega_i = CA_{i-1} - C_{i-1} - N_i - CA_i
        ca_curr = nerf_step(prev_ca, prev_c, n_curr, L_N_CA, A_C_N_CA, omega[:, i])
        # C_i: torsion phi_i = C_{i-1} - N_i - CA_i - C_i
        c_curr = nerf_step(prev_c, n_curr, ca_curr, L_CA_C, A_N_CA_C, phi[:, i])

        residues.append(torch.stack([n_curr, ca_curr, c_curr], dim=1))
        prev_n, prev_ca, prev_c = n_curr, ca_curr, c_curr

    return torch.stack(residues, dim=1)


# Ideal trans peptide: positions of C(i), O(i) and N(i+1) relative to CA(i) in
# the peptide plane, as (component along CA(i)->CA(i+1), component along the
# in-plane normal), for a CA-CA distance of 3.80 A.
_PEPTIDE_PLANE = {"C": (1.4304, 0.5287), "O": (1.6551, 1.7390), "N": (2.3941, -0.3864)}

# Weights of the linear estimate of the in-plane normal from the neighbouring
# CA-CA vector v: normal ~ w_perp * v_perp + w_cross * cross(...). Fitted on
# six crystal structures (1008 peptides); the estimate from the following
# residue has a median error of about 15 degrees.
_NORMAL_FROM_NEXT = (0.0208, 0.2278)       # v = CA(i+2) - CA(i+1), cross(e, v)
_NORMAL_FROM_PREVIOUS = (0.0419, -0.0451)  # v = CA(i) - CA(i-1), cross(v, e); last peptide only


def _unit_or_fallback(v: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Normalise v; where it vanishes, return some unit vector perpendicular to e."""
    axis = torch.zeros_like(e)
    axis[..., 0] = 1.0
    axis[e[..., 0].abs() > 0.9] = torch.tensor([0.0, 1.0, 0.0], dtype=e.dtype, device=e.device)
    fallback = torch.cross(e, axis, dim=-1)
    degenerate = v.norm(dim=-1, keepdim=True) < 1e-6
    return normalize_vector(torch.where(degenerate, fallback, v))


def ca_to_backbone(ca: torch.Tensor) -> torch.Tensor:
    """
    Place N, C and O atoms around a CA trace.

    The CA positions are kept exactly. Every peptide unit CA(i) - C(i) - N(i+1)
    - CA(i+1) is built as an ideal planar trans peptide. The plane contains the
    CA(i) - CA(i+1) axis; its orientation about that axis is estimated from the
    direction of the following CA-CA vector. With CA-CA distances of 3.8 A the
    bond lengths CA-C, C-N, N-CA and C=O are ideal. The N-CA-C angle is not
    constrained, and the plane orientation is an estimate (median error about
    15 degrees on crystal structures, giving RMS errors of about 0.3 A for N
    and C and 1.1 A for O), so this is a model for visualisation and for tools
    that need backbone records, not a refined structure.

    Args:
        ca: (L, 3) CA coordinates in Angstrom, L >= 3

    Returns:
        (L, 4, 3) coordinates of N, CA, C, O
    """
    L = ca.shape[0]
    if L < 3:
        raise ValueError("ca_to_backbone needs at least 3 residues")

    b = ca[1:] - ca[:-1]  # (L-1, 3) virtual bonds
    d = b.norm(dim=-1, keepdim=True)
    e = b / (d + 1e-8)

    def perp(v, axis):
        return v - (v * axis).sum(-1, keepdim=True) * axis

    # In-plane normal of every peptide
    w_perp, w_cross = _NORMAL_FROM_NEXT
    nxt = b[1:]
    normal = w_perp * perp(nxt, e[:-1]) + w_cross * torch.cross(e[:-1], nxt, dim=-1)
    w_perp, w_cross = _NORMAL_FROM_PREVIOUS
    prev = b[-2:-1]
    last = w_perp * perp(prev, e[-1:]) + w_cross * torch.cross(prev, e[-1:], dim=-1)
    p = _unit_or_fallback(torch.cat([normal, last], dim=0), e)  # (L-1, 3)

    scale = d / CA_CA_TRANS

    def place(name):
        along, across = _PEPTIDE_PLANE[name]
        return ca[:-1] + along * scale * e + across * p

    c_pos, o_pos, n_next = place("C"), place("O"), place("N")

    # Chain ends: N of the first and C, O of the last residue, with the ideal
    # N-CA-C angle in the plane of the adjacent peptide
    def end_atom(origin, partner, side, bond):
        u = normalize_vector(partner - origin)
        v = _unit_or_fallback(perp(side, u), u)
        return origin + bond * (math.cos(A_N_CA_C) * u + math.sin(A_N_CA_C) * v)

    n_first = end_atom(ca[0], c_pos[0], -p[0], L_N_CA)
    c_last = end_atom(ca[-1], n_next[-1], p[-1], L_CA_C)
    o_last = c_last + L_C_O * normalize_vector(normalize_vector(c_last - ca[-1]) + p[-1])

    n_pos = torch.cat([n_first[None], n_next], dim=0)
    c_pos = torch.cat([c_pos, c_last[None]], dim=0)
    o_pos = torch.cat([o_pos, o_last[None]], dim=0)
    return torch.stack([n_pos, ca, c_pos, o_pos], dim=1)


def kabsch_align(
    mobile: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Rigidly superpose ``mobile`` onto ``target`` (least-squares, no reflection).

    Args:
        mobile: (B, L, 3)
        target: (B, L, 3)
        mask: optional (B, L) weights; the fit uses only positions with mask > 0,
              the transform is applied to all positions

    Returns:
        (B, L, 3) transformed copy of ``mobile``
    """
    if mask is None:
        mask = mobile.new_ones(mobile.shape[:2])
    w = mask.unsqueeze(-1).to(mobile.dtype)
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    mu_m = (mobile * w).sum(dim=1, keepdim=True) / wsum
    mu_t = (target * w).sum(dim=1, keepdim=True) / wsum
    m, t = mobile - mu_m, target - mu_t

    H = (m * w).transpose(1, 2) @ t  # (B, 3, 3)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)  # rotates mobile onto target

    return m @ R.transpose(1, 2) + mu_t


def kabsch_rmsd(
    a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """RMSD (B,) between ``a`` and ``b`` after optimal superposition."""
    if mask is None:
        mask = a.new_ones(a.shape[:2])
    aligned = kabsch_align(a, b, mask)
    sq = ((aligned - b) ** 2).sum(dim=-1) * mask
    return torch.sqrt(sq.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8))


def random_rotations(n: int, device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    """Uniformly distributed rotation matrices, shape (n, 3, 3)."""
    # Draw on the device of the generator (CPU when there is none)
    q = torch.randn(n, 4, generator=generator, dtype=dtype,
                    device=generator.device if generator is not None else None)
    q = (q / q.norm(dim=-1, keepdim=True)).to(device)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(n, 3, 3)
