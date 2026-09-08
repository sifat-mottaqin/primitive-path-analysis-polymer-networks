# type: ignore
"""
Simple entanglement analysis engine.

Kept logic only:
  1. Detect local entangled zones from smoothed local GLN signal.
  2. After zones are detected, find neighboring DIFFERENT zone pairs.
  3. For each neighboring zone pair, compute one full-zone GLN/COE.
  4. A zone cannot entangle with itself, but one zone can entangle with many other zones.

Removed on purpose:
  - same-zone/self-entanglement special handling
  - auxiliary/weak supporting zone logic
  - temporal consensus logic
  - contact-cluster COE logic

box_bounds expected as (3,2): [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
"""

import numpy as np
from collections import deque
from scipy.spatial import cKDTree


# USER SETTINGS
allowed_types = {2}
exclude_types = {1}

# Bead-level cutoff used to compute local GLN signal.
cutoff_d = 1.5

# Same-chain nearby beads separated by <= n_sep bonds are ignored when
# computing local proximity contacts.
n_sep = 4

# Local GLN window size and smoothing window.
n_bonds = 6
n_avg = 4

# Zone detection threshold. This is deliberately tiny because the local GLN
# signal already decides whether a bead has nonzero entanglement activity.
zone_threshold = 1e-12

# Final zone-neighbor search. The simple algorithm uses 2 * cutoff_d.
ZONE_NEIGHBOR_CUTOFF_FACTOR = 2.0

# Final zone-pair COE acceptance threshold.
# Use 0.5 for conservative GLN-based COEs. Set to 0.0 only for debugging.
LN_pair_abs_threshold = 0.0

# What field to append to train_atoms dump.
ENT_DEGREE_FIELD = "smooth_LN_abs"  # "smooth_LN_abs" or "local_LN_abs"

debug_prints = False


def analyze_entanglements(bead_data, bond_data, box_bounds):
    """
    Args:
        bead_data: Nx6 [id, type, mol, x, y, z]
        bond_data: Mx8 [index, bead1_idx, bead2_idx, rx, ry, rz, rn, f]
        box_bounds: 3x2 [[xlo,xhi],[ylo,yhi],[zlo,zhi]]

    Returns dict:
        nChains, nEntanglements, entanglement_rows,
        bead_index, molid, smooth_ln_abs, zone_members, zone_members_idx
    """
    if debug_prints:
        print("\n=== SIMPLE ENTANGLEMENT ANALYSIS STARTING ===")

    N = len(bead_data)
    bead_ids = bead_data[:, 0].astype(int)
    bead_types = bead_data[:, 1].astype(int)
    bead_mols_input = bead_data[:, 2].astype(int)
    pos = bead_data[:, 3:6].astype(float)

    bond_topo_idx = bond_data[:, 1:3].astype(int)

    if np.any(bead_mols_input != 0):
        molid_np = bead_mols_input.copy().astype(np.int64)
    else:
        molid_np = _get_molid_from_bonds(N, bond_topo_idx)

    adj = _build_adjacency(N, bond_topo_idx)

    blocked = np.isin(bead_types, np.array(list(exclude_types))) if exclude_types else np.zeros(N, dtype=bool)
    passable = ~blocked

    box_lengths = np.array([
        box_bounds[0, 1] - box_bounds[0, 0],
        box_bounds[1, 1] - box_bounds[1, 0],
        box_bounds[2, 1] - box_bounds[2, 0],
    ], dtype=float)

    pos_wrapped = pos.copy()
    for dim in range(3):
        L = box_lengths[dim]
        if L > 0:
            pos_wrapped[:, dim] = ((pos[:, dim] - box_bounds[dim, 0]) % L)

    tree = cKDTree(pos_wrapped, boxsize=tuple(box_lengths))

    # 1. Bead proximity pairs for local GLN signal 
    prox_pairs = tree.query_pairs(cutoff_d, output_type="ndarray")
    prox_pairs_i, prox_pairs_j, prox_pairs_d = [], [], []

    for i, j in prox_pairs:
        i = int(i)
        j = int(j)

        if not passable[i] or not passable[j]:
            continue

        # Ignore very nearby beads along the same chain.
        if molid_np[i] == molid_np[j]:
            if _graph_distance_leq(adj, i, j, n_sep, passable):
                continue

        d = _pbc_dist(pos[i], pos[j], box_lengths)
        if d <= cutoff_d:
            prox_pairs_i.append(i)
            prox_pairs_j.append(j)
            prox_pairs_d.append(float(d))

    local_ln = np.zeros(N, dtype=float)
    local_ln_abs = np.zeros(N, dtype=float)
    smooth_ln = np.zeros(N, dtype=float)
    smooth_ln_abs = np.zeros(N, dtype=float)

    if len(prox_pairs_i) == 0:
        return _empty_result(molid_np, adj, smooth_ln_abs, bead_ids)
  
    # 2. Local GLN around nearby bead pairs   
    best_ln = np.zeros(N, dtype=float)
    best_ln_abs = np.zeros(N, dtype=float)
    half_span = int(np.ceil(n_bonds / 2.0))

    for i, j, _d in zip(prox_pairs_i, prox_pairs_j, prox_pairs_d):
        win_i, _ = _bfs_ball(adj, i, half_span, passable)
        win_j, _ = _bfs_ball(adj, j, half_span, passable)

        if len(win_i) < 2 or len(win_j) < 2:
            continue

        win_i = np.array(win_i, dtype=int)
        win_j = np.array(win_j, dtype=int)

        ln_val, _W = _linking_number_and_weight(pos[win_i], pos[win_j])
        ln_abs = abs(float(ln_val))

        if ln_abs > best_ln_abs[i]:
            best_ln_abs[i] = ln_abs
            best_ln[i] = ln_val
        if ln_abs > best_ln_abs[j]:
            best_ln_abs[j] = ln_abs
            best_ln[j] = ln_val

    local_ln[:] = best_ln
    local_ln_abs[:] = np.abs(local_ln)
 
    # 3. Smooth local GLN signal along bonded contour   
    if n_avg > 0:
        for i in range(N):
            if not passable[i]:
                continue
            nodes, _ = _bfs_ball(adj, i, n_avg, passable)
            if nodes:
                nodes = np.array(nodes, dtype=int)
                smooth_ln[i] = np.mean(local_ln[nodes])
                smooth_ln_abs[i] = np.mean(local_ln_abs[nodes])
    else:
        smooth_ln[:] = local_ln
        smooth_ln_abs[:] = local_ln_abs
  
    # 4. Zone detection: connected high-signal regions on same molecule
    zone_mask = (smooth_ln_abs > zone_threshold) & passable
    zone_id_np, zone_to_nodes, zone_to_molid = _connected_components_zones(adj, zone_mask, molid_np)
  
    # 5. Zone-pair candidate detection using 2*cutoff_d
    zone_neighbor_cutoff = float(ZONE_NEIGHBOR_CUTOFF_FACTOR) * float(cutoff_d)
    prox_zone_pairs = tree.query_pairs(zone_neighbor_cutoff, output_type="ndarray")

    coupled = {}
    for i, j in prox_zone_pairs:
        i = int(i)
        j = int(j)
        zi = int(zone_id_np[i])
        zj = int(zone_id_np[j])

        if zi <= 0 or zj <= 0:
            continue
        if zi == zj:
            # Simple requested rule: a zone cannot entangle with itself.
            continue

        d = _pbc_dist(pos[i], pos[j], box_lengths)
        if d > zone_neighbor_cutoff:
            continue

        za, zb = (zi, zj) if zi <= zj else (zj, zi)
        key = (int(za), int(zb))

        if key not in coupled:
            coupled[key] = {"min_dist": float(d), "count_edges": 1}
        else:
            coupled[key]["count_edges"] += 1
            if d < coupled[key]["min_dist"]:
                coupled[key]["min_dist"] = float(d)

    # 6. One COE per neighboring different-zone pair 
    zone_order_cache = {
        int(zid): _order_zone_nodes_as_path(nodes, adj, molid_np)
        for zid, nodes in zone_to_nodes.items()
    }

    rows_for_csv = []
    ent_id = 0

    for za, zb in sorted(coupled.keys()):
        A_list = zone_order_cache.get(int(za), [])
        B_list = zone_order_cache.get(int(zb), [])
        if len(A_list) < 2 or len(B_list) < 2:
            continue

        A = np.asarray(A_list, dtype=int)
        B = np.asarray(B_list, dtype=int)

        LNpair, _Wpair, COE, i_bond, j_bond, _nB1, _nB2 = _linking_number_weight_and_coe(
            pos[A], pos[B], box_bounds
        )

        if np.any(np.isnan(COE)):
            continue

        lnabs = abs(float(LNpair))
        if lnabs < LN_pair_abs_threshold:
            continue

        ent_id += 1
        info = coupled[(za, zb)]

        A0, A1 = int(A[0]), int(A[-1])
        B0, B1 = int(B[0]), int(B[-1])
        zone1_n_bonds = max(len(A) - 1, 0)
        zone2_n_bonds = max(len(B) - 1, 0)

        z1_start_to_coe = (i_bond + 1) if i_bond >= 0 else -1
        z1_coe_to_end = (zone1_n_bonds - (i_bond + 1)) if i_bond >= 0 else -1
        z2_start_to_coe = (j_bond + 1) if j_bond >= 0 else -1
        z2_coe_to_end = (zone2_n_bonds - (j_bond + 1)) if j_bond >= 0 else -1

        chain1 = int(zone_to_molid.get(int(za), molid_np[A0]))
        chain2 = int(zone_to_molid.get(int(zb), molid_np[B0]))

        rows_for_csv.append([
            int(ent_id), int(chain1), int(chain2), int(za), int(zb),
            float(LNpair), float(COE[0]), float(COE[1]), float(COE[2]),
            float(info["min_dist"]), int(info["count_edges"]),
            int(bead_ids[A0]), int(bead_ids[A1]), int(zone1_n_bonds),
            int(bead_ids[B0]), int(bead_ids[B1]), int(zone2_n_bonds),
            int(i_bond), int(j_bond),
            int(z1_start_to_coe), int(z1_coe_to_end),
            int(z2_start_to_coe), int(z2_coe_to_end),
        ])

    bead_index = _compute_bead_index_within_chain(adj, molid_np)
    ent_degree = smooth_ln_abs if ENT_DEGREE_FIELD == "smooth_LN_abs" else local_ln_abs

    if debug_prints:
        print(f"Zones: {len(zone_to_nodes)} | candidate zone pairs: {len(coupled)} | accepted COEs: {ent_id}")
        print("=== SIMPLE ENTANGLEMENT ANALYSIS COMPLETE ===\n")

    return {
        "nChains": len(np.unique(molid_np)),
        "nEntanglements": ent_id,
        "entanglement_rows": rows_for_csv,
        "bead_index": bead_index,
        "molid": molid_np,
        "smooth_ln_abs": ent_degree,
        "zone_members": {int(z): [int(bead_ids[i]) for i in nodes] for z, nodes in zone_to_nodes.items()},
        "zone_members_idx": {int(z): [int(i) for i in nodes] for z, nodes in zone_to_nodes.items()},
    }


# Helpers
def _empty_result(molid_np, adj, smooth_ln_abs, bead_ids):
    return {
        "nChains": len(np.unique(molid_np)),
        "nEntanglements": 0,
        "entanglement_rows": [],
        "bead_index": _compute_bead_index_within_chain(adj, molid_np),
        "molid": molid_np,
        "smooth_ln_abs": smooth_ln_abs,
        "zone_members": {},
        "zone_members_idx": {},
    }


def _pbc_dist(a, b, box_lengths):
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    for dim in range(3):
        L = box_lengths[dim]
        if L > 0:
            delta[dim] -= L * np.round(delta[dim] / L)
    return float(np.linalg.norm(delta))


def _get_molid_from_bonds(N, bond_topo_idx):
    parent = np.arange(N, dtype=np.int64)
    rank = np.zeros(N, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in bond_topo_idx:
        union(a, b)

    roots = np.array([find(i) for i in range(N)], dtype=np.int64)
    _, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int64) + 1


def _build_adjacency(N, bond_topo_idx):
    adj = [[] for _ in range(N)]
    for a, b in bond_topo_idx:
        a = int(a)
        b = int(b)
        if a < 0 or b < 0:
            continue
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _bfs_ball(adj, start, max_depth, passable):
    start = int(start)
    if not passable[start]:
        return [], {start: 0}

    q = deque([start])
    dist = {start: 0}

    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= max_depth:
            continue
        for w in adj[v]:
            w = int(w)
            if not passable[w]:
                continue
            if w not in dist:
                dist[w] = dv + 1
                q.append(w)

    return list(dist.keys()), dist


def _graph_distance_leq(adj, a, b, max_depth, passable):
    a = int(a)
    b = int(b)
    if a == b:
        return True
    if not passable[a] or not passable[b]:
        return False

    q = deque([a])
    dist = {a: 0}
    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= max_depth:
            continue
        for w in adj[v]:
            w = int(w)
            if not passable[w]:
                continue
            if w == b:
                return True
            if w not in dist:
                dist[w] = dv + 1
                q.append(w)
    return False


def _solid_angle_3d(a, b, c):
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]

    cxbc_x = by * cz - bz * cy
    cxbc_y = bz * cx - bx * cz
    cxbc_z = bx * cy - by * cx
    triple = ax * cxbc_x + ay * cxbc_y + az * cxbc_z

    na = np.sqrt(ax * ax + ay * ay + az * az)
    nb = np.sqrt(bx * bx + by * by + bz * bz)
    nc = np.sqrt(cx * cx + cy * cy + cz * cz)

    ab = ax * bx + ay * by + az * bz
    ca = cx * ax + cy * ay + cz * az
    bc = bx * cx + by * cy + bz * cz

    denom = na * nb * nc + ab * nc + ca * nb + bc * na
    return np.arctan2(triple, denom)


def _linking_number_and_weight(chain1_xyz, chain2_xyz):
    if chain1_xyz.shape[0] < 2 or chain2_xyz.shape[0] < 2:
        return 0.0, 0.0

    rI, rIp1 = chain1_xyz[:-1], chain1_xyz[1:]
    rJ, rJp1 = chain2_xyz[:-1], chain2_xyz[1:]
    N1, N2 = rI.shape[0], rJ.shape[0]

    RI = rI.reshape((N1, 1, 3))
    RIP1 = rIp1.reshape((N1, 1, 3))
    RJ = rJ.reshape((1, N2, 3))
    RJP1 = rJp1.reshape((1, N2, 3))

    k = RI - RJ
    l = RIP1 - RJ
    m = RIP1 - RJP1
    n = RI - RJP1

    Omega1 = _solid_angle_3d(k, l, m)
    Omega2 = _solid_angle_3d(m, n, k)
    OmegaIJ = 2.0 * (Omega1 + Omega2)

    denom = 4.0 * np.pi
    LN = float(np.sum(OmegaIJ) / denom)
    W = float(np.sum(np.abs(OmegaIJ)) / denom)
    return LN, W


def _linking_number_weight_and_coe(chain1_xyz, chain2_xyz, box_bounds):
    if chain1_xyz.shape[0] < 2 or chain2_xyz.shape[0] < 2:
        return 0.0, 0.0, np.array([np.nan, np.nan, np.nan]), -1, -1, 0, 0

    chain1_crosses = _chain_crosses_pbc(chain1_xyz, box_bounds)
    chain2_crosses = _chain_crosses_pbc(chain2_xyz, box_bounds)
    need_unwrap = chain1_crosses or chain2_crosses

    chain1_use = _unwrap_chain(chain1_xyz, box_bounds) if chain1_crosses else chain1_xyz
    chain2_use = _unwrap_chain(chain2_xyz, box_bounds) if chain2_crosses else chain2_xyz

    if need_unwrap and box_bounds is not None:
        ref = chain1_use[0]
        chain2_use = chain2_use.copy()
        for dim in range(3):
            L = box_bounds[dim, 1] - box_bounds[dim, 0]
            if L > 0:
                delta = chain2_use[0, dim] - ref[dim]
                shift = -L * np.round(delta / L)
                chain2_use[:, dim] += shift

    rI, rIp1 = chain1_use[:-1], chain1_use[1:]
    rJ, rJp1 = chain2_use[:-1], chain2_use[1:]
    N1, N2 = rI.shape[0], rJ.shape[0]

    RI = rI.reshape((N1, 1, 3))
    RIP1 = rIp1.reshape((N1, 1, 3))
    RJ = rJ.reshape((1, N2, 3))
    RJP1 = rJp1.reshape((1, N2, 3))

    k = RI - RJ
    l = RIP1 - RJ
    m = RIP1 - RJP1
    n = RI - RJP1

    Omega1 = _solid_angle_3d(k, l, m)
    Omega2 = _solid_angle_3d(m, n, k)
    OmegaIJ = 2.0 * (Omega1 + Omega2)

    denom = 4.0 * np.pi
    LN = float(np.sum(OmegaIJ) / denom)

    w = np.abs(OmegaIJ)
    W = float(np.sum(w) / denom)
    wsum = float(np.sum(w))

    if wsum <= 0.0:
        return LN, W, np.array([np.nan, np.nan, np.nan]), -1, -1, N1, N2

    midI = 0.5 * (rI + rIp1)
    midJ = 0.5 * (rJ + rJp1)

    coe = np.zeros(3, dtype=float)
    for a in range(N1):
        wa = w[a, :]
        s_wa = float(np.sum(wa))
        if s_wa == 0.0:
            continue
        termJ = (wa.reshape(-1, 1) * midJ).sum(axis=0)
        coe += 0.5 * (midI[a] * s_wa + termJ)
    coe /= wsum

    if need_unwrap and box_bounds is not None:
        for dim in range(3):
            L = box_bounds[dim, 1] - box_bounds[dim, 0]
            if L > 0:
                lo = box_bounds[dim, 0]
                coe[dim] = lo + ((coe[dim] - lo) % L)

    wI = np.sum(w, axis=1)
    wJ = np.sum(w, axis=0)
    i_bond = int(np.argmax(wI)) if N1 > 0 else -1
    j_bond = int(np.argmax(wJ)) if N2 > 0 else -1

    return LN, W, coe, i_bond, j_bond, N1, N2


def _chain_crosses_pbc(coords, box_bounds):
    if coords.shape[0] <= 1 or box_bounds is None:
        return False
    for i in range(1, len(coords)):
        for dim in range(3):
            L = box_bounds[dim, 1] - box_bounds[dim, 0]
            if L > 0:
                if abs(coords[i, dim] - coords[i - 1, dim]) > L / 2:
                    return True
    return False


def _unwrap_chain(coords, box_bounds):
    if coords.shape[0] <= 1 or box_bounds is None:
        return coords.copy()
    unwrapped = coords.copy()
    for i in range(1, len(unwrapped)):
        for dim in range(3):
            L = box_bounds[dim, 1] - box_bounds[dim, 0]
            if L > 0:
                delta = unwrapped[i, dim] - unwrapped[i - 1, dim]
                if delta > L / 2:
                    unwrapped[i, dim] -= L
                elif delta < -L / 2:
                    unwrapped[i, dim] += L
    return unwrapped


def _connected_components_zones(adj, nodes_mask, molid_np):
    N = len(nodes_mask)
    zone_id = np.zeros(N, dtype=np.int64)
    zone_to_nodes = {}
    zone_to_molid = {}

    zid = 0
    visited = np.zeros(N, dtype=bool)

    for s in range(N):
        if visited[s] or not nodes_mask[s]:
            continue
        zid += 1
        mol = int(molid_np[s])
        q = deque([s])
        visited[s] = True
        zone_id[s] = zid
        comp = [s]

        while q:
            v = q.popleft()
            for w in adj[v]:
                w = int(w)
                if visited[w] or not nodes_mask[w]:
                    continue
                if int(molid_np[w]) != mol:
                    continue
                visited[w] = True
                zone_id[w] = zid
                q.append(w)
                comp.append(w)

        zone_to_nodes[zid] = comp
        zone_to_molid[zid] = mol

    return zone_id, zone_to_nodes, zone_to_molid


def _order_zone_nodes_as_path(zone_nodes, adj, molid_np):
    if len(zone_nodes) <= 1:
        return list(zone_nodes)

    zset = set(int(x) for x in zone_nodes)
    mol = int(molid_np[zone_nodes[0]])

    def zdeg(v):
        return sum((int(w) in zset) and (int(molid_np[w]) == mol) for w in adj[int(v)])

    degs = {int(v): zdeg(v) for v in zset}
    ends = [v for v, d in degs.items() if d == 1]

    if ends:
        start = min(ends)
        ordered = [start]
        prev = -1
        cur = start
        visited = {start}
        while True:
            nbrs = [int(w) for w in adj[cur] if (int(w) in zset) and (int(molid_np[w]) == mol)]
            cand = [w for w in nbrs if w != prev and w not in visited]
            if not cand:
                break
            nxt = min(cand)
            ordered.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if len(ordered) == len(zset):
            return ordered

    start = min(zset)
    q = deque([start])
    seen = {start}
    ordered = [start]
    while q:
        v = q.popleft()
        nbrs = sorted([int(w) for w in adj[v] if (int(w) in zset) and (int(molid_np[w]) == mol)])
        for w in nbrs:
            if w in seen:
                continue
            seen.add(w)
            q.append(w)
            ordered.append(w)
    return ordered


def _compute_bead_index_within_chain(adj, molid_np):
    N = len(molid_np)
    bead_idx = -np.ones(N, dtype=np.int64)

    for mol in np.unique(molid_np):
        mol = int(mol)
        nodes = np.where(molid_np == mol)[0]
        if nodes.size == 0:
            continue
        nset = set(int(x) for x in nodes)
        deg = {int(v): sum(int(w) in nset for w in adj[int(v)]) for v in nodes}
        ends = [int(v) for v in nodes if deg[int(v)] == 1]
        start = int(np.min(ends)) if ends else int(np.min(nodes))

        ordered = [start]
        visited = {start}
        prev = -1
        cur = start
        while True:
            nbrs = [int(w) for w in adj[cur] if int(w) in nset]
            cand = [w for w in nbrs if w != prev and w not in visited]
            if not cand:
                break
            nxt = min(cand)
            ordered.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        if len(visited) < len(nset):
            ordered.extend(sorted([v for v in nset if v not in visited]))

        for k, v in enumerate(ordered):
            bead_idx[int(v)] = int(k)

    return bead_idx
