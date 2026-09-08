# type: ignore
"""
Simple zone-anchored primitive-path construction.

Kept logic only:
  5. COE spatial merging.
  6. One anchor per zone, not per molecule.
  7. merge_same_anchor=False by default and honored.
  8. Corrected N using ceil(COE-to-anchor correction distance).
  9. Path-based duplicate handling.

Removed on purpose:
  - auxiliary/supporting-zone special logic
  - forced same-anchor merge
  - old MATLAB one-anchor-per-molecule behavior
  - endpoint-pair duplicate collapse
  - temporal consensus logic

Output:
    vertices: [id, vtype, x, y, z, mol, LN]
    edges:    [edge_i, etype, i, j, rx, ry, rz, mol, N]

Edge convention:
    [rx, ry, rz] = x_i - x_j_image
"""

from collections import deque
import numpy as np


def _parse_header_columns(header_line):
    header_str = str(header_line)
    for prefix in ("ITEM: ATOMS", "ITEM: ENTRIES"):
        if prefix in header_str:
            header_str = header_str.replace(prefix, "")
            break
    return {c: i for i, c in enumerate(header_str.strip().split())}


def _union_find(n):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri == rj:
            return False
        parent[rj] = ri
        return True

    return parent, find, union


def build_primitive_path_vertices_edges(
    atoms_data,
    atoms_header,
    bonds_data,
    ent_rows,
    box_bounds,
    num_atoms,
    merge_cutoff=1.0,
    merge_same_anchor=False,
    zone_members=None,
    zone_members_idx=None,
):
    """
    Build DNM vertices and primitive-path edges.

    Important behavior:
      - COEs merge only spatially using merge_cutoff, unless the caller explicitly
        sets merge_same_anchor=True.
      - Anchor contributions are keyed by zone_id, not mol_id.
      - Edges are built by sorting stops along each molecule contour and connecting
        consecutive stops.
      - Duplicate handling uses the actual walked bead path, not just endpoint IDs.
    """

    col = _parse_header_columns(atoms_header)

    aid = atoms_data[:, col["id"]].astype(int)
    atype = atoms_data[:, col["type"]].astype(int)
    ax = atoms_data[:, col["x"]].astype(float)
    ay = atoms_data[:, col["y"]].astype(float)
    az = atoms_data[:, col["z"]].astype(float)
    amol = atoms_data[:, col["mol"]].astype(int) if "mol" in col else np.zeros(len(aid), dtype=int)

    id_to_row = {int(aid[i]): i for i in range(len(aid))}

    box_lengths = np.array([
        box_bounds[0, 1] - box_bounds[0, 0],
        box_bounds[1, 1] - box_bounds[1, 0],
        box_bounds[2, 1] - box_bounds[2, 0],
    ], dtype=float)

    def min_image_delta(vec):
        out = np.asarray(vec, dtype=float).copy()
        for d in range(3):
            L = box_lengths[d]
            if L > 0:
                out[d] -= L * np.round(out[d] / L)
        return out

    def pbc_dist(p, q):
        return float(np.linalg.norm(min_image_delta(np.asarray(q, dtype=float) - np.asarray(p, dtype=float))))

    def wrap_point(p):
        p = np.asarray(p, dtype=float).copy()
        for d in range(3):
            L = box_lengths[d]
            if L > 0:
                lo = box_bounds[d, 0]
                p[d] = lo + ((p[d] - lo) % L)
        return p

    def unwrap_points_near_reference(points, ref):
        pts = np.asarray(points, dtype=float).copy()
        ref = np.asarray(ref, dtype=float)
        for ii in range(len(pts)):
            pts[ii] = ref + min_image_delta(pts[ii] - ref)
        return pts

    def atom_pos(atom_id):
        r = id_to_row[int(atom_id)]
        return np.array([ax[r], ay[r], az[r]], dtype=float)

    # Bond graph and stored LAMMPS bond vectors
    # bonds_data expected columns from wrapper:
    # [index, btype?, atom1, atom2, rx, ry, rz, dist, force]
    bond_a = bonds_data[:, 2].astype(int)
    bond_b = bonds_data[:, 3].astype(int)
    bond_rx = bonds_data[:, 4].astype(float)
    bond_ry = bonds_data[:, 5].astype(float)
    bond_rz = bonds_data[:, 6].astype(float)

    max_id = int(np.max(aid)) if len(aid) else 0
    neighbour_map = [[] for _ in range(max_id + 1)]

    for bi in range(len(bonds_data)):
        a1 = int(bond_a[bi])
        a2 = int(bond_b[bi])
        if 0 <= a1 <= max_id:
            neighbour_map[a1].append((a2, bi))
        if 0 <= a2 <= max_id:
            neighbour_map[a2].append((a1, bi))

    def same_mol(atom_id, mol_id):
        atom_id = int(atom_id)
        return atom_id in id_to_row and int(amol[id_to_row[atom_id]]) == int(mol_id)

    def step_vec(current_atom, next_atom, bond_index):
        """Return x_current - x_next using stored bond vector."""
        current_atom = int(current_atom)
        next_atom = int(next_atom)
        bond_index = int(bond_index)

        if int(bond_a[bond_index]) == current_atom and int(bond_b[bond_index]) == next_atom:
            return np.array([bond_rx[bond_index], bond_ry[bond_index], bond_rz[bond_index]], dtype=float)
        if int(bond_b[bond_index]) == current_atom and int(bond_a[bond_index]) == next_atom:
            return -np.array([bond_rx[bond_index], bond_ry[bond_index], bond_rz[bond_index]], dtype=float)

        return min_image_delta(atom_pos(current_atom) - atom_pos(next_atom))

    def shortest_path_same_mol(start, end, mol_id):
        start = int(start)
        end = int(end)
        mol_id = int(mol_id)

        if start not in id_to_row or end not in id_to_row:
            return []

        q = deque([start])
        parent = {start: None}

        while q:
            cur = q.popleft()
            if cur == end:
                break
            if cur < 0 or cur > max_id:
                continue
            for nb, _bi in neighbour_map[cur]:
                nb = int(nb)
                if nb in parent:
                    continue
                if not same_mol(nb, mol_id):
                    continue
                parent[nb] = cur
                q.append(nb)

        if end not in parent:
            return []

        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path
   
    # Zone bead extraction
    zone_members = zone_members or {}

    def get_zone_beads(zone):
        zid = int(zone["zone_id"])

        # Main path: actual zone members from processEntanglement.
        if zid in zone_members:
            beads = [int(b) for b in zone_members[zid] if int(b) in id_to_row]
            if beads:
                mols = sorted(set(int(amol[id_to_row[b]]) for b in beads))
                if len(mols) > 1:
                    counts = {m: sum(int(amol[id_to_row[b]]) == m for b in beads) for m in mols}
                    mol = max(counts, key=counts.get)
                    beads = [b for b in beads if int(amol[id_to_row[b]]) == mol]
                else:
                    mol = mols[0]
                return beads, int(mol), "zone_members"

        # Fallback: explicit start-end path stored in ent_rows.
        start_pid = int(zone["start_pid"])
        end_pid = int(zone["end_pid"])
        if start_pid in id_to_row and end_pid in id_to_row:
            mol_start = int(amol[id_to_row[start_pid]])
            mol_end = int(amol[id_to_row[end_pid]])
            if mol_start == mol_end:
                beads = shortest_path_same_mol(start_pid, end_pid, mol_start)
                if beads:
                    return beads, int(mol_start), "start_end_path"

        return [], None, "missing"

    def nearest_bead_inside_zone(coe_pos, zone):
        beads, mol_id, source = get_zone_beads(zone)
        if not beads or mol_id is None:
            return None

        type2 = [b for b in beads if int(atype[id_to_row[b]]) == 2]
        candidates = type2 if type2 else beads

        d = np.array([pbc_dist(coe_pos, atom_pos(b)) for b in candidates], dtype=float)
        best_i = int(np.argmin(d))
        best_bead = int(candidates[best_i])

        return {
            "mol": int(mol_id),
            "anchor_bead": best_bead,
            "dist": float(d[best_i]),
            "zone_beads": beads,
            "zone_id": int(zone["zone_id"]),
            "source": source,
            "start_pid": int(zone["start_pid"]),
            "end_pid": int(zone["end_pid"]),
        }

    # Parse raw COEs
    raw_coes = []
    for row in ent_rows:
        ent_id = int(row[0])
        ln_val = float(row[5])
        coe = np.array([float(row[6]), float(row[7]), float(row[8])], dtype=float)

        z1 = {
            "zone_id": int(row[3]),
            "start_pid": int(row[11]),
            "end_pid": int(row[12]),
            "n_bonds": int(row[13]),
        }
        z2 = {
            "zone_id": int(row[4]),
            "start_pid": int(row[14]),
            "end_pid": int(row[15]),
            "n_bonds": int(row[16]),
        }

        raw_coes.append({
            "raw_index": len(raw_coes),
            "old_ent_ids": [ent_id],
            "pos": coe,
            "ln_values": [ln_val],
            "zones": [z1, z2],
        })

    n_raw = len(raw_coes)
    parent, parent_find, parent_union = _union_find(n_raw)

    # Spatial COE merge only 
    for i in range(n_raw):
        for j in range(i + 1, n_raw):
            if pbc_dist(raw_coes[i]["pos"], raw_coes[j]["pos"]) <= merge_cutoff:
                parent_union(i, j)

    def build_merged_from_parent():
        groups = {}
        for i in range(n_raw):
            groups.setdefault(parent_find(i), []).append(i)

        merged_local = []
        for new_id, (_root, idxs) in enumerate(sorted(groups.items()), start=1):
            ref = raw_coes[idxs[0]]["pos"]
            pts = np.array([raw_coes[i]["pos"] for i in idxs], dtype=float)
            pts_unwrapped = unwrap_points_near_reference(pts, ref)
            pos_mean = wrap_point(np.mean(pts_unwrapped, axis=0))

            old_ids, ln_values, zones = [], [], []
            for i in idxs:
                old_ids.extend(raw_coes[i]["old_ent_ids"])
                ln_values.extend(raw_coes[i]["ln_values"])
                zones.extend(raw_coes[i]["zones"])

            merged_local.append({
                "new_id": int(new_id),
                "vid": int(num_atoms + new_id),
                "pos": pos_mean,
                "old_ent_ids": old_ids,
                "ln": float(np.sum(ln_values)) if ln_values else 0.0,
                "raw_indices": list(idxs),
                "zones": zones,
                "zone_contributions": [],
                "chains": set(),
            })
        return merged_local

    def assign_zone_contributions(merged_list):
        """Keep one anchor per zone_id, not one anchor per molecule."""
        for c in merged_list:
            by_zone = {}
            for zone in c["zones"]:
                anch = nearest_bead_inside_zone(c["pos"], zone)
                if anch is None:
                    print(f"    [WARN] COE {c['vid']}: could not assign anchor for zone {zone}")
                    continue

                zid = int(anch["zone_id"])
                if zid not in by_zone or anch["dist"] < by_zone[zid]["dist"]:
                    by_zone[zid] = anch

            c["zone_contributions"] = []
            c["chains"] = set()
            for _zid, anch in sorted(by_zone.items()):
                item = dict(anch)
                item["vid"] = int(c["vid"])
                item["pos"] = np.asarray(c["pos"], dtype=float)
                item["old_ent_ids"] = list(c["old_ent_ids"])
                item["ln"] = float(c["ln"])
                c["zone_contributions"].append(item)
                c["chains"].add(int(item["mol"]))
        return merged_list

    # Optional same-anchor merge is available but OFF by default.
    if merge_same_anchor:
        changed = True
        pass_id = 0
        while changed:
            pass_id += 1
            changed = False
            temp = assign_zone_contributions(build_merged_from_parent())
            anchor_map = {}
            for c in temp:
                rep_raw = int(c["raw_indices"][0])
                for anch in c["zone_contributions"]:
                    key = (int(anch["mol"]), int(anch["anchor_bead"]))
                    if key not in anchor_map:
                        anchor_map[key] = rep_raw
                    else:
                        if parent_union(anchor_map[key], rep_raw):
                            changed = True
                            print(f"    [MERGE] same-anchor mol={key[0]}, bead={key[1]}")
            if pass_id > n_raw + 2:
                print("    [WARN] same-anchor merge loop exceeded safety limit")
                break

    merged = assign_zone_contributions(build_merged_from_parent())

    print(
        f"  Raw COEs: {n_raw} | merged COEs: {len(merged)} "
        f"using spatial cutoff {merge_cutoff:g} | merge_same_anchor={merge_same_anchor}"
    )
  
    # Vertices: type-1 beads + type-2 merged COEs 
    built_ids = aid[atype == 1]

    verts_beads = []
    for sid in built_ids:
        r = id_to_row[int(sid)]
        verts_beads.append([
            int(sid), 1,
            float(ax[r]), float(ay[r]), float(az[r]),
            int(amol[r]), 0.0,
        ])

    coe_mol_val = int(np.max(amol)) + 1 if len(amol) else 1
    verts_coe = []
    for c in merged:
        x, y, z = c["pos"]
        verts_coe.append([
            int(c["vid"]), 2,
            float(x), float(y), float(z),
            int(coe_mol_val), float(c["ln"]),
        ])

    vertices = np.array(sorted(verts_beads + verts_coe, key=lambda v: v[0])) if (verts_beads or verts_coe) else np.empty((0, 7))
 
    # Contour-sorted edge construction 
    def event_pos(event):
        if event["kind"] == "atom":
            return atom_pos(event["vid"])
        return np.asarray(event["pos"], dtype=float)

    def bond_index_between(a, b):
        a = int(a)
        b = int(b)
        if 0 <= a <= max_id:
            for nb, bi in neighbour_map[a]:
                if int(nb) == b:
                    return int(bi)
        return None

    def order_molecule_beads(mol_id):
        mol_id = int(mol_id)
        nodes = [int(a) for a in aid[amol == mol_id].tolist()]
        if not nodes:
            return []

        nset = set(nodes)
        deg = {v: sum(1 for nb, _bi in neighbour_map[v] if int(nb) in nset) for v in nodes}
        type1_nodes = sorted([int(a) for a in aid[(amol == mol_id) & (atype == 1)].tolist()])
        ends = sorted([v for v in nodes if deg.get(v, 0) == 1])

        start = type1_nodes[0] if type1_nodes else (ends[0] if ends else min(nodes))

        ordered = [start]
        visited = {start}
        prev = None
        cur = start

        while True:
            nbrs = [int(nb) for nb, _bi in neighbour_map[cur] if int(nb) in nset]
            cand = [nb for nb in nbrs if nb != prev and nb not in visited]
            if not cand:
                break
            nxt = min(cand)
            ordered.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        if len(visited) < len(nset):
            remaining = sorted([v for v in nodes if v not in visited])
            print(f"    [WARN] Mol {mol_id}: contour order visited {len(visited)}/{len(nset)}; appending {len(remaining)} beads")
            ordered.extend(remaining)

        return ordered

    def path_between_ordered(order, index_map, a, b):
        a = int(a)
        b = int(b)
        ia = int(index_map[a])
        ib = int(index_map[b])

        if ia == ib:
            return [a], []

        lo, hi = min(ia, ib), max(ia, ib)
        atoms_forward = [int(x) for x in order[lo:hi + 1]]
        path_atoms = list(reversed(atoms_forward)) if ia > ib else atoms_forward

        path_bonds = []
        for u, v in zip(path_atoms[:-1], path_atoms[1:]):
            bi = bond_index_between(u, v)
            if bi is None:
                print(f"    [WARN] Missing bond between contour atoms {u}-{v}; skipping interval")
                return [], []
            path_bonds.append(int(bi))

        return path_atoms, path_bonds

    def make_edge_vector_from_path(start_event, stop_event, path_atoms, path_bonds):
        start_bead = int(path_atoms[0])
        stop_bead = int(path_atoms[-1])

        vec = min_image_delta(event_pos(start_event) - atom_pos(start_bead))
        for idx, bi in enumerate(path_bonds):
            cur = int(path_atoms[idx])
            nxt = int(path_atoms[idx + 1])
            vec += step_vec(cur, nxt, int(bi))
        vec += min_image_delta(atom_pos(stop_bead) - event_pos(stop_event))
        return vec

    def primitive_path_N(start_event, stop_event, path_bonds):
        N_eff = float(len(path_bonds))

        if start_event["kind"] == "coe":
            a = int(start_event["anchor_bead"])
            d = float(np.linalg.norm(min_image_delta(event_pos(start_event) - atom_pos(a))))
            N_eff += np.ceil(max(d, 0.0))

        if stop_event["kind"] == "coe":
            a = int(stop_event["anchor_bead"])
            d = float(np.linalg.norm(min_image_delta(atom_pos(a) - event_pos(stop_event))))
            N_eff += np.ceil(max(d, 0.0))

        return max(int(np.ceil(N_eff)), 1)

    # Group COE-anchor events by molecule.
    coes_by_mol = {}
    for c in merged:
        for anch in c["zone_contributions"]:
            coes_by_mol.setdefault(int(anch["mol"]), []).append({
                "kind": "coe",
                "vid": int(c["vid"]),
                "pos": np.asarray(c["pos"], dtype=float),
                "anchor_bead": int(anch["anchor_bead"]),
                "mol": int(anch["mol"]),
                "zone_id": int(anch["zone_id"]),
                "old_ent_ids": list(c["old_ent_ids"]),
                "source": anch.get("source", "unknown"),
                "anchor_dist": float(anch.get("dist", np.nan)),
            })

    candidate_edges = []

    for mol_id in sorted(set(int(m) for m in amol.tolist())):
        order = order_molecule_beads(mol_id)
        if len(order) < 2:
            continue
        index_map = {int(a): idx for idx, a in enumerate(order)}

        events = []

        # Type-1 ends as stops.
        for sid in sorted(aid[(atype == 1) & (amol == int(mol_id))].astype(int).tolist()):
            if int(sid) not in index_map:
                continue
            events.append({
                "kind": "atom",
                "vid": int(sid),
                "pos": atom_pos(sid),
                "anchor_bead": int(sid),
                "mol": int(mol_id),
                "zone_id": -1,
                "contour_index": float(index_map[int(sid)]),
                "sort_id": int(sid),
            })

        # COE-zone anchors as stops.
        for ce in coes_by_mol.get(int(mol_id), []):
            anchor = int(ce["anchor_bead"])
            if anchor not in index_map:
                print(f"    [WARN] Mol {mol_id}: COE {ce['vid']} anchor {anchor} not in contour order; skipping")
                continue
            ev = dict(ce)
            ev["contour_index"] = float(index_map[anchor])
            ev["sort_id"] = int(ce["vid"])
            events.append(ev)

        if len(events) < 2:
            continue

        def event_sort_key(ev):
            kind_rank = 0 if ev["kind"] == "atom" else 1
            return (float(ev["contour_index"]), kind_rank, int(ev.get("sort_id", ev["vid"])))

        events_sorted = sorted(events, key=event_sort_key)

        # Connect consecutive stops.
        for ev0, ev1 in zip(events_sorted[:-1], events_sorted[1:]):
            if int(ev0["vid"]) == int(ev1["vid"]):
                # Same COE appears twice on same chain. It is an internal loop to same DNM node.
                continue

            path_atoms, path_bonds = path_between_ordered(
                order, index_map,
                int(ev0["anchor_bead"]),
                int(ev1["anchor_bead"]),
            )
            if not path_atoms:
                continue

            vec = make_edge_vector_from_path(ev0, ev1, path_atoms, path_bonds)
            Nval = primitive_path_N(ev0, ev1, path_bonds)

            candidate_edges.append({
                "i": int(ev0["vid"]),
                "j": int(ev1["vid"]),
                "rx": float(vec[0]),
                "ry": float(vec[1]),
                "rz": float(vec[2]),
                "mol": int(mol_id),
                "N": int(Nval),
                "path": [int(x) for x in path_atoms],
                "source": "contour-sorted",
            })

    # Path-based duplicate handling
    unique = {}
    duplicate_count = 0

    def canonical_path_key(e):
        path = tuple(int(x) for x in e.get("path", []))
        rpath = tuple(reversed(path))
        return (int(e["mol"]), path if path <= rpath else rpath)

    for e in candidate_edges:
        if int(e["i"]) == int(e["j"]):
            duplicate_count += 1
            continue

        key = canonical_path_key(e)
        if key not in unique:
            unique[key] = e
            continue

        duplicate_count += 1
        old = unique[key]

        # Same walked interval. Keep larger N to avoid undercounting; if tied, keep shorter vector.
        keep_new = False
        if int(e["N"]) > int(old["N"]):
            keep_new = True
        elif int(e["N"]) == int(old["N"]):
            r_new = float(np.linalg.norm([e["rx"], e["ry"], e["rz"]]))
            r_old = float(np.linalg.norm([old["rx"], old["ry"], old["rz"]]))
            if r_new < r_old:
                keep_new = True

        if keep_new:
            unique[key] = e

    if duplicate_count:
        print(f"  [DEDUP] Removed {duplicate_count} exact self/duplicate contour intervals")

    edges_list = []
    for edge_counter, e in enumerate(unique.values(), start=1):
        rnorm = float(np.linalg.norm([e["rx"], e["ry"], e["rz"]]))
        rn = rnorm / max(float(e["N"]), 1.0)
        edges_list.append([
            edge_counter, 1,
            int(e["i"]), int(e["j"]),
            float(e["rx"]), float(e["ry"]), float(e["rz"]),
            int(e["mol"]), int(e["N"]),
        ])
        if rn >= 1.0:
            print(
                f"      [BAD EDGE r/N>=1] edge={edge_counter}, "
                f"{int(e['i'])}->{int(e['j'])}, mol={int(e['mol'])}, "
                f"N={int(e['N'])}, r={rnorm:.6f}, r/N={rn:.6f}, path={e.get('path', [])}"
            )

    edges = np.array(edges_list) if edges_list else np.empty((0, 9))

    # Minimal diagnostic.
    if len(edges) > 0:
        for c in merged:
            vid = int(c["vid"])
            deg = int(np.sum((edges[:, 2].astype(int) == vid) | (edges[:, 3].astype(int) == vid)))
            if deg == 0:
                print(f"    [DIAG] COE {vid}: degree 0; old_ent_ids={c['old_ent_ids']}")

    return vertices, edges


# Compatibility alias

def build_vertices_and_edges(
    atoms_data,
    atoms_header,
    bonds_data,
    ent_rows,
    box_bounds,
    num_atoms,
    merge_cutoff=1.0,
    merge_same_anchor=False,
    zone_members=None,
    zone_members_idx=None,
):
    return build_primitive_path_vertices_edges(
        atoms_data,
        atoms_header,
        bonds_data,
        ent_rows,
        box_bounds,
        num_atoms,
        merge_cutoff=merge_cutoff,
        merge_same_anchor=merge_same_anchor,
        zone_members=zone_members,
        zone_members_idx=zone_members_idx,
    )
