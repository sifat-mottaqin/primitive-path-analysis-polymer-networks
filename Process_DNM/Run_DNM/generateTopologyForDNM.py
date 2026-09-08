# type: ignore

import os
import math
import numpy as np


# Dump-table reader
def read_dump_table(filename, count_label, data_label):
    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    nitems = None
    box = []
    columns = None
    rows = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == count_label:
            nitems = int(lines[i + 1].strip())
            i += 2
            continue

        if line.startswith("ITEM: BOX BOUNDS"):
            box = []
            for k in range(3):
                lo, hi = map(float, lines[i + 1 + k].split()[:2])
                box.append((lo, hi))
            i += 4
            continue

        if line.startswith(data_label):
            parts = line.split()
            columns = parts[2:]
            i += 1

            if nitems is None:
                raise ValueError(
                    f"Could not find {count_label} before {data_label} in {filename}"
                )

            for _ in range(nitems):
                vals = lines[i].split()
                if len(vals) != len(columns):
                    raise ValueError(
                        f"Column mismatch in {filename}\n"
                        f"Expected {len(columns)} columns: {columns}\n"
                        f"Got {len(vals)} values: {vals}"
                    )
                row = {col: val for col, val in zip(columns, vals)}
                rows.append(row)
                i += 1
            continue

        i += 1

    if nitems is None:
        raise ValueError(f"Could not find {count_label} in {filename}")
    if len(box) != 3:
        raise ValueError(f"Could not read full box bounds from {filename}")
    if columns is None:
        raise ValueError(f"Could not read table data from {filename}")

    return nitems, box, columns, rows


def fmt_num(x):
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.10g}"


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# Geometry helpers
def distance3(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def edge_vector_length_from_row(row):
    """
    Correct diagnostic distance for target_dump/edges_*.dump.

    The primitive-path edge dump already stores the PBC-corrected walked
    edge vector as rx, ry, rz. Do NOT recompute r from wrapped vertex
    coordinates, because that can create fake huge r/N near periodic boundaries.
    """
    rx = float(row["rx"])
    ry = float(row["ry"])
    rz = float(row["rz"])
    return math.sqrt(rx * rx + ry * ry + rz * rz)


# Vertex merging
def merge_close_coes(vrows_sorted, cutoff):
    """
    Merge close vertices/COEs according to cutoff.

    NOTE:
    This function uses direct Euclidean distance of vertex coordinates, matching
    the earlier topology-generation behavior. For debugging, use cutoff=0.0
    if you do not want this second-stage topology merge.
    """
    old_ids = [int(r["i"]) for r in vrows_sorted]
    positions = {
        int(r["i"]): (float(r["x"]), float(r["y"]), float(r["z"]))
        for r in vrows_sorted
    }

    uf = UnionFind(old_ids)

    n = len(old_ids)
    if cutoff > 0:
        for i in range(n):
            id_i = old_ids[i]
            pi = positions[id_i]
            for j in range(i + 1, n):
                id_j = old_ids[j]
                pj = positions[id_j]
                if distance3(pi, pj) < cutoff:
                    uf.union(id_i, id_j)

    groups = {}
    for old_id in old_ids:
        root = uf.find(old_id)
        groups.setdefault(root, []).append(old_id)

    merged_nodes = []
    old_to_merged = {}

    for merged_id, members in enumerate(groups.values(), start=1):
        xs = [positions[m][0] for m in members]
        ys = [positions[m][1] for m in members]
        zs = [positions[m][2] for m in members]

        x_avg = sum(xs) / len(xs)
        y_avg = sum(ys) / len(ys)
        z_avg = sum(zs) / len(zs)

        first = next(r for r in vrows_sorted if int(r["i"]) == members[0])

        merged_nodes.append({
            "id": merged_id,
            "members": members,
            "mol": int(first["mol"]),
            "vtype": int(first["vtype"]),
            "x": x_avg,
            "y": y_avg,
            "z": z_avg,
        })

        for m in members:
            old_to_merged[m] = merged_id

    return merged_nodes, old_to_merged, groups


# Bond processing with merge modes
def _ceil_int(x):
    return int(math.ceil(float(x)))


def process_edges(erows_sorted, old_to_merged, bond_merge_mode="keep_all"):
    """
    Convert target_dump edges into topology bonds.

    bond_merge_mode options:
        keep_all
            Keep all non-self primitive-path edges.

        smallest_Np
            Collapse parallel bonds between the same two topology nodes and keep
            one bond with the smallest N.

        avg_Np
            Collapse parallel bonds between the same two topology nodes and keep
            one bond with N = ceil(mean(N)).

    Diagnostics use target_dump edge rx/ry/rz, not wrapped vertex-coordinate
    distances. This is the key correction.
    """
    mode = str(bond_merge_mode).strip()
    valid_modes = {"keep_all", "smallest_Np", "avg_Np"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid bond_merge_mode={bond_merge_mode!r}. "
            f"Use one of {sorted(valid_modes)}."
        )

    raw_nonself = 0
    dropped_self = 0

    # Store processed raw edges with correct diagnostic r from rx/ry/rz.
    processed = []
    for row in erows_sorted:
        a_old = int(row["i"])
        b_old = int(row["j"])
        Nval = int(float(row["N"]))
        Nval = max(Nval, 1)

        if a_old not in old_to_merged or b_old not in old_to_merged:
            raise ValueError(f"Bond uses missing atom id: {a_old}, {b_old}")

        a_new = old_to_merged[a_old]
        b_new = old_to_merged[b_old]

        # If vertex merging turns an edge into an internal bond, remove it.
        if a_new == b_new:
            dropped_self += 1
            continue

        raw_nonself += 1

        r_edge = edge_vector_length_from_row(row)
        rn_edge = r_edge / max(float(Nval), 1.0)

        processed.append({
            "a1": int(a_new),
            "a2": int(b_new),
            "N": int(Nval),
            "r_edge": float(r_edge),
            "rn_edge": float(rn_edge),
            "old_i": int(a_old),
            "old_j": int(b_old),
        })

    if mode == "keep_all":
        merged_edges = [{"a1": e["a1"], "a2": e["a2"], "N": e["N"]} for e in processed]
        collapsed_parallel = 0
        diag_rn_values = [e["rn_edge"] for e in processed]
        return merged_edges, raw_nonself, dropped_self, collapsed_parallel, diag_rn_values

    # Group parallel bonds by unordered topology node pair.
    groups = {}
    for e in processed:
        pair = tuple(sorted((int(e["a1"]), int(e["a2"]))))
        groups.setdefault(pair, []).append(e)

    merged_edges = []
    diag_rn_values = []
    collapsed_parallel = 0

    for pair, items in groups.items():
        if len(items) > 1:
            collapsed_parallel += len(items) - 1

        if mode == "smallest_Np":
            chosen = min(items, key=lambda x: x["N"])
            Nout = int(chosen["N"])
            # Diagnostic for the kept bond uses the kept edge vector r.
            diag_rn = float(chosen["r_edge"]) / max(float(Nout), 1.0)

        elif mode == "avg_Np":
            Nout = _ceil_int(np.mean([it["N"] for it in items]))
            # For a collapsed bond, there is no unique target_dump rx/ry/rz.
            # Diagnostic conservatively reports the worst original r divided by
            # the averaged N.
            r_worst = max(float(it["r_edge"]) for it in items)
            diag_rn = r_worst / max(float(Nout), 1.0)

        else:
            raise RuntimeError("Unexpected bond_merge_mode")

        merged_edges.append({
            "a1": int(pair[0]),
            "a2": int(pair[1]),
            "N": max(int(Nout), 1),
        })
        diag_rn_values.append(float(diag_rn))

    return merged_edges, raw_nonself, dropped_self, collapsed_parallel, diag_rn_values


# Main topology writer
def generateTopologyForDNM(
    fileTag,
    mainFolder,
    merge_cutoff=0.0,
    bond_merge_mode="keep_all",
    title="DNM Topology",
):
    """
    Reads:
        target_dump/vertices_<fileTag>.dump
        target_dump/edges_<fileTag>.dump

    Writes:
        Input/topology_<fileTag>_DNM.txt

    Important controls:
        merge_cutoff:
            topology-level vertex/COE merge cutoff. For debugging, use 0.0
            because COE merging is usually already done in processPrimitivePath.

        bond_merge_mode:
            "keep_all"     : keep all primitive-path edges
            "smallest_Np"  : collapse parallel bonds and keep smallest N
            "avg_Np"       : collapse parallel bonds and use ceil(mean(N))
    """
    target_dump_dir = os.path.join(mainFolder, "target_dump")
    inp_dir = os.path.join(mainFolder, "Input")
    os.makedirs(inp_dir, exist_ok=True)

    vertices_file = os.path.join(target_dump_dir, f"vertices_{fileTag}.dump")
    edges_file = os.path.join(target_dump_dir, f"edges_{fileTag}.dump")
    output_file = os.path.join(inp_dir, f"topology_{fileTag}_DNM.txt")

    if not os.path.isfile(vertices_file):
        raise RuntimeError(f"Missing vertices dump: {vertices_file}")
    if not os.path.isfile(edges_file):
        raise RuntimeError(f"Missing edges dump: {edges_file}")

    natoms_raw, vbox, vcols, vrows = read_dump_table(
        vertices_file,
        count_label="ITEM: NUMBER OF ATOMS",
        data_label="ITEM: ATOMS",
    )

    nbonds_raw, ebox, ecols, erows = read_dump_table(
        edges_file,
        count_label="ITEM: NUMBER OF ENTRIES",
        data_label="ITEM: ENTRIES",
    )

    required_vcols = {"i", "vtype", "x", "y", "z", "mol"}
    missing_v = required_vcols - set(vcols)
    if missing_v:
        raise ValueError(f"Vertices dump missing required columns: {sorted(missing_v)}")

    required_ecols = {"i", "j", "N", "rx", "ry", "rz"}
    missing_e = required_ecols - set(ecols)
    if missing_e:
        raise ValueError(
            f"Edges dump missing required columns: {sorted(missing_e)}. "
            "Need rx ry rz for correct PBC-safe r/N diagnostic."
        )

    vrows_sorted = sorted(vrows, key=lambda r: int(r["i"]))
    merged_nodes, old_to_merged, groups = merge_close_coes(vrows_sorted, merge_cutoff)

    atoms_out = []
    for node in merged_nodes:
        atoms_out.append({
            "id": int(node["id"]),
            "mol": int(node["mol"]),
            "vtype": int(node["vtype"]),
            "x": float(node["x"]),
            "y": float(node["y"]),
            "z": float(node["z"]),
        })

    if "edge_i" in ecols:
        erows_sorted = sorted(erows, key=lambda r: int(r["edge_i"]))
    else:
        erows_sorted = erows

    merged_edges, raw_nonself, dropped_self, collapsed_parallel, diag_rn_values = process_edges(
        erows_sorted,
        old_to_merged,
        bond_merge_mode=bond_merge_mode,
    )

    bond_coeffs_out = []
    bonds_out = []

    for bond_id, edge in enumerate(merged_edges, start=1):
        bond_coeffs_out.append({
            "bond_id": int(bond_id),
            "style_flag": 1,
            "N": max(int(edge["N"]), 1),
        })

        bonds_out.append({
            "bond_id": int(bond_id),
            "bond_type": int(bond_id),
            "a1": int(edge["a1"]),
            "a2": int(edge["a2"]),
        })

    natoms = len(atoms_out)
    nbonds = len(bonds_out)

    atom_types = max(a["vtype"] for a in atoms_out) if atoms_out else 0
    bond_types = nbonds
    nchains = len(set(a["mol"] for a in atoms_out)) if atoms_out else 0

    xlo, xhi = vbox[0]
    ylo, yhi = vbox[1]
    zlo, zhi = vbox[2]

    with open(output_file, "w") as f:
        f.write(f"{title} {nchains} chains {natoms} atoms\n\n")

        f.write(f"{natoms} atoms\n")
        f.write(f"{nbonds} bonds\n")
        f.write("0 angles\n")
        f.write("0 dihedrals\n")
        f.write("0 impropers\n\n")

        f.write(f"{atom_types} atom types\n")
        f.write(f"{bond_types} bond types\n\n")

        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for atype in range(1, atom_types + 1):
            f.write(f"{atype} 1\n")

        f.write("\nBond Coeffs\n\n")
        for bc in bond_coeffs_out:
            f.write(f"{bc['bond_id']} {bc['style_flag']} {bc['N']}\n")

        f.write("\nAtoms\n\n")
        for a in atoms_out:
            f.write(
                f"{a['id']} {a['mol']} {a['vtype']} "
                f"{fmt_num(a['x'])} {fmt_num(a['y'])} {fmt_num(a['z'])}\n"
            )

        f.write("\nVelocities\n\n")
        for a in atoms_out:
            f.write(f"{a['id']} 0 0 0\n")

        f.write("\nBonds\n\n")
        for b in bonds_out:
            f.write(f"{b['bond_id']} {b['bond_type']} {b['a1']} {b['a2']}\n")

    print(f"Topology written: {output_file}")
    print(f"Original atoms = {natoms_raw}, merged atoms = {natoms}")
    print(f"Original edges = {nbonds_raw}, final bonds = {nbonds}")
    print(f"Bond merge mode = {bond_merge_mode}")
    print(
        f"Bond processing: raw non-self={raw_nonself}, "
        f"dropped self={dropped_self}, collapsed parallel={collapsed_parallel}"
    )

    if diag_rn_values:
        arr = np.asarray(diag_rn_values, dtype=float)
        print(
            "Topology r/N diagnostic from edge rx,ry,rz: "
            f"max={np.max(arr):.4f}, "
            f"count >=0.95={int(np.sum(arr >= 0.95))}, "
            f"count >=1.0={int(np.sum(arr >= 1.0))}"
        )
    else:
        print("Topology r/N diagnostic from edge rx,ry,rz: no bonds")

    return output_file
