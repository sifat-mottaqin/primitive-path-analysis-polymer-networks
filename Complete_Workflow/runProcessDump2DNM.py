# type: ignore
"""
  1. Auto-detects folders: training_data/, train_dump/, target_dump/
  2. Loads atom/bond dump files from training_data/
  3. Runs zone-pair COE detection via processEntanglement
  4. Builds primitive paths via processPrimitivePath
  5. Writes:
       train_dump/train_atoms_<tag>.dump
       train_dump/train_bonds_<tag>.dump
       target_dump/vertices_<tag>.dump
       target_dump/edges_<tag>.dump
"""

import os
import sys
import glob
import csv
import json
import numpy as np

from dumpIO import (
    load_last_timestep_atoms,
    load_last_timestep_bonds,
    load_specific_timestep_atoms,
    load_specific_timestep_bonds,
    write_atoms_dump,
    write_bonds_dump,
    write_vertices_dump,
    write_edges_dump,
)

from processEntanglement import analyze_entanglements
from processPrimitivePath import build_primitive_path_vertices_edges


# USER CONTROLS
# This is the relaxed, undeformed state used to distill the DNM for
# tension, compression, and pure-shear cases.
TARGET_TIMESTEP = "undeformed"  # Last saved frame before mechanical loadin or set any other timestep at any stretch
MERGE_CUTOFF = 1.0              # spatial COE merge distance (1*b)
MERGE_SAME_ANCHOR = False       # requested: do not merge by same anchor, it was set for future exploration


def _detect_pairs(folder):
    atom_files = sorted(glob.glob(os.path.join(folder, "atoms_*.dump")))
    pairs = []
    for af in atom_files:
        base = os.path.basename(af)
        tag = base.replace("atoms_", "", 1).replace(".dump", "")
        bf = os.path.join(folder, f"bonds_{tag}.dump")
        if os.path.isfile(bf):
            pairs.append((af, bf, tag))
        else:
            print(f"[WARN] No matching bonds file for {base}, skipping.")
    return pairs


def _parse_header_columns(header_line):
    header_str = str(header_line)
    for prefix in ("ITEM: ATOMS", "ITEM: ENTRIES"):
        if prefix in header_str:
            header_str = header_str.replace(prefix, "")
            break
    return {c: i for i, c in enumerate(header_str.strip().split())}


def _prepare_bead_and_bond_arrays(atoms_data, atoms_header, bonds_data):
    col_idx = _parse_header_columns(atoms_header)

    needed = {"id", "type", "mol", "x", "y", "z"}
    missing = needed - set(col_idx)
    if missing:
        raise RuntimeError(f"Atoms header missing columns {missing}. Header: {atoms_header}")

    ids = atoms_data[:, col_idx["id"]].astype(int)
    types = atoms_data[:, col_idx["type"]].astype(int)
    x = atoms_data[:, col_idx["x"]]
    y = atoms_data[:, col_idx["y"]]
    z = atoms_data[:, col_idx["z"]]
    mols = atoms_data[:, col_idx["mol"]].astype(int) if "mol" in col_idx else np.zeros_like(ids)

    bead_data = np.column_stack([ids, types, mols, x, y, z])

    # Bonds dump convention expected from your current files:
    # columns: index, type, atom1, atom2, rx, ry, rz, dist, force
    atom1_ids = bonds_data[:, 2].astype(int)
    atom2_ids = bonds_data[:, 3].astype(int)

    id_to_idx = {int(aid): idx for idx, aid in enumerate(ids)}
    idx_a = np.array([id_to_idx.get(int(a), -1) for a in atom1_ids], dtype=int)
    idx_b = np.array([id_to_idx.get(int(b), -1) for b in atom2_ids], dtype=int)

    bad = (idx_a < 0) | (idx_b < 0)
    if np.any(bad):
        raise ValueError("Bond endpoints missing from atoms dump; cannot distill an inconsistent topology")

    bond_indices = np.arange(len(idx_a), dtype=int)
    n_bcols = bonds_data.shape[1]
    rx = bonds_data[:, 4] if n_bcols > 4 else np.zeros(len(idx_a))
    ry = bonds_data[:, 5] if n_bcols > 5 else np.zeros(len(idx_a))
    rz = bonds_data[:, 6] if n_bcols > 6 else np.zeros(len(idx_a))
    rn = bonds_data[:, 7] if n_bcols > 7 else np.zeros(len(idx_a))
    f = bonds_data[:, 8] if n_bcols > 8 else np.zeros(len(idx_a))

    # analyze_entanglements needs zero-based atom indices in columns 1 and 2.
    bond_arr = np.column_stack([bond_indices, idx_a, idx_b, rx, ry, rz, rn, f])
    return bead_data, bond_arr


def last_zero_strain_step(atoms_path):
    """Find the last isotropically swollen frame before the box changes shape.

    These CGMD runs start in a cubic box. Swelling changes all three sides
    equally; tension, compression and pure shear change their ratios.
    Scan only box headers so bead coordinates do not need to be converted.
    """
    last = step = None
    with open(atoms_path) as dump:
        for line in dump:
            if line.strip() == "ITEM: TIMESTEP":
                step = int(next(dump))
            elif line.startswith("ITEM: BOX BOUNDS"):
                bounds = np.array([list(map(float, next(dump).split())) for _ in range(3)])
                if bounds.shape != (3, 2) or step is None:
                    raise ValueError("Expected a timestep and orthorhombic box")
                lengths = bounds[:, 1] - bounds[:, 0]
                if not np.allclose(lengths, lengths[0], rtol=1e-8, atol=1e-8):
                    break
                last = step
    if last is None:
        raise ValueError("No zero-strain cubic frame found before loading")
    return last


def process_one(atoms_path, bonds_path, tag, train_dump_dir, target_dump_dir, target_timestep=TARGET_TIMESTEP):
    print(f"\n{'=' * 60}")
    print(f"Processing: {tag}")
    print(f"{'=' * 60}")

    if target_timestep == "undeformed":
        target_timestep = last_zero_strain_step(atoms_path)
    if target_timestep is None:
        atoms_data, atoms_header, num_atoms, box_bounds, ts_atoms = load_last_timestep_atoms(atoms_path)
        bonds_data, bonds_header, num_bonds, box_bounds_b, ts_bonds = load_last_timestep_bonds(bonds_path)
    else:
        atoms_data, atoms_header, num_atoms, box_bounds, ts_atoms = load_specific_timestep_atoms(atoms_path, target_timestep)
        bonds_data, bonds_header, num_bonds, box_bounds_b, ts_bonds = load_specific_timestep_bonds(bonds_path, target_timestep)

    if ts_atoms != ts_bonds:
        raise RuntimeError(f"Timestep mismatch: atoms={ts_atoms}, bonds={ts_bonds}")

    print(f"  Atoms: timestep={ts_atoms}  n_atoms={num_atoms}  header='{atoms_header}'")
    print(f"  Bonds: timestep={ts_bonds}  n_bonds={num_bonds}  header='{bonds_header}'")

    if not np.allclose(box_bounds, box_bounds_b):
        raise ValueError("Atoms and bonds have different box bounds")
    if len(np.unique(atoms_data[:, _parse_header_columns(atoms_header)["id"]])) != num_atoms:
        raise ValueError("Duplicate atom IDs")
    if bonds_data.shape[1] != 9:
        raise ValueError("Expected nine bond columns: index type atom1 atom2 dx dy dz dist force")

    bead_arr, bond_arr = _prepare_bead_and_bond_arrays(atoms_data, atoms_header, bonds_data)

    results = analyze_entanglements(
        bead_data=bead_arr,
        bond_data=bond_arr,
        box_bounds=box_bounds,
    )

    ent_rows = results["entanglement_rows"]
    ent_degree = results["smooth_ln_abs"]
    n_ent = results["nEntanglements"]
    print(f"  Entanglements found: {n_ent}")

    out_atoms = os.path.join(train_dump_dir, f"train_atoms_{tag}.dump")
    write_atoms_dump(out_atoms, atoms_data, atoms_header, num_atoms, box_bounds, ts_atoms, ent_degree)
    print(f"  Saved: {out_atoms}")

    out_bonds = os.path.join(train_dump_dir, f"train_bonds_{tag}.dump")
    write_bonds_dump(out_bonds, bonds_data, bonds_header, num_bonds, box_bounds_b, ts_bonds)
    print(f"  Saved: {out_bonds}")

    vertices, edges = build_primitive_path_vertices_edges(
        atoms_data,
        atoms_header,
        bonds_data,
        ent_rows,
        box_bounds,
        num_atoms,
        merge_cutoff=MERGE_CUTOFF,
        merge_same_anchor=MERGE_SAME_ANCHOR,
        zone_members=results.get("zone_members", None),
        zone_members_idx=results.get("zone_members_idx", None),
    )

    out_verts = os.path.join(target_dump_dir, f"vertices_{tag}.dump")
    write_vertices_dump(out_verts, vertices, box_bounds, ts_atoms)
    print(f"  Saved: {out_verts}  ({len(vertices)} vertices)")

    out_edges = os.path.join(target_dump_dir, f"edges_{tag}.dump")
    write_edges_dump(out_edges, edges, box_bounds_b, ts_bonds)
    print(f"  Saved: {out_edges}  ({len(edges)} edges)")

    # Save detected zone pairs separately from spatially merged DNM vertices.
    processed = os.path.join(os.path.dirname(target_dump_dir), "Output", "ProcessedData")
    os.makedirs(processed, exist_ok=True)
    columns = ["entanglement_id", "chain1", "chain2", "zone1", "zone2", "GLN",
               "coe_x", "coe_y", "coe_z", "minimum_contact_distance", "contact_count",
               "zone1_first_bead", "zone1_last_bead", "zone1_bonds",
               "zone2_first_bead", "zone2_last_bead", "zone2_bonds",
               "zone1_coe_bond_index", "zone2_coe_bond_index", "zone1_start_to_coe",
               "zone1_coe_to_end", "zone2_start_to_coe", "zone2_coe_to_end"]
    with open(os.path.join(processed, f"entanglements_{tag}.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(columns); writer.writerows(ent_rows)
    intra = sum(int(r[1]) == int(r[2]) for r in ent_rows)
    summary = {"timestep": int(ts_atoms), "detected_zone_pairs": int(n_ent),
               "intrachain_pairs": intra, "interchain_pairs": int(n_ent)-intra,
               "primitive_path_vertices": len(vertices), "primitive_path_edges": len(edges)}
    with open(os.path.join(processed, f"distillation_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2); f.write("\n")


def main():
    base_dir = os.getcwd()
    target_timestep = TARGET_TIMESTEP

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        value = sys.argv[2].strip().lower()
        target_timestep = None if value in {"none", "last"} else ("undeformed" if value == "undeformed" else int(value))

    training_data_dir = os.path.join(base_dir, "training_data")
    train_dump_dir = os.path.join(base_dir, "train_dump")
    target_dump_dir = os.path.join(base_dir, "target_dump")

    if not os.path.isdir(training_data_dir):
        print(f"ERROR: '{training_data_dir}' not found.")
        print("Expected folder structure:")
        print("  <base>/training_data/  (atoms_*.dump + bonds_*.dump)")
        print("  <base>/train_dump/     (auto-created)")
        print("  <base>/target_dump/    (auto-created)")
        sys.exit(1)

    os.makedirs(train_dump_dir, exist_ok=True)
    os.makedirs(target_dump_dir, exist_ok=True)

    pairs = _detect_pairs(training_data_dir)
    if not pairs:
        print(f"No atoms_*.dump / bonds_*.dump pairs found in {training_data_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} file pair(s) in {training_data_dir}")
    print(f"Target timestep: {target_timestep}")
    print(f"MERGE_CUTOFF={MERGE_CUTOFF}, MERGE_SAME_ANCHOR={MERGE_SAME_ANCHOR}")

    for atoms_path, bonds_path, tag in pairs:
        process_one(
            atoms_path,
            bonds_path,
            tag,
            train_dump_dir,
            target_dump_dir,
            target_timestep=target_timestep,
        )

    print(f"\n{'=' * 60}")
    print("ALL DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
