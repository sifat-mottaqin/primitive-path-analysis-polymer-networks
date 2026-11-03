"""Export per-frame geometry, bond statistics and LAMMPS thermo tables."""
import csv
from itertools import zip_longest
import numpy as np
from dumpIO import iter_frames, load_specific_timestep_atoms


def export_thermo(log, output):
    if not log.exists():
        return
    columns = None; rows = []; block = 0
    for line in log.read_text(errors='replace').splitlines():
        words = line.split()
        if words and words[0] == 'Step':
            columns = words; block += 1; continue
        if columns and len(words) == len(columns):
            try:
                values = [float(x) for x in words]
            except ValueError:
                columns = None; continue
            rows.append({'block': block, **dict(zip(columns, values))})
        elif columns:
            columns = None
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with output.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def processResults(base, tag, reference_timestep=None, make_plots=True):
    atoms = base/'Output/Dump'/f'atoms_{tag}.dump'
    bonds = base/'Output/Dump'/f'bonds_{tag}.dump'
    out = base/'Output/ProcessedData'; out.mkdir(parents=True, exist_ok=True)
    reference = None
    if reference_timestep is not None:
        reference = np.diff(load_specific_timestep_atoms(atoms, reference_timestep)[3], axis=1).ravel()
    keys = ['timestep','atoms','bonds','Lx','Ly','Lz','volume','lambda_x','lambda_y','lambda_z','engineering_strain_x','mean_bond_length','mean_abs_bond_force']
    rows = []
    with (out/f'frames_{tag}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for a,b in zip_longest(iter_frames(atoms), iter_frames(bonds)):
            if a is None or b is None or a[4] != b[4] or not np.allclose(a[3], b[3]):
                raise ValueError('Atom and bond frames must match in number, timestep and box bounds')
            columns = b[1].split()[2:]
            if columns != ['index','c_Pairs[1]','c_Pairs[2]','c_Pairs[3]','c_Bonds[1]','c_Bonds[2]','c_Bonds[3]','c_Bonds[4]','c_Bonds[5]']:
                raise ValueError('Unexpected bond columns; use the supplied input generators')
            lengths = np.diff(a[3], axis=1).ravel()
            if reference is None:
                reference = lengths.copy()
            stretches = lengths/reference
            row = dict(zip(keys, [a[4],a[2],b[2],*lengths,np.prod(lengths),*stretches,stretches[0]-1,
                                  np.mean(b[0][:,7]) if b[2] else float('nan'),
                                  np.mean(np.abs(b[0][:,8])) if b[2] else float('nan')]))
            writer.writerow(row); rows.append(row)
    if not rows:
        raise ValueError('No frames to process')
    export_thermo(base/'Output/Logs'/f'log_{tag}.lammps', out/f'thermo_{tag}.csv')
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = [r for r in rows if reference_timestep is None or r['timestep'] >= reference_timestep]
        fig, axes = plt.subplots(1,2,figsize=(9,3.5))
        for ax,key,label in zip(axes,['mean_bond_length','mean_abs_bond_force'],['Mean bond length (LJ)','Mean absolute bond force (LJ)']):
            ax.plot([r['engineering_strain_x'] for r in selected],[r[key] for r in selected],lw=1)
            ax.set(xlabel='Engineering strain in x',ylabel=label)
        fig.tight_layout(); fig.savefig(out/f'bond_statistics_{tag}.png',dpi=200); plt.close(fig)
