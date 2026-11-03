"""CGMD generation, topology distillation, DNM simulation and CSV processing."""
import argparse
import copy
import csv
from itertools import product
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent


def read_config(path):
    return validate_case(json.loads(path.read_text()))


def validate_case(cfg):
    """Resolve the common strain grid and check one case before creating files."""
    if 'strain_grid' in cfg:
        grid = cfg['strain_grid']
        if type(grid['count']) is not int or grid['count'] < 2:
            raise ValueError('strain_grid.count must be an integer >= 2')
        strains = np.linspace(grid['start'], grid['stop'], grid['count']).tolist()
        for model in ('cgmd', 'dnm'):
            cfg[model]['strains'] = strains.copy()
    n = cfg['network']
    for key in ('chains', 'beads_per_chain'):
        if type(n[key]) is not int or n[key] < (2 if key == 'beads_per_chain' else 1):
            raise ValueError(f'Invalid network.{key}')
    if not 0 < n['phi_initial'] <= 1:
        raise ValueError('phi_initial must be in (0, 1]')
    cg = cfg['cgmd']
    if 'swelling_factor' in cg or 'phi_target' not in cg:
        raise ValueError('Set cgmd.phi_target and remove the legacy swelling_factor setting')
    if not 0 < cg['phi_target'] <= n['phi_initial']:
        raise ValueError('Require 0 < cgmd.phi_target <= network.phi_initial')
    mode = cg['loading_mode']
    if mode not in ('tension', 'compression', 'pure_shear'):
        raise ValueError('Invalid cgmd.loading_mode')
    label = 'shear' if mode == 'pure_shear' else mode
    phi_code = int(np.floor(100 * cg['phi_target'] + 0.5))
    cfg.setdefault('tag', f"C{n['chains']}_B{n['beads_per_chain']}_phi{phi_code}_{label}_1")
    if not cfg['tag'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in cfg['tag']):
        raise ValueError('tag must contain only letters, numbers, underscores and hyphens')
    from generateInputForDNM import _get_deformation_mode
    if _get_deformation_mode(cfg['tag']) != mode:
        raise ValueError('File tag and cgmd.loading_mode must agree')
    for model in ('cgmd', 'dnm'):
        p = cfg[model]
        for k in ('dump_every', 'pull_steps', 'hold_steps') + (('initial_hold_steps',) if model == 'cgmd' else ()):
            if type(p[k]) is not int or p[k] <= 0:
                raise ValueError(f'{model}.{k} must be a positive integer')
        if p['pull_steps'] % 100:
            raise ValueError(f'{model}.pull_steps must be a multiple of the deform interval (100)')
        if p['dt'] <= 0 or not np.isfinite(p['dt']):
            raise ValueError(f'Invalid {model}.dt')
        strains = p['strains']
        if not strains or strains[0] != 0 or not np.isfinite(strains).all() or any(x < 0 for x in strains) or any(a >= b for a,b in zip(strains, strains[1:])):
            raise ValueError(f'{model}.strains must start at zero and increase')
    if cfg['cgmd']['strains'] != cfg['dnm']['strains']:
        raise ValueError('Use matching CGMD/DNM strain grids')
    if mode == 'compression' and max(cg['strains']) >= 1:
        raise ValueError('Compression strain magnitude must be below 1')
    d = cfg['distillation']
    for k in ('ln_threshold', 'merge_cutoff', 'topology_merge_cutoff'):
        if not np.isfinite(d[k]) or d[k] < 0:
            raise ValueError(f'Invalid distillation.{k}')
    if d['timestep'] not in (None, 'undeformed') and (type(d['timestep']) is not int or d['timestep'] < 0):
        raise ValueError('distillation.timestep must be a undeformed, a nonnegative integer, or null')
    for key in ('cgmd_command', 'dnm_command'):
        if not isinstance(cfg[key], list) or not cfg[key] or not all(isinstance(x, str) for x in cfg[key]):
            raise ValueError(f'{key} must be a nonempty list of command arguments')
    return cfg


def workspace(cfg, config_path):
    p = Path(cfg['output_directory']).expanduser()
    return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()


def folders(base):
    for sub in ('Input', 'Coordinates', 'Output/Dump', 'Output/Logs', 'Output/ProcessedData', 'train_dump', 'target_dump'):
        (base / sub).mkdir(parents=True, exist_ok=True)


def prepare_cgmd(cfg, root):
    """Build the initial self-avoiding chains and write the CGMD topology and loading input."""
    from generateSAWCoordinates import generateSAWCoordinates
    from generateTopologyFromSAW import generateTopologyFromSAW
    from generateInputFromSAW import generateInputFromSAW
    base = root / 'CGMD'; folders(base)
    n = cfg['network']; tag = cfg['tag']
    np.random.seed(n['seed']); random.seed(n['seed'])
    generateSAWCoordinates(n['chains'], n['beads_per_chain'], n['phi_initial'], tag, str(base))
    with (base/'Coordinates'/f'coords_{tag}.pkl').open('rb') as f:
        coords = pickle.load(f)
    generateTopologyFromSAW(coords, tag, str(base))
    factor = generateInputFromSAW(coords, tag, str(base), cfg['cgmd'])
    L = coords['meta']['L_grid']
    # Integer lattice dimensions can make the realized density differ from the requested fraction.
    actual = n['chains']*n['beads_per_chain']/L**3
    (base/'density.json').write_text(json.dumps({'phi_initial': n['phi_initial'], 'phi_target': cfg['cgmd']['phi_target'], 'initial_side': L, 'swollen_side': L*factor, 'swelling_factor': factor, 'initial_number_density': actual, 'swollen_number_density': actual/factor**3}, indent=2)+'\n')


def preflight(command, dnm=False):
    # The command can include an MPI/Slurm launcher. -help reaches LAMMPS.
    result = subprocess.run(command + ['-help'], capture_output=True, text=True, check=True)
    help_text = result.stdout + result.stderr
    if 'brownian' not in help_text:
        raise RuntimeError('LAMMPS help does not list brownian; use the build required by these inputs')
    if dnm and 'pade' not in help_text:
        raise RuntimeError('LAMMPS help does not list pade. The custom bond-style source is not included in this archive.')


def run_lammps(cfg, root, model):
    dnm = model == 'DNM'; base = root/model; folders(base)
    tag = cfg['tag'] + ('_DNM' if dnm else '')
    command = cfg['dnm_command' if dnm else 'cgmd_command']
    preflight(command, dnm)
    inp = base/'Input'/f'input_{tag}.in'
    if not inp.is_file():
        raise FileNotFoundError(inp)
    for kind in ('atoms', 'bonds'):
        p = base/'Input'/f'{kind}_{tag}.dump'
        if p.exists():
            raise FileExistsError(f'Partial dump exists: {p}. Inspect and remove it before retrying.')
    log = base/'Output/Logs'/f'log_{tag}.lammps'
    screen = base/'Output/Logs'/f'screen_{tag}.txt'
    with screen.open('w') as output:
        subprocess.run(command + ['-in', inp.name, '-log', str(log)], cwd=inp.parent,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    for kind in ('atoms', 'bonds'):
        p = base/'Input'/f'{kind}_{tag}.dump'
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f'LAMMPS did not create {p}; see {screen}')
    for kind in ('atoms', 'bonds'):
        p = base/'Input'/f'{kind}_{tag}.dump'
        os.replace(p, base/'Output/Dump'/p.name)


def distill(cfg, root):
    """Read one paired atom/bond frame and convert its chain ends and entanglements into a primitive-path graph."""
    import processEntanglement as detector
    import runProcessDump2DNM as pipeline
    base = root/'DNM'; folders(base)
    tag = cfg['tag']; d = cfg['distillation']
    # Apply the GLN threshold to zone pairs before spatial merging of their centers.
    detector.LN_pair_abs_threshold = d['ln_threshold']
    pipeline.MERGE_CUTOFF = d['merge_cutoff']
    pipeline.MERGE_SAME_ANCHOR = False
    source = root/'CGMD/Output/Dump'
    pipeline.process_one(str(source/f'atoms_{tag}.dump'), str(source/f'bonds_{tag}.dump'), tag,
                         str(base/'train_dump'), str(base/'target_dump'), d['timestep'])


def prepare_dnm(cfg, root):
    """Convert the graph vertices and edges into a LAMMPS data file and loading script."""
    from generateTopologyForDNM import generateTopologyForDNM
    from generateInputForDNM import generateInputForDNM
    base = root/'DNM'; folders(base)
    generateTopologyForDNM(cfg['tag'], str(base), merge_cutoff=cfg['distillation']['topology_merge_cutoff'], bond_merge_mode='keep_all')
    generateInputForDNM(cfg['tag'], str(base), cfg['dnm'])


def process(cfg, root):
    """Export box stretches and bond statistics for both representations of the network."""
    from processResults import processResults
    for model in ('CGMD', 'DNM'):
        tag = cfg['tag'] + ('_DNM' if model == 'DNM' else '')
        # CGMD reference is the selected undeformed state; DNM starts there.
        reference = cfg['distillation']['timestep'] if model == 'CGMD' else None
        if reference == 'undeformed':
            summary = root/'DNM/Output/ProcessedData'/f'distillation_{tag}.json'
            reference = json.loads(summary.read_text())['timestep']
        processResults(root/model, tag, reference, cfg['make_plots'])


STAGES = ('prepare-cgmd', 'run-cgmd', 'distill', 'prepare-dnm', 'run-dnm', 'process')


def run_case(cfg, path, stage_name, overwrite):
    """Run the requested stages in order. Record a stage only after it succeeds."""
    root = workspace(cfg, path)
    root.mkdir(parents=True, exist_ok=True)
    saved = root/'settings_used.json'
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise RuntimeError('Settings differ from this run. Use a new output_directory to avoid mixing cases.')
    saved.write_text(json.dumps(cfg, indent=2)+'\n')
    state_path = root/'completed_stages.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else []
    actions = dict(zip(STAGES, (prepare_cgmd, lambda c,r: run_lammps(c,r,'CGMD'), distill,
                               prepare_dnm, lambda c,r: run_lammps(c,r,'DNM'), process)))
    selected = STAGES if stage_name == 'all' else (stage_name,)
    for stage in selected:
        if stage in state and not overwrite:
            print(f'Skipping completed stage: {stage}'); continue
        index = STAGES.index(stage)
        # Repeating an upstream stage invalidates results recorded for later stages.
        state = [s for s in state if STAGES.index(s) < index]
        state_path.write_text(json.dumps(state)+'\n')
        print(f'Running {stage}', flush=True)
        actions[stage](cfg, root)
        state.append(stage); state_path.write_text(json.dumps(state)+'\n')
    print(f'Outputs: {root}')



def batch_cases(settings):
    """Expand the sweep table into independent networks with reproducible seeds and separate output directories."""
    batch = settings.get('batch')
    if batch is None:
        return [validate_case(copy.deepcopy(settings))]
    sizes = batch.get('beads_per_chain', [None])
    axes = [batch['chains'], sizes, batch['phi_initial']]
    for axis in axes:
        if not isinstance(axis, list) or not axis or len(set(axis)) != len(axis):
            raise ValueError('Batch axes must be nonempty lists without duplicates')
    cases = []
    seen = set()
    for chains, beads, initial in product(*axes):
        if beads is None:
            beads = (batch['total_beads'] + chains - 1) // chains
        for row_index, row in enumerate(batch['sweep_plan']):
            for mode in ('tension', 'compression', 'pure_shear'):
                repeats = row.get(mode, 0)
                if type(repeats) is not int or repeats < 0:
                    raise ValueError('Run counts must be nonnegative integers')
                for repeat in range(1, repeats + 1):
                    key = (chains, beads, initial, row['phi_target'], mode, repeat)
                    if key in seen:
                        raise ValueError(f'Duplicate batch case: {key}')
                    seen.add(key)
                    cfg = copy.deepcopy(settings)
                    cfg.pop('batch')
                    cfg.pop('tag', None)
                    cfg['network'].update(chains=chains, beads_per_chain=beads, phi_initial=initial)
                    network_index = axes[0].index(chains) + len(axes[0]) * (axes[2].index(initial) + len(axes[2]) * (0 if sizes == [None] else sizes.index(beads)))
                    mode_index = ('tension', 'compression', 'pure_shear').index(mode)
                    cfg['network']['seed'] = settings['network']['seed'] + 100000 * network_index + 10000 * row_index + 1000 * mode_index + repeat
                    cfg['cgmd'].update(phi_target=row['phi_target'], loading_mode=mode)
                    initial_label = str(initial).replace('.', 'p')
                    target_label = str(row['phi_target']).replace('.', 'p')
                    cfg['tag'] = f'C{chains}_B{beads}_phiI{initial_label}_phiT{target_label}_{mode}_{repeat}'
                    cfg['output_directory'] = str(Path(settings['output_directory']) / cfg['tag'])
                    cases.append(validate_case(cfg))
    if not cases:
        raise ValueError('The batch requests zero runs')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'workflow.json')
    parser.add_argument('--stage', choices=('all', 'check')+STAGES, default='all')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--list-cases', action='store_true', help='Print the batch without running simulations')
    args = parser.parse_args()
    path = args.config.resolve()
    settings = json.loads(path.read_text())
    cases = batch_cases(settings)
    for number, cfg in enumerate(cases, 1):
        print(f"{number}/{len(cases)}  {cfg['tag']}  seed={cfg['network']['seed']}", flush=True)
    if args.list_cases:
        return
    if args.stage == 'check':
        preflight(settings['cgmd_command'])
        preflight(settings['dnm_command'], True)
        return
    summary = []
    for number, cfg in enumerate(cases, 1):
        print(f"Case {number}/{len(cases)}: {cfg['tag']}", flush=True)
        try:
            run_case(cfg, path, args.stage, args.overwrite)
            summary.append({'case': cfg['tag'], 'status': 'completed', 'error': ''})
        except Exception as error:
            if 'batch' not in settings:
                raise
            summary.append({'case': cfg['tag'], 'status': 'failed', 'error': str(error)})
            print(f"Failed {cfg['tag']}: {error}", file=sys.stderr, flush=True)
        if 'batch' in settings:
            base = workspace(settings, path)
            base.mkdir(parents=True, exist_ok=True)
            with (base/'batch_summary.csv').open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=['case', 'status', 'error'])
                writer.writeheader()
                writer.writerows(summary)
    if any(row['status'] == 'failed' for row in summary):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
