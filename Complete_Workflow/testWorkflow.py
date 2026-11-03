"""Small interface tests. No LAMMPS dynamics or validation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import runCompleteWorkflow as workflow
from dumpIO import iter_frames
from generateInputForDNM import _get_deformation_mode


def synthetic_run(cfg, root, model):
    """Create one static frame from the real generated topology for plumbing tests."""
    tag = cfg['tag'] + ('_DNM' if model == 'DNM' else '')
    base = root/model
    text = (base/'Input'/f'topology_{tag}.txt').read_text()
    bounds=[]
    for line in text.splitlines():
        if line.strip().endswith(('xlo xhi','ylo yhi','zlo zhi')):
            bounds.append([float(x) for x in line.split()[:2]])
    atom_section = text.split('\nAtoms')[1].split('\nBonds')[0]
    atom_rows = [[float(x) for x in l.split()] for l in atom_section.split('Velocities')[0].splitlines()[1:] if l.strip()]
    bond_section = text.split('\nBonds')[1]
    bond_rows = [[float(x) for x in l.split()] for l in bond_section.splitlines() if l.strip()]
    points = {int(r[0]):np.array(r[3:6]) for r in atom_rows}
    lengths = np.diff(bounds, axis=1).ravel()
    timestep = 139000000 if model == 'CGMD' else 0
    for kind, rows, header in [('atoms',atom_rows,'ITEM: ATOMS id x y z vx vy vz type mol'),('bonds',bond_rows,'ITEM: ENTRIES index c_Pairs[1] c_Pairs[2] c_Pairs[3] c_Bonds[1] c_Bonds[2] c_Bonds[3] c_Bonds[4] c_Bonds[5]')]:
        p=base/'Output/Dump'/f'{kind}_{tag}.dump'
        with p.open('w') as f:
            f.write(f'ITEM: TIMESTEP\n{timestep}\nITEM: NUMBER OF '+('ATOMS' if kind=='atoms' else 'ENTRIES')+f'\n{len(rows)}\nITEM: BOX BOUNDS pp pp pp\n')
            for lo,hi in bounds: f.write(f'{lo} {hi}\n')
            f.write(header+'\n')
            for row in rows:
                if kind=='atoms':
                    aid,mol,typ,x,y,z=row[:6]; values=[aid,x,y,z,0,0,0,typ,mol]
                else:
                    bid,typ,a,b=row; delta=points[int(a)]-points[int(b)]; delta-=lengths*np.round(delta/lengths)
                    values=[bid,typ,a,b,*delta,np.linalg.norm(delta),0]
                f.write(' '.join(map(str,values))+'\n')


class WorkflowTests(unittest.TestCase):
    def test_end_to_end_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); cfg=json.loads((workflow.HERE/'workflow.json').read_text()); cfg.pop('batch', None)
            cfg['network'].update(chains=2,beads_per_chain=12,phi_initial=0.05)
            cfg['tag']='small_tension_1'; cfg['output_directory']=str(p/'results')
            cfg['make_plots']=True
            config=p/'config.json'; config.write_text(json.dumps(cfg))
            with patch.object(sys,'argv',['runCompleteWorkflow.py','--config',str(config)]), patch.object(workflow,'run_lammps',side_effect=synthetic_run), contextlib.redirect_stdout(io.StringIO()):
                workflow.main()
                workflow.main()  # completed stages must be skipped
            result=p/'results'
            self.assertEqual(json.loads((result/'completed_stages.json').read_text()),list(workflow.STAGES))
            for model,tag in [('CGMD','small_tension_1'),('DNM','small_tension_1_DNM')]:
                self.assertTrue((result/model/'Output/ProcessedData'/f'frames_{tag}.csv').is_file())
                self.assertTrue((result/model/'Output/ProcessedData'/f'bond_statistics_{tag}.png').is_file())

    def test_batch_box_commands(self):
        from generateInputFromSAW import generateInputFromSAW
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'Input').mkdir()
            cfg = json.loads((workflow.HERE/'workflow.json').read_text()); cfg.pop('batch', None)
            initial_side = 37  # ceil((25000/0.5)**(1/3))
            meta = {'meta': {'L_grid': initial_side, 'phi': 0.5}}
            for mode in ('tension', 'compression', 'pure_shear'):
                for target in (0.02, 0.1, 0.5):
                    with self.subTest(mode=mode, target=target):
                        cfg['cgmd'].update(phi_target=target, loading_mode=mode)
                        path = base/'case.json'
                        path.write_text(json.dumps(cfg))
                        resolved = workflow.read_config(path)
                        tag = resolved['tag']
                        self.assertEqual(resolved['distillation']['timestep'], 'undeformed')
                        (base/'Input'/f'topology_{tag}.txt').touch()
                        generateInputFromSAW(meta, tag, base, resolved['cgmd'])
                        lines = (base/'Input'/f'input_{tag}.in').read_text().splitlines()
                        def bounds(prefix):
                            tokens = next(l for l in lines if l.startswith(prefix)).split()
                            return np.array([[float(tokens[i]), float(tokens[i+1])] for i in (7,11,15)])
                        swollen = bounds('fix EXPAND ')
                        side = np.diff(swollen).ravel()
                        self.assertAlmostEqual(0.5 * initial_side**3 / np.prod(side), target, places=7)
                        np.testing.assert_allclose(swollen.mean(axis=1), initial_side/2, atol=1e-6)
                        final = bounds('fix DEF20 ')
                        stretch = 0.7 if mode == 'compression' else 1.3
                        transverse = [1/stretch, 1] if mode == 'pure_shear' else [stretch**-0.5]*2
                        np.testing.assert_allclose(np.diff(final).ravel()/side, [stretch, *transverse], atol=1e-6)
                        if mode == 'pure_shear':
                            np.testing.assert_allclose(final.mean(axis=1), initial_side/2, atol=1e-6)
                        else:
                            self.assertAlmostEqual(final[0,0], swollen[0,0], places=6)

    def test_batch_combinations(self):
        settings = json.loads((workflow.HERE/'workflow.json').read_text())
        settings['batch'] = {'chains': [2, 3], 'beads_per_chain': [12, 15], 'phi_initial': [0.5],
                             'sweep_plan': [{'phi_target': 0.02, 'tension': 2, 'compression': 1, 'pure_shear': 1},
                                            {'phi_target': 0.1, 'tension': 1}]}
        cases = workflow.batch_cases(settings)
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c['output_directory'] for c in cases}), 20)
        self.assertEqual(len({c['network']['seed'] for c in cases}), 20)
        np.testing.assert_allclose(cases[0]['cgmd']['strains'], np.linspace(0, 0.3, 20))
        self.assertEqual(cases[0]['cgmd']['strains'], cases[0]['dnm']['strains'])
        self.assertEqual(cases, workflow.batch_cases(settings))

    def test_truncated_dump_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.dump'
            p.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z\n')
            with self.assertRaises(ValueError): list(iter_frames(p))

    def test_loading_mode_alias(self):
        self.assertEqual(_get_deformation_mode('C25_B1000_phi2_shear_1'),'pure_shear')
        self.assertEqual(_get_deformation_mode('C25_B1000_phi2_pure_shear_1'),'pure_shear')
        with self.assertRaises(ValueError): _get_deformation_mode('C50_B1000_phi50_1')

    def test_failed_simulation_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); cfg=json.loads((workflow.HERE/'workflow.json').read_text()); cfg.pop('batch', None); cfg['output_directory']=str(p/'results')
            config=p/'config.json'; config.write_text(json.dumps(cfg))
            with patch.object(sys,'argv',['workflow','--config',str(config),'--stage','run-cgmd']), patch.object(workflow,'run_lammps',side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError): workflow.main()
            self.assertEqual(json.loads((p/'results/completed_stages.json').read_text()),[])


if __name__ == '__main__':
    unittest.main()
