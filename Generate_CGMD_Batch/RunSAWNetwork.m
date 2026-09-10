clear;
close all;
clc;

%%  PROJECT ROOT
mainFolder = '/data/home/bsifatmottaq/2026_Ent_Study';
if ~exist(mainFolder,'dir')
    error('runSAWNetwork:MissingProjectFolder', ...
        'Project folder does not exist: %s', mainFolder);
end
cd(mainFolder);

%%  NETWORK CONFIGURATION
total_beads = 25000;       % total number of beads in the system
chains_list = [25];		   % % total number of chains in the system

% IMPORTANT:
% Every SAW network is generated at this INITIAL packing fraction.
% The LAMMPS input then swells that network to each requested phi_target.
phi_initial = 0.50;

%% SWEEP PLAN
%  Columns:
%    1 = phi_target
%    2 = number of uniaxial-tension cases
%    3 = number of uniaxial-compression cases
%    4 = number of pure-shear cases
%
%  Set any count to 0 to skip that mode at that phi_target.
%  Add/remove rows freely.

sweep_plan = [
      0.10         0           0             0
      0.05         0           0             0
      0.02         1           0             0
];

%%  CONTROL SWITCHES
create_folders        = 1;
gen_SAW_network       = 1;
vis_SAW_network       = 0;
save_figures          = 0;
generate_topology     = 1;
generate_input        = 1;
run_lammps            = 1;

override_SAW_network  = 1;
override_figures      = 1;
override_topology     = 1;
override_input        = 1;
override_lammps       = 1;


%%  VALIDATE SWEEP PLAN

if size(sweep_plan,2) ~= 4
    error('runSAWNetwork:BadSweepPlan', ...
        'sweep_plan must have exactly 4 columns: phi_target, tension, compression, pure_shear.');
end

if any(~isfinite(sweep_plan(:)))
    error('runSAWNetwork:BadSweepPlan','sweep_plan contains non-finite values.');
end

if any(sweep_plan(:,1) <= 0) || any(sweep_plan(:,1) > phi_initial)
    error('runSAWNetwork:BadTargetPhi', ...
        'Every phi_target must satisfy 0 < phi_target <= phi_initial (%.6g).', phi_initial);
end

runCounts = sweep_plan(:,2:4);
if any(runCounts(:) < 0) || any(abs(runCounts(:) - round(runCounts(:))) > 1e-12)
    error('runSAWNetwork:BadRunCounts', ...
        'Tension/compression/pure-shear case counts must be nonnegative integers.');
end
runCounts = round(runCounts);
sweep_plan(:,2:4) = runCounts;

mode_names = {'tension','compression','pure_shear'};
mode_file_names = {'tension','compression','shear'};

cases_per_chain = sum(sum(sweep_plan(:,2:4)));
total_combinations = numel(chains_list) * cases_per_chain;
if total_combinations < 1
    error('runSAWNetwork:EmptySweep','The sweep plan requests zero simulations.');
end


%%  HPC LAMMPS / MPI CONFIGURATION
%  This workflow intentionally requires MPI.

useGPU      = false;
parallelRun = true;

lammps_exe = getenv('LAMMPS_EXE');
if isempty(lammps_exe)
    lammps_exe = '/data/home/bsifatmottaq/lammps_build/build_cpu/lmp';
end

ranks = str2double(getenv('LAMMPS_RANKS'));
if ~isfinite(ranks) || ranks < 2
    ranks = str2double(getenv('SLURM_NTASKS'));
end
if ~isfinite(ranks) || ranks < 2
    error('runSAWNetwork:MPIRequired', ...
        ['No valid MPI rank count was found. Submit this workflow with ', ...
         'sbatch run_DNM_v2.sh.']);
end
ranks = round(ranks);

mpiLauncher = getenv('LAMMPS_MPIRUN');
if isempty(mpiLauncher)
    error('runSAWNetwork:MissingMPIEnvironment', ...
        'LAMMPS_MPIRUN is empty. Submit with sbatch run_DNM_v2.sh.');
end

if ~exist(lammps_exe,'file')
    error('runSAWNetwork:MissingLAMMPS', ...
        'LAMMPS executable not found: %s', lammps_exe);
end


%%  FOLDERS
if create_folders
    subDirs = {'Output','Input','Coordinates','Figures', ...
        'Output/Dump','Output/Logs','Output/RawData','Output/ProcessedData'};
    for k = 1:numel(subDirs)
        d = fullfile(mainFolder, subDirs{k});
        if ~exist(d,'dir'), mkdir(d); end
    end
end

inpDir   = fullfile(mainFolder,'Input');
outDump  = fullfile(mainFolder,'Output','Dump');
outRaw   = fullfile(mainFolder,'Output','RawData');
outProc  = fullfile(mainFolder,'Output','ProcessedData');
coordDir = fullfile(mainFolder,'Coordinates');
figDir   = fullfile(mainFolder,'Figures');


%  INITIALIZE SWEEP TRACKING
sweep_log = struct([]);
sweep_counter = 0;
overall_start = tic;

fprintf('====================================================================\n');
fprintf('Starting phi / loading-mode parameter sweep on HPC\n');
fprintf('Project root        : %s\n', mainFolder);
fprintf('Total beads         : %d\n', total_beads);
fprintf('Chain counts        : %s\n', mat2str(chains_list));
fprintf('Initial SAW phi     : %.6g\n', phi_initial);
fprintf('Total requested runs: %d\n', total_combinations);
fprintf('LAMMPS executable   : %s\n', lammps_exe);
fprintf('MPI launcher        : %s\n', mpiLauncher);
fprintf('MPI ranks/run       : %d\n', ranks);
fprintf('--------------------------------------------------------------------\n');
fprintf('Sweep plan:\n');
fprintf(' phi_target   fact        tension  compression  pure_shear\n');
for s_idx = 1:size(sweep_plan,1)
    phi_target = sweep_plan(s_idx,1);
    fact = (phi_initial / phi_target)^(1/3);
    fprintf(' %-11.5g %-11.6f %-8d %-12d %-10d\n', ...
        phi_target, fact, sweep_plan(s_idx,2), sweep_plan(s_idx,3), sweep_plan(s_idx,4));
end
fprintf('====================================================================\n\n');


%%  PARAMETER SWEEP
%  Each requested case is an independent SAW realization generated at
%  phi_initial, followed by swelling to phi_target and the selected loading.

for c_idx = 1:numel(chains_list)

    no_chains = chains_list(c_idx);
    no_beads = ceil(total_beads / no_chains);

    for s_idx = 1:size(sweep_plan,1)

        phi_target = sweep_plan(s_idx,1);
        fact = (phi_initial / phi_target)^(1/3);

        for mode_idx = 1:numel(mode_names)

            deformation_mode = mode_names{mode_idx};
            mode_file = mode_file_names{mode_idx};
            num_runs_this_mode = sweep_plan(s_idx,1 + mode_idx);

            for run = 1:num_runs_this_mode

                sweep_counter = sweep_counter + 1;

                % Deterministic, unique seed for chain/phi/mode/run combination.
                random_seed = 26 ...
                    + 100000*(c_idx-1) ...
                    + 10000*(s_idx-1) ...
                    + 1000*(mode_idx-1) ...
                    + run;
                rng(random_seed);

                % phi in the file tag means TARGET phi, not initial SAW phi.
                phiCode = round(phi_target * 100);
                fileTag = sprintf('C%d_B%d_phi%d_%s_%d', ...
                    no_chains, no_beads, phiCode, mode_file, run);

                coordOut = fullfile(coordDir, ['coords_', fileTag, '.mat']);
                topoOut  = fullfile(inpDir, sprintf('topology_%s.txt', fileTag));
                inputOut = fullfile(inpDir, sprintf('input_%s.in', fileTag));
                atomDumpOut = fullfile(outDump, sprintf('atoms_%s.dump', fileTag));
                bondDumpOut = fullfile(outDump, sprintf('bonds_%s.dump', fileTag));
                rawOut = fullfile(outRaw, sprintf('raw_%s.mat', fileTag)); %#ok<NASGU>
                ppOut  = fullfile(outProc, sprintf('raw_%s_PP.mat', fileTag)); %#ok<NASGU>
                figPng = fullfile(figDir, ['visSAW_', fileTag, '.png']);
                figFig = fullfile(figDir, ['visSAW_', fileTag, '.fig']);

                run_start = tic;

                fprintf('\n====================================================================\n');
                fprintf('Case %d of %d\n', sweep_counter, total_combinations);
                fprintf('Mode               : %s\n', deformation_mode);
                fprintf('Initial phi        : %.6g\n', phi_initial);
                fprintf('Target phi         : %.6g\n', phi_target);
                fprintf('Swelling fact      : %.10g\n', fact);
                fprintf('Check phi_i/fact^3 : %.10g\n', phi_initial/fact^3);
                fprintf('Chains             : %d\n', no_chains);
                fprintf('Beads/chain        : %d\n', no_beads);
                fprintf('Total beads        : %d\n', no_chains * no_beads);
                fprintf('Mode iteration     : %d of %d\n', run, num_runs_this_mode);
                fprintf('Random seed        : %d\n', random_seed);
                fprintf('File tag           : %s\n', fileTag);
                fprintf('====================================================================\n');

                coordsOK = false;
                ok = false;
                error_message = '';

                try
                    % SAW NETWORK GENERATION AT phi_initial
                    if gen_SAW_network
                        if exist(coordOut,'file')
                            if override_SAW_network
                                fprintf('[SAW] Coordinates exist; override=1 -> regenerating at phi_initial=%.6g ...\n', phi_initial);
                                generateSAWCordinates(no_chains, no_beads, phi_initial, fileTag, mainFolder);
                            else
                                fprintf('[SAW] Coordinates exist; override=0 -> skipping ...\n');
                            end
                        else
                            fprintf('[SAW] Coordinates missing -> generating at phi_initial=%.6g ...\n', phi_initial);
                            generateSAWCordinates(no_chains, no_beads, phi_initial, fileTag, mainFolder);
                        end
                    end

                    % LOAD COORDINATES
                    coordsOK = exist(coordOut,'file') ~= 0;
                    if coordsOK
                        S = load(coordOut, 'chainPos', 'combined', 'meta');
                        if ~isfield(S,'meta') || ~isfield(S.meta,'phi')
                            error('runSAWNetwork:BadCoordinates', ...
                                'Coordinate file is missing meta.phi: %s', coordOut);
                        end
                        if abs(S.meta.phi - phi_initial) > 1e-12
                            error('runSAWNetwork:WrongInitialPhi', ...
                                ['Coordinate file was generated at phi=%.10g, but this sweep ', ...
                                 'requires phi_initial=%.10g. Regenerate coordinates.'], ...
                                 S.meta.phi, phi_initial);
                        end
                        fprintf('[LOAD] Coordinates loaded; SAW phi = %.6g\n', S.meta.phi);
                    else
                        error('runSAWNetwork:MissingCoordinates', ...
                            'Coordinates are missing: %s', coordOut);
                    end

                    % VISUALIZATION (OPTIONAL)
                    if vis_SAW_network
                        if save_figures && (exist(figPng,'file') || exist(figFig,'file')) && ~override_figures
                            fprintf('[VIS] Figure exists; override=0 -> skipping plot save\n');
                            visSAWNetwork(S.chainPos, S.meta, fileTag, mainFolder, false, figPng, figFig);
                        else
                            visSAWNetwork(S.chainPos, S.meta, fileTag, mainFolder, save_figures, figPng, figFig);
                        end
                    end

                    % TOPOLOGY GENERATION
                    if generate_topology
                        if exist(topoOut,'file')
                            if override_topology
                                fprintf('[TOPO] Topology exists; override=1 -> regenerating ...\n');
                                generateTopologyFromSAW(S, fileTag, mainFolder);
                            else
                                fprintf('[TOPO] Topology exists; override=0 -> skipping ...\n');
                            end
                        else
                            fprintf('[TOPO] Topology missing -> generating ...\n');
                            generateTopologyFromSAW(S, fileTag, mainFolder);
                        end
                    end

                    % ONE INPUT GENERATOR FOR ALL THREE MODES
                    if generate_input
                        if exist(inputOut,'file') && ~override_input
                            fprintf('[INPUT] Input exists; override=0 -> skipping ...\n');
                        else
                            if exist(inputOut,'file')
                                fprintf('[INPUT] Input exists; override=1 -> regenerating ...\n');
                            else
                                fprintf('[INPUT] Input missing -> generating ...\n');
                            end

                            generateInputFromSAW( ...
                                S, fileTag, mainFolder, deformation_mode, phi_target);

                            fprintf('[INPUT] Mode=%s | phi_i=%.6g | phi_target=%.6g | fact=%.10g\n', ...
                                deformation_mode, phi_initial, phi_target, fact);
                        end
                    end

                    % LAMMPS MPI SIMULATION
                    if run_lammps
                        if exist(atomDumpOut,'file') && exist(bondDumpOut,'file') && ~override_lammps
                            fprintf('[LAMMPS] Dumps exist; override=0 -> skipping ...\n');
                            ok = true;
                        else
                            if exist(atomDumpOut,'file') && exist(bondDumpOut,'file')
                                fprintf('[LAMMPS] Dumps exist; override=1 -> re-running ...\n');
                            else
                                fprintf('[LAMMPS] Dumps missing -> running MPI LAMMPS ...\n');
                            end

                            ok = runLammpsFromSAW_HPC( ...
                                lammps_exe, fileTag, mainFolder, ...
                                'Parallel', parallelRun, ...
                                'Ranks', ranks, ...
                                'UseGPU', useGPU);
                        end
                    else
                        ok = exist(atomDumpOut,'file') && exist(bondDumpOut,'file');
                    end

                    if ~ok
                        error('runSAWNetwork:LAMMPSFailed', ...
                            'LAMMPS did not complete successfully for %s.', fileTag);
                    end

                catch ME
                    ok = false;
                    error_message = ME.message;
                    fprintf(2, '[ERROR] %s\n', ME.message);
                    if ~isempty(ME.stack)
                        fprintf(2, '[ERROR] %s line %d\n', ME.stack(1).name, ME.stack(1).line);
                    end
                end

                run_duration = toc(run_start);
                status = ok && coordsOK;

                entry = struct( ...
                    'run', sweep_counter, ...
                    'no_chains', no_chains, ...
                    'no_beads_per_chain', no_beads, ...
                    'total_beads', no_chains * no_beads, ...
                    'phi_initial', phi_initial, ...
                    'phi_target', phi_target, ...
                    'swelling_factor', fact, ...
                    'deformation_mode', deformation_mode, ...
                    'mode_iteration', run, ...
                    'random_seed', random_seed, ...
                    'fileTag', fileTag, ...
                    'status', status, ...
                    'duration_sec', run_duration, ...
                    'error_message', error_message);

                if isempty(sweep_log)
                    sweep_log = entry;
                else
                    sweep_log(end+1,1) = entry; 
                end

                if status
                    fprintf('[RESULT] Case %d: PASSED in %.2f sec (%.2f min)\n', ...
                        sweep_counter, run_duration, run_duration/60);
                else
                    fprintf('[RESULT] Case %d: FAILED in %.2f sec (%.2f min)\n', ...
                        sweep_counter, run_duration, run_duration/60);
                end
            end
        end
    end
end


%  FINAL SUMMARY
overall_duration = toc(overall_start);
if isempty(sweep_log)
    num_success = 0;
else
    num_success = sum([sweep_log.status]);
end
num_failed = sweep_counter - num_success;

fprintf('\n====================================================================\n');
fprintf('SWEEP COMPLETE\n');
fprintf('====================================================================\n');
fprintf('Total requested : %d\n', total_combinations);
fprintf('Total attempted : %d\n', sweep_counter);
fprintf('Successful      : %d\n', num_success);
fprintf('Failed          : %d\n', num_failed);
fprintf('Total duration  : %.2f sec (%.2f hours)\n', overall_duration, overall_duration/3600);
fprintf('====================================================================\n');

% Save MATLAB log.
sweep_log_mat = fullfile(mainFolder, 'sweep_log.mat');
save(sweep_log_mat, 'sweep_log', 'sweep_plan', 'phi_initial', ...
    'total_beads', 'chains_list', 'ranks');
fprintf('MAT log saved to: %s\n', sweep_log_mat);

% Save easy-to-read text summary.
sweep_log_txt = fullfile(mainFolder, 'sweep_log.txt');
fid = fopen(sweep_log_txt, 'w');
if fid < 0
    error('runSAWNetwork:LogOpenFail','Cannot open summary log: %s', sweep_log_txt);
end

fprintf(fid, '2026 Entanglement Study Sweep Summary\n');
fprintf(fid, '=====================================\n');
fprintf(fid, 'Initial SAW phi: %.10g\n', phi_initial);
fprintf(fid, 'MPI ranks/run  : %d\n', ranks);
fprintf(fid, 'Total duration : %.2f seconds (%.2f hours)\n', ...
    overall_duration, overall_duration/3600);
fprintf(fid, 'Success rate   : %.1f%% (%d/%d)\n\n', ...
    100*num_success/max(sweep_counter,1), num_success, sweep_counter);

fprintf(fid, 'Sweep plan\n');
fprintf(fid, 'phi_target  fact        tension  compression  pure_shear\n');
for s_idx = 1:size(sweep_plan,1)
    pt = sweep_plan(s_idx,1);
    ff = (phi_initial/pt)^(1/3);
    fprintf(fid, '%-10.5g %-11.7f %-8d %-12d %-10d\n', ...
        pt, ff, sweep_plan(s_idx,2), sweep_plan(s_idx,3), sweep_plan(s_idx,4));
end

fprintf(fid, '\nRun  phi_i   phi_t   fact       mode          iter  seed      status  duration_s  fileTag\n');
for i = 1:numel(sweep_log)
    if sweep_log(i).status
        status_str = 'PASS';
    else
        status_str = 'FAIL';
    end
    fprintf(fid, '%-4d %-7.3g %-7.3g %-10.6f %-13s %-5d %-9d %-7s %-11.2f %s\n', ...
        sweep_log(i).run, ...
        sweep_log(i).phi_initial, ...
        sweep_log(i).phi_target, ...
        sweep_log(i).swelling_factor, ...
        sweep_log(i).deformation_mode, ...
        sweep_log(i).mode_iteration, ...
        sweep_log(i).random_seed, ...
        status_str, ...
        sweep_log(i).duration_sec, ...
        sweep_log(i).fileTag);

    if ~sweep_log(i).status && ~isempty(sweep_log(i).error_message)
        fprintf(fid, '     ERROR: %s\n', sweep_log(i).error_message);
    end
end
fclose(fid);
fprintf('Text log saved to: %s\n', sweep_log_txt);

fprintf('\nSweep finished at: %s\n', char(datetime('now')));

% Propagate any failed simulation to MATLAB/SLURM.
if num_failed > 0
    error('runSAWNetwork:FailedRuns', ...
        '%d simulation(s) failed. SLURM job is marked failed.', num_failed);
end
