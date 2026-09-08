function runTwoChains
% Main driver: generates topology & input, runs LAMMPS, extracts data.
% - Random N1,N2 from [20,100] each run (no saved plan)
% - Fixed count of target_r per case; values i.i.d. U[0.10,0.90], 2 decimals
% - Topology tag excludes target_r; per-target files include target_r
% - SLURM array uses legacy linearized iter mapping (one array id -> one (case,target))
% - In ARRAY mode, topology is generated ONLY for the selected (case,target) to avoid side effects

clear; close all; clc; rng(29); %29

%% Globals Parameters
global b tau0 dt K eps gamma nhold npull

%% Control Switches 
showWaitbar              = 0;

create_folders           = 1;
generate_input_files     = 1;
generate_topology_files  = 1;
run_lammps_simulation    = 1;
run_data_extract         = 1;

% Override switches
override_folders           = 0;
override_input_files       = 0;
override_topology_files    = 0;
override_lammps_simulation = 0;
override_data_extract      = 0;

%% Folders 
mainFolder = '/data/home/bsifatmottaq/twoChainsML';

if create_folders
    subDirs = {'Output','Input','Output/Dump','Output/RawData'};
    for k = 1:numel(subDirs)
        d = fullfile(mainFolder,subDirs{k});
        if ~exist(d,'dir'); mkdir(d); end
    end
end

%% LAMMPS run config (CPU-serial by default) 
useGPU        = 0;
parallelRun   = 0;
ranks         = 1;
gpusPerTask   = 1;
srunGPUFlags  = '--gpus-per-task=1';
ldPath        = getenv('LMP_LD_PATH');
lmpExtra      = '';

if useGPU
    lammps_exe = '/data/home/bsifatmottaq/lammps/gpu_install/bin/lmp';
else
    lammps_exe = '/data/home/bsifatmottaq/lammps/cpu_install/bin/lmp';
end

%%  Sweep parameters 
numPairs       = 100;    % how many random (N1,N2) pairs this run builds
targetsPerCase = 10;     % fixed number of target_r per case

Nmin           = 20;
Nmax           = 100;
rhoVals        = 0.1;   % can be vector
runsPerCase    = 10;     % repeats per (N1,N2,rho)

%%  Simulation variables 
b      = 1;				% Kuhn's length
tau0   = 1e-8;			% Diffusion Timescale
dt     = 1e-5;			% Time Step
K      = 800;			% FENE Spring Const
nhold  = 1e8;			% Relaxtion Timestep
npull  = 1e6;			% Relaxtion Timestep during affine defromation
eps    = 1.0;			% LJ Interaction Scale
gamma  = 1;				% Damping Co-efficient

%% Paths used repeatedly 
inpDir   = fullfile(mainFolder,'Input');
outRaw   = fullfile(mainFolder,'Output','RawData');
if ~exist(inpDir,'dir'),  mkdir(inpDir);  end
if ~exist(outRaw,'dir'),  mkdir(outRaw);  end

%% Build cases fresh (no persistence) 
pairN1 = randi([Nmin Nmax], numPairs, 1);
pairN2 = randi([Nmin Nmax], numPairs, 1);

cases = {};
idx = 0;
for p = 1:numPairs
    for rv = 1:numel(rhoVals)
        for R = 1:runsPerCase
            idx = idx + 1;
            N1 = pairN1(p); N2 = pairN2(p); rho = rhoVals(rv);

            % Per-case fixed count of targets; values are i.i.d. and rounded
            tvals = 0.10 + (0.90-0.10).*rand(targetsPerCase,1); %  tvals = 0.10 + (0.90-0.10).*rand(targetsPerCase,1);
            tvals = round(tvals, 2);

            % Stable per-case RNG seed (keeps coords stable within this run)
            baseTag = sprintf('%d_%d_%d_%d', N1, N2, round(rho*100), R);
            seed = seedFromTag(baseTag);

            cases{idx,1} = struct( ...
                'caseIdx', idx, 'pairIdx', p, ...
                'N1', N1, 'N2', N2, 'rho', rho, 'run', R, ...
                'seed', seed, 'target_r_list', tvals(:)' );
        end
    end
end
plan = struct('numCases', numel(cases), 'cases', {cases});
totalCases = plan.numCases;
targetsPerCase_fixed = targetsPerCase;

%% Array / local selection 
caseStart = str2double(getenv('CASE_START')); if isnan(caseStart), caseStart = 1; end
caseEnd   = str2double(getenv('CASE_END'));   if isnan(caseEnd),   caseEnd   = totalCases; end
caseStart = max(1, caseStart);
caseEnd   = min(totalCases, caseEnd);

arrayIdStr = getenv('SLURM_ARRAY_TASK_ID');
if ~isempty(arrayIdStr)
    arrayId = str2double(arrayIdStr);
    if isnan(arrayId) || arrayId < 1, arrayId = 1; end
    totalIter = totalCases * targetsPerCase_fixed;
    if arrayId > totalIter
        fprintf('Array task %d > total targets %d — nothing to do. Exiting.\n', arrayId, totalIter);
        return
    end
    arrayMode = true;
else
    arrayMode = false;
    arrayId   = 0;
end

if showWaitbar && ~arrayMode
    h = waitbar(0,'Running selected cases ...');
end

if arrayMode
    fprintf('Starting array mode (legacy iter mapping) ... total targets=%d, this task=%d\n', ...
        totalCases * targetsPerCase_fixed, arrayId);
else
    fprintf('Starting local mode ... (total cases: %d; running cases %d..%d)\n', ...
        totalCases, caseStart, caseEnd);
end
tic;

%% Main loop 
iter = 0;  % linearized counter: increments once per (case, target_r)

for cIdx = 1:totalCases

    % Local slicing only in non-array mode
    if ~arrayMode && (cIdx < caseStart || cIdx > caseEnd)
        continue
    end

    C = plan.cases{cIdx};
    N1 = C.N1; N2 = C.N2; rhoVal = C.rho; R = C.run; targetRvals = C.target_r_list;

    rng(C.seed,'twister'); % reproducible coords within this run

    % Tags: caseTag has NO target_r; per-target tags include it
    caseTag    = sprintf('%d_%d_%d_%d', N1, N2, round(rhoVal*100), R);
    topoOut    = fullfile(inpDir,  sprintf('topology_%s.txt', caseTag));

    mkTagT     = @(tr) sprintf('%s_%d', caseTag, round(tr*100));
    rawOut     = @(tr) fullfile(outRaw,  sprintf('raw_%s.mat',   mkTagT(tr)));
    inputOut   = @(tr) fullfile(inpDir,  sprintf('input_%s.in',  mkTagT(tr)));

    % Generate coordinates for this case 
    [~, coords, dim] = generateCoordinatesSAW(N1, N2, rhoVal);

    % Topology generation placement
    if ~arrayMode
        % LOCAL MODE: generate once per case before per-target loop
        if generate_topology_files
            if exist(topoOut,'file')
                if override_topology_files
                    fprintf('Topology exists; override=1 -> regenerating (%s)\n', caseTag);
                    generateTopology(coords, N1, N2, caseTag, mainFolder, dim);
                else
                    fprintf('Topology exists; override=0 -> skipping (%s)\n', caseTag);
                end
            else
                fprintf('Topology missing -> generating (%s)\n', caseTag);
                generateTopology(coords, N1, N2, caseTag, mainFolder, dim);
            end
        end
    end

    % Per-target work
    for tr = 1:numel(targetRvals)
        target_r     = targetRvals(tr);
        fileTagT     = mkTagT(target_r);
        inputOutPath = inputOut(target_r);
        rawOutPath   = rawOut(target_r);

        % advance the linearized iterator
        iter = iter + 1;

        % ARRAY MODE: only execute selected (case,target) when iter matches arrayId
        if arrayMode && iter ~= arrayId
            if showWaitbar && ~arrayMode
                waitbar(min(1, iter/(totalCases*targetsPerCase_fixed)));
            end
            continue
        end

        % ARRAY MODE: generate topology ONLY now (avoid touching other cases)
        if arrayMode && generate_topology_files
            if exist(topoOut,'file')
                if override_topology_files
                    fprintf('Topology exists; override=1 -> regenerating (%s)\n', caseTag);
                    generateTopology(coords, N1, N2, caseTag, mainFolder, dim);
                else
                    fprintf('Topology exists; override=0 -> skipping (%s)\n', caseTag);
                end
            else
                fprintf('Topology missing -> generating (%s)\n', caseTag);
                generateTopology(coords, N1, N2, caseTag, mainFolder, dim);
            end
        end

        fprintf('[case %d] N=(%d,%d) rho=%.2f R=%d  target_r=%.2f  (iter %d)\n', ...
            cIdx, N1, N2, rhoVal, R, target_r, iter);

        % INPUT
        if generate_input_files
            if exist(inputOutPath,'file')
                if override_input_files
                    fprintf('Input exists; override=1 -> regenerating (%s)\n', fileTagT);
                    generateInput(N1, N2, fileTagT, caseTag, mainFolder, dim, target_r, coords);
                else
                    fprintf('Input exists; override=0 -> skipping (%s)\n', fileTagT);
                end
            else
                fprintf('Input missing -> generating (%s)\n', fileTagT);
                generateInput(N1, N2, fileTagT, caseTag, mainFolder, dim, target_r, coords);
            end
        end

        % LAMMPS (resume via RAW only)
        needWork = ~( exist(rawOutPath,'file') && ...
                      ~override_lammps_simulation && ...
                      ~override_data_extract );
        ok = true;
        if run_lammps_simulation && needWork
            ok = runLammpsSimulation( ...
                lammps_exe, fileTagT, mainFolder, ...
                'Parallel', parallelRun, 'Ranks', ranks, ...
                'UseGPU', useGPU, 'GPUs', gpusPerTask, ...
                'SrunGPU', srunGPUFlags, ...
                'LDPath', ldPath, 'Extra', lmpExtra);
        end

        % EXTRACT
        if ok && run_data_extract
            if exist(rawOutPath,'file')
                if override_data_extract
                    fprintf('Raw MAT exists; override=1 -> re-extracting (%s)\n', fileTagT);
                    runDataExtract(N1, N2, fileTagT, mainFolder);
                else
                    fprintf('Raw MAT exists; override=0 -> skipping (%s)\n', fileTagT);
                end
            else
                fprintf('Raw MAT missing -> extracting (%s)\n', fileTagT);
                runDataExtract(N1, N2, fileTagT, mainFolder);
            end
        else
            if ~ok
                fprintf('Skipped extract for %s — LAMMPS failed.\n', fileTagT);
            end
        end

        % In array mode we only ever wanted one iter; return immediately after it
        if arrayMode
            fprintf('Task %d completed target %s (caseIdx=%d).\n', arrayId, fileTagT, cIdx);
            if showWaitbar && exist('h','var') && isvalid(h), close(h); end
            toc; return
        end

        if showWaitbar && ~arrayMode
            waitbar(min(1, iter/(totalCases*targetsPerCase_fixed)));
        end
    end
end

if showWaitbar && exist('h','var') && isvalid(h), close(h); end
toc;
end

%% Helper function for seedFromTag 
function s = seedFromTag(tag)
h = 5381.0;
for k = 1:numel(tag)
    h = mod(h*33.0 + double(tag(k)), 2^31 - 1);
end
if h < 1, h = 1; end
s = h;
end
