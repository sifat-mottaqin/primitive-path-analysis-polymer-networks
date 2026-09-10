function ok = runLammpsFromSAW_HPC(lammps_exe, fileTag, mainFolder, varargin)
% ok = runLammpsFromSAW_HPC(lammps_exe, fileTag, mainFolder, ...)
%
% HPC-only LAMMPS launcher.
% This version REQUIRES MPI (>= 2 ranks) and launches LAMMPS with the exact
% OpenMPI environment saved by run_DNM_v2.sh before MATLAB is loaded.
%
% Required environment variables from run_DNM_v2.sh:
%   LAMMPS_MPIRUN
%   LAMMPS_LD_LIBRARY_PATH
%   PATH_BEFORE_MATLAB
%
% The actual command is:
%   mpirun -np <Ranks> <LAMMPS executable> -in input_<tag>.in

ok = false;

%% Parse options
opt = struct('Parallel',true, 'Ranks',32, 'UseGPU',false, 'GPUs',1);
for k = 1:2:numel(varargin)
    key = lower(varargin{k});
    val = varargin{k+1};
    switch key
        case 'parallel'
            opt.Parallel = logical(val);
        case 'ranks'
            opt.Ranks = double(val);
        case 'usegpu'
            opt.UseGPU = logical(val);
        case 'gpus'
            opt.GPUs = double(val);
        otherwise
            error('runLammpsFromSAW_HPC:BadOption', 'Unknown option "%s"', key);
    end
end

if ~opt.Parallel || ~isfinite(opt.Ranks) || opt.Ranks < 2
    error('runLammpsFromSAW_HPC:MPIRequired', ...
        'This package requires MPI. Parallel must be true and Ranks must be >= 2.');
end
opt.Ranks = round(opt.Ranks);

if ~exist(lammps_exe,'file')
    error('runLammpsFromSAW_HPC:MissingLAMMPS', ...
        'LAMMPS executable not found: %s', lammps_exe);
end

%% Recover clean MPI environment saved BEFORE MATLAB was loaded
mpiLauncher = getenv('LAMMPS_MPIRUN');
cleanLD     = getenv('LAMMPS_LD_LIBRARY_PATH');
cleanPATH   = getenv('PATH_BEFORE_MATLAB');

if isempty(mpiLauncher)
    error('runLammpsFromSAW_HPC:MissingMPI', ...
        'LAMMPS_MPIRUN is empty. Submit through run_DNM_v2.sh.');
end
if isempty(cleanLD)
    error('runLammpsFromSAW_HPC:MissingLDPath', ...
        'LAMMPS_LD_LIBRARY_PATH is empty. Submit through run_DNM_v2.sh.');
end
if isempty(cleanPATH)
    error('runLammpsFromSAW_HPC:MissingPATH', ...
        'PATH_BEFORE_MATLAB is empty. Submit through run_DNM_v2.sh.');
end

%% GPU flags (normally disabled in this CPU package)
gpuFlags = '';
if opt.UseGPU
    gpuFlags = sprintf('-sf gpu -pk gpu %d', opt.GPUs);
end

%% Paths
inpDir  = fullfile(mainFolder,'Input');
outDir  = fullfile(mainFolder,'Output','Dump');
logDir  = fullfile(mainFolder,'Output','Logs');

if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(logDir,'dir'), mkdir(logDir); end

inpName = sprintf('input_%s.in', fileTag);
inpPath = fullfile(inpDir, inpName);
if ~exist(inpPath,'file')
    error('runLammpsFromSAW_HPC:MissingInput', ...
        'LAMMPS input file not found: %s', inpPath);
end

atomName = sprintf('atoms_%s.dump', fileTag);
bondName = sprintf('bonds_%s.dump', fileTag);
logName  = sprintf('log_%s.lammps', fileTag);
screenName = sprintf('screen_%s.txt', fileTag);

atomDumpInRun = fullfile(inpDir, atomName);
bondDumpInRun = fullfile(inpDir, bondName);
logFileInRun  = fullfile(inpDir, logName);
screenFileInRun = fullfile(inpDir, screenName);

safeDelete(atomDumpInRun);
safeDelete(bondDumpInRun);
safeDelete(logFileInRun);
safeDelete(screenFileInRun);

%% Build a wrapper that restores the clean module environment
wrapper_script = fullfile(inpDir, '.run_lammps_wrapper.sh');
fid = fopen(wrapper_script, 'w');
if fid < 0
    error('runLammpsFromSAW_HPC:WrapperOpenFail', ...
        'Cannot create MPI wrapper: %s', wrapper_script);
end

fprintf(fid, '#!/bin/bash\n');
fprintf(fid, 'set -e\n');
fprintf(fid, 'export LD_LIBRARY_PATH="%s"\n', cleanLD);
fprintf(fid, 'export PATH="%s"\n', cleanPATH);
fprintf(fid, 'export OMP_NUM_THREADS=1\n');
fprintf(fid, 'export OMP_DYNAMIC=false\n');
fprintf(fid, 'echo "MPI launcher: %s"\n', mpiLauncher);
fprintf(fid, 'echo "MPI ranks: %d"\n', opt.Ranks);
fprintf(fid, 'echo "LAMMPS executable: %s"\n', lammps_exe);

if isempty(gpuFlags)
    fprintf(fid, '"%s" -np %d "%s" -log "%s" -screen "%s" -in "%s"\n', ...
        mpiLauncher, opt.Ranks, lammps_exe, logName, screenName, inpName);
else
    fprintf(fid, '"%s" -np %d "%s" %s -log "%s" -screen "%s" -in "%s"\n', ...
        mpiLauncher, opt.Ranks, lammps_exe, gpuFlags, logName, screenName, inpName);
end
fclose(fid);

cmd = sprintf('cd "%s" && bash "%s"', inpDir, wrapper_script);

fprintf('[LAMMPS] MPI launcher : %s\n', mpiLauncher);
fprintf('[LAMMPS] MPI ranks    : %d\n', opt.Ranks);
fprintf('[LAMMPS] Executable   : %s\n', lammps_exe);
fprintf('[LAMMPS] Run directory: %s\n', inpDir);
fprintf('[LAMMPS] Starting MPI simulation for %s ...\n', fileTag);

%% Run
lmp_start = tic;
[status, cmdout] = system(cmd);
lmp_duration = toc(lmp_start);

fprintf('[LAMMPS] MPI launch finished in %.2f sec (status=%d)\n', ...
    lmp_duration, status);
if ~isempty(strtrim(cmdout))
    fprintf('[LAMMPS] Launcher output:\n%s\n', cmdout);
end

% Keep LAMMPS logs whether the simulation succeeds or fails.
safeMove(logFileInRun, logDir);
safeMove(screenFileInRun, logDir);
safeDelete(wrapper_script);

if status ~= 0
    fprintf('[LAMMPS] ERROR: MPI LAMMPS simulation failed (status=%d).\n', status);
    fprintf('[LAMMPS] Check Output/Logs/screen_%s.txt and log_%s.lammps\n', ...
        fileTag, fileTag);
    cleanup_partials();
    return
end

%% Verify required output
if ~exist(atomDumpInRun,'file') || ~exist(bondDumpInRun,'file')
    fprintf('[LAMMPS] ERROR: LAMMPS returned status 0 but required dumps are missing.\n');
    fprintf('[LAMMPS] atoms dump exists: %d\n', exist(atomDumpInRun,'file'));
    fprintf('[LAMMPS] bonds dump exists: %d\n', exist(bondDumpInRun,'file'));
    cleanup_partials();
    return
end

safeMove(atomDumpInRun, outDir);
safeMove(bondDumpInRun, outDir);

fprintf('[LAMMPS] SUCCESS: %s completed using %d MPI ranks.\n', ...
    fileTag, opt.Ranks);
fprintf('[LAMMPS] Dumps -> Output/Dump/\n');
fprintf('[LAMMPS] Logs  -> Output/Logs/\n');
ok = true;

    function cleanup_partials()
        safeDelete(atomDumpInRun);
        safeDelete(bondDumpInRun);
    end
end

function safeMove(src, dstDir)
if exist(src,'file')
    [~,name,ext] = fileparts(src);
    dst = fullfile(dstDir, [name ext]);
    if exist(dst,'file'), delete(dst); end
    movefile(src, dst);
end
end

function safeDelete(p)
if exist(p,'file')
    delete(p);
end
end
