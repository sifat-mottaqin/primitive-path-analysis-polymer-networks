function ok = runLammpsSimulation(lammps_exe,fileTag,mainFolder,varargin)
% runs LAMMPS on input_FILETAG.in, moves dump files, deletes log.lammps
% Works on:
%   - Windows: via WSL (single-rank or mpirun if available)
%   - Linux/HPC: direct or srun under Slurm (CPU/GPU; parallel on/off)
%
% Name-value options:
%   'Parallel'  true/false   [default: true if inside Slurm, else false]
%   'Ranks'     integer      [default: 8]
%   'UseGPU'    true/false   [default: false]
%   'GPUs'      integer      [default: 1]         % for -pk gpu <GPUs>
%   'SrunGPU'   string       [default: '']        % e.g. '--gpus-per-task=1' or '--gres=gpu:1'
%   'Extra'     string       [default: '']        % additional flags after -in <input>
%   'LDPath'    string       [default: '']        % exported as LD_LIBRARY_PATH for the run
%
% Returns:
%   ok = true  if dumps were produced and moved successfully
%        false otherwise (and cleans up partial outputs)

ok = false;

% ----- parse options -----
inSlurm = ~isempty(getenv('SLURM_JOB_ID'));
opt = struct('Parallel',[], 'Ranks',8, 'UseGPU',false, 'GPUs',1, ...
             'SrunGPU','', 'Extra','', 'LDPath','');
for k = 1:2:numel(varargin)
    key = lower(varargin{k});
    val = varargin{k+1};
    switch key
        case 'parallel', opt.Parallel = logical(val);
        case 'ranks',    opt.Ranks    = val;
        case 'usegpu',   opt.UseGPU   = logical(val);
        case 'gpus',     opt.GPUs     = val;
        case 'srungpu',  opt.SrunGPU  = val;
        case 'extra',    opt.Extra    = val;
        case 'ldpath',   opt.LDPath   = val;
        otherwise
            error('Unknown option "%s"', varargin{k});
    end
end
if isempty(opt.Parallel)
    opt.Parallel = inSlurm; % default: parallel inside Slurm
end
if isnan(opt.Ranks) || opt.Ranks<=0, opt.Ranks = 8; end

% GPU flags (only if requested)
gpuFlags = '';
if opt.UseGPU
    gpuFlags = sprintf('-sf gpu -pk gpu %d ', opt.GPUs);
end

% ----- paths -----
inpDir   = fullfile(mainFolder,'Input');
outDir   = fullfile(mainFolder,'Output','Dump');
if ~exist(outDir,'dir'); mkdir(outDir); end

inpName  = sprintf('input_%s.in',fileTag);
inpPath  = fullfile(inpDir,inpName);
if ~exist(inpPath,'file')
    fprintf('Skipping – input file not found: %s\n',inpName);
    return
end

atomDump = fullfile(inpDir, sprintf('atoms_%s.dump',fileTag));
bondDump = fullfile(inpDir, sprintf('bonds_%s.dump',fileTag));
logFile  = fullfile(inpDir, 'log.lammps');

    function cleanup_partials()
        safeDelete(atomDump);
        safeDelete(bondDump);
        safeDelete(logFile);
    end

% ----- build command -----
isWindows = ispc;
if isWindows
    linuxDir = win2lin(inpDir);
    core = sprintf('%s %s -in %s %s', lammps_exe, gpuFlags, inpName, opt.Extra);
    if opt.Parallel && opt.Ranks>1
        core = sprintf('mpirun -np %d %s', opt.Ranks, core);
    end
    cmd = sprintf('wsl bash -c "cd %s && %s"', linuxDir, core);
else
    if inSlurm
        if opt.Parallel && opt.Ranks>1
            if isempty(opt.LDPath)
                cmd = sprintf('cd "%s" && srun --ntasks=%d %s %s %s -in %s %s', ...
                              inpDir, opt.Ranks, opt.SrunGPU, lammps_exe, gpuFlags, inpName, opt.Extra);
            else
                cmd = sprintf(['cd "%s" && srun --ntasks=%d %s env LD_LIBRARY_PATH="%s" ' ...
                               '%s %s -in %s %s'], ...
                               inpDir, opt.Ranks, opt.SrunGPU, opt.LDPath, ...
                               lammps_exe, gpuFlags, inpName, opt.Extra);
            end
        else
            % single-rank inside Slurm allocation
            if isempty(opt.LDPath)
                cmd = sprintf('cd "%s" && %s %s -in %s %s', ...
                               inpDir, lammps_exe, gpuFlags, inpName, opt.Extra);
            else
                cmd = sprintf('cd "%s" && env LD_LIBRARY_PATH="%s" %s %s -in %s %s', ...
                               inpDir, opt.LDPath, lammps_exe, gpuFlags, inpName, opt.Extra);
            end
        end
    else
        % no Slurm allocation
        core = sprintf('%s %s -in %s %s', lammps_exe, gpuFlags, inpName, opt.Extra);
        if opt.Parallel && opt.Ranks>1
            core = sprintf('mpirun -np %d %s', opt.Ranks, core);
        end
        if isempty(opt.LDPath)
            cmd = sprintf('cd "%s" && %s', inpDir, core);
        else
            cmd = sprintf('cd "%s" && env LD_LIBRARY_PATH="%s" %s', inpDir, opt.LDPath, core);
        end
    end
end

% ----- run -----
fprintf('Running LAMMPS for %s\n',fileTag);
[status,cmdout] = system(cmd);

if status ~= 0
    fprintf('LAMMPS failed for %s (status=%d)\n%s\n',fileTag,status,cmdout);
    cleanup_partials();
    return
end

if ~exist(atomDump,'file') || ~exist(bondDump,'file')
    fprintf(['LAMMPS ended without required dump files for %s. ' ...
        '(atoms exists? %d, bonds exists? %d)\n'], ...
        fileTag, exist(atomDump,'file'), exist(bondDump,'file'));
    cleanup_partials();
    return
end

% ----- success: move dumps to Output/Dump, remove log -----
safeMove(atomDump, outDir);
safeMove(bondDump, outDir);
if exist(logFile,'file'); delete(logFile); end

fprintf('Simulation %s finished (dumps moved).\n',fileTag);
ok = true;
end

% ======================= helpers =======================
function pathLinux = win2lin(pathWin)
drv = lower(pathWin(1));
pathLinux = ['/mnt/' drv pathWin(3:end)];
pathLinux = strrep(pathLinux,'\','/');
end

function safeMove(src,dstDir)
if exist(src,'file'), movefile(src,dstDir); end
end

function safeDelete(p)
if exist(p,'file'), delete(p); end
end
