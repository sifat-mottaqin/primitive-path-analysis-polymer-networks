function ParallelRunScript

clear
clear global
clc
fclose all;
close all
close all hidden

%% Unit conversions
LengthConversion = 3.74e-9;     % meters per unit code length
DamperConversion = 2.89e-4;     % N*s/m per unit damper in code
ForceConversion = 1.08e-12;     % N per unit force in code

%% Work flow controls
% Bead-spring force-vs-r* workflow only
GenerateTopologies = 1;     % 1 to generate topology/input files
RunLAMMPS = 1;              % 1 to run LAMMPS simulations
CompileEnsembleData = 1;    % 1 to parse raw data and compute force + r*
PostProcess = 1;            % 1 to generate force-vs-r* CSV/plot

ToggleDynamics = 0;         % 1 for dynamics, 0 for control systems (DO NOT TOUCH)

% Overrides
OverrideTopologies = 0;     % 1 to override initiated topologies
OverrideInputScripts = 0;   % 1 to rewrite input scripts
OverrideRun = 0;            % 1 to rerun simulations
OverrideCompile = 1;        % 1 to override parsing of LAMMPS outputs
OverrideCompute = 1;        % 1 to override force/r* computation

%% Directory and processor control
NoProcessors = 1;
CurrentFolder = DefineCurrentFolder;   % PWD only; ends with /
OutputFolder = CurrentFolder;          % all outputs remain under PWD
LAMMPSExecutablePath = '/mnt/c/Users/bsifatmottaq/Documents/LAMMPS/CPU_PARA_BUILD/lammps/build_mpi/lmp';

%% Only the outputs needed for force vs r*
CalculateForces = 1;
CalculateEndtoEnd = 0;
CalculateAlignmentt = 0;

%% Bead-spring model only
BeadSpringOrMeso = 0;
CompareModels = 0;

%% Input Parameters
Np = 1;
b = 0.1667;                 % Kuhn length
kbT = 293*1.38e-23;         % thermal energy [J]

%% Sweeping Parameters
b_SI = b*LengthConversion;  % m
N_Kuhn = 20;                % N = 20 only
D_nom = 1e-10;              % nominal diffusion coefficient of a monomer [m^2/s]
D = D_nom*10.^[-2 0];       % m^2/s
tau0 = (b_SI^2)./D;         % s
damps = kbT./D;             % kg/s or N*s/m
damps = damps/DamperConversion;
dtFact = 320;
dt = 1/dtFact*min(tau0);
D = D(1);
damps = damps(1); 

Stiffnesses_SI = [800]*kbT/b_SI^2;
Stiffnesses = Stiffnesses_SI/kbT*b_SI^2;
BondType = 1;               % 0 harmonic, 1 nonlinear
Samples = 1:20;
NoParam = 8;

%% Compile Input Parameters
Perms = length(N_Kuhn)*length(D)*length(Stiffnesses)*length(Samples);
Package = zeros(Perms,NoParam);
ct = 0;
for i=1:length(Samples)
    for j=1:length(Np)
        for k=1:length(D)
            for l=1:length(N_Kuhn)
                for m=1:length(Stiffnesses)
                    ct = ct+1;
                    Package(ct,1) = Samples(i);
                    Package(ct,2) = Np(j);
                    Package(ct,3) = D(k);
                    Package(ct,4) = N_Kuhn(l);
                    Package(ct,5) = Stiffnesses(m);
                    Package(ct,6) = kbT;
                    Package(ct,7) = b;
                    Package(ct,8) = dt;
                end
            end
        end
    end
end

%% Generate topologies and LAMMPS input files
if GenerateTopologies==1
    if NoProcessors>1
        delete(gcp('nocreate'))
        parpool(NoProcessors)
        parfor n=1:size(Package,1)
            GenerateTopology(Package(n,:),OverrideTopologies,...
                OverrideInputScripts,CurrentFolder,OutputFolder,...
                LengthConversion,DamperConversion,...
                BeadSpringOrMeso,dtFact,BondType);
        end
        delete(gcp('nocreate'))
    else
        wb2 = waitbar(0,'Generating all input scripts...');
        for m1=1:size(Package,1)
            GenerateTopology(Package(m1,:),OverrideTopologies,...
                OverrideInputScripts,CurrentFolder,OutputFolder,...
                LengthConversion,DamperConversion,...
                BeadSpringOrMeso,dtFact,BondType);
            waitbar(m1/size(Package,1),wb2,'Generating all input scripts...')
        end
        close(wb2)
    end
end

%% Run LAMMPS
if RunLAMMPS==1
    wb = waitbar(0,'Running all jobs...');
    for m1=1:size(Package,1)
        RunSimulation(Package(m1,:),OverrideRun,ToggleDynamics,...
            LengthConversion,DamperConversion,BeadSpringOrMeso,...
            CurrentFolder,OutputFolder,...
            BondType,LAMMPSExecutablePath);
        waitbar(m1/size(Package,1),wb,'Running all jobs...')
    end
    close(wb)
end

%% Compile only the quantities needed for force vs r*
if CompileEnsembleData==1
    CompileData(Package,OverrideCompile,OverrideCompute,...
        LengthConversion,DamperConversion,BeadSpringOrMeso,...
        CalculateForces,CalculateEndtoEnd,...
        CalculateAlignmentt,CurrentFolder,OutputFolder,...
        NoProcessors,BondType);
end

%% Force vs r* only
if PostProcess==1
    PostProcessData(Package,ToggleDynamics,...
        LengthConversion,DamperConversion,ForceConversion,...
        BeadSpringOrMeso,CompareModels,...
        CurrentFolder,OutputFolder,BondType);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function CurrentFolder = DefineCurrentFolder

CurrentFolder = pwd;
CurrentFolder = strrep(CurrentFolder, '\', '/');
if CurrentFolder(end) ~= '/'
    CurrentFolder = [CurrentFolder,'/'];
end

end
