function RunSimulation(Package,Override,ToggleDynamics,LC,DC,BSOM,CF,OF,BT,...
    LAMMPSExecutablePath)

global InputFileName OutputAtom_loc OutputBond_loc ...
    OutputAtom OutputBond OutputSuccess TurnOnDynamics...
    LengthConversion DamperConversion BeadSpringOrMeso CurrentFolder OutputFolder...
    OutputDir BondType RawDataFileName ForceDataFileName TimeStretchDataFileName

TurnOnDynamics = ToggleDynamics;
LengthConversion = LC;
DamperConversion = DC;
BeadSpringOrMeso = BSOM;
CurrentFolder = CF;
OutputFolder = OF;
BondType = BT;

%% Unpack Swept Input Parameters
Sample = Package(1);    % Sample index
Np = Package(2);        % Number of molecules
D = Package(3);         % Diffusion coefficient [m2/s]
N_Kuhn = Package(4);    % Number of Kuhn segments in chain
stiffness = Package(5); % Stiffness of single harmonic bond
kbT = Package(6);       % Thermal energy
b = Package(7);         % Kuhn length
dt = Package(8);        % timestep

%% Initialize filenames
InputScript(Sample,Np,D,N_Kuhn,stiffness,kbT,b,dt);
SetDirAndFileNames;

%% Run only when needed
NeedRun = Override==1 || ~isfile(OutputAtom) || ~isfile(OutputBond) || ~isfile(OutputSuccess);
if NeedRun
    % Remove stale outputs BEFORE rerunning.  This prevents a failed rerun
    % from being mistaken for a successful old run.
    FilesToDelete = {OutputAtom,OutputBond,OutputSuccess,OutputAtom_loc,OutputBond_loc,RawDataFileName};
    for i=1:numel(FilesToDelete)
        if isfile(FilesToDelete{i})
            delete(FilesToDelete{i});
        end
    end

    % These ensemble files are shared by all samples of this parameter set.
    % Delete them now so CompileData must rebuild them from the current set
    % of successful runs.
    if isfile(ForceDataFileName); delete(ForceDataFileName); end
    if isfile(TimeStretchDataFileName); delete(TimeStretchDataFileName); end

    command = ['wsl ',LAMMPSExecutablePath,' -in ',InputFileName];
    [status,cmdout] = system(command);

    % A case is successful only when LAMMPS exits cleanly AND both complete
    % pull-stage dump files were actually produced.
    atomOK = isfile(OutputAtom_loc);
    bondOK = isfile(OutputBond_loc);
    if atomOK
        tmp = dir(OutputAtom_loc);
        atomOK = ~isempty(tmp) && tmp.bytes>0;
    end
    if bondOK
        tmp = dir(OutputBond_loc);
        bondOK = ~isempty(tmp) && tmp.bytes>0;
    end

    RunSucceeded = (status==0) && atomOK && bondOK;

    if RunSucceeded
        [okA,msgA] = movefile(OutputAtom_loc,OutputDir);
        [okB,msgB] = movefile(OutputBond_loc,OutputDir);
        if okA && okB
            fid = fopen(OutputSuccess,'wt');
            fprintf(fid,'LAMMPS run completed successfully.\n');
            fclose(fid);
            fprintf('SUCCESS: sample %d completed and will be included in force averaging.\n',Sample);
        else
            if isfile(OutputAtom); delete(OutputAtom); end
            if isfile(OutputBond); delete(OutputBond); end
            warning('Sample %d finished LAMMPS but output move failed. Atom: %s Bond: %s',...
                Sample,msgA,msgB);
        end
    else
        % Never preserve partial dumps from a failed case.
        if isfile(OutputAtom_loc); delete(OutputAtom_loc); end
        if isfile(OutputBond_loc); delete(OutputBond_loc); end
        if isfile(OutputAtom); delete(OutputAtom); end
        if isfile(OutputBond); delete(OutputBond); end
        if isfile(OutputSuccess); delete(OutputSuccess); end

        warning('LAMMPS FAILED for sample %d (status %d). This case will be skipped during force compilation.\n%s',...
            Sample,status,cmdout);
    end
end

clear global

end
