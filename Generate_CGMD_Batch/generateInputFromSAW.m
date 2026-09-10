function generateInputFromSAW(S, fileTag, mainFolder, deformationMode, phiTarget)
% generateInputFromSAW

% ONE LAMMPS-input generator for all three loading modes:
%   1) tension      : incompressible uniaxial tension in x
%   2) compression  : incompressible uniaxial compression in x
%   3) pure_shear   : incompressible pure shear / planar extension
%
% The SAW is generated at S.meta.phi (= phiInitial).  Before mechanical
% loading the box is swollen isotropically to phiTarget using
%
%       fact = (phiInitial / phiTarget)^(1/3)
%
% so that, nominally,
%
%       phiTarget = phiInitial / fact^3.
%
% Writes: Input/input_<fileTag>.in
% Uses  : Input/topology_<fileTag>.txt
% Dumps : Output/Dump/atoms_<fileTag>.dump
%         Output/Dump/bonds_<fileTag>.dump


if nargin < 5
    error('generateInputFromSAW:NotEnoughInputs', ...
        'Need (S,fileTag,mainFolder,deformationMode,phiTarget).');
end
if ~isfield(S,'meta')
    error('generateInputFromSAW:BadS','S.meta missing.');
end
if ~isfield(S.meta,'phi') || ~isfield(S.meta,'L_grid')
    error('generateInputFromSAW:BadMeta','S.meta must contain phi and L_grid.');
end


%% Simulation parameters
b      = 1;
tau0   = 1e-8;
dt     = 1e-5;
npull  = 3e6;
nhold  = 3e7;
nhold0 = 1e8;
iout   = 1e6;
epsLJ  = 1.0;
gamma  = 1;

% Small mechanical deformation range used for ALL modes.
% For compression, these are positive compression magnitudes and
% lambda = 1 - strain.
engStrains = linspace(0.0, 0.3, 20);


% Initial and target packing fractions -> swelling factor

meta       = S.meta;
phiInitial = meta.phi;
L          = meta.L_grid;

if ~isfinite(phiInitial) || phiInitial <= 0
    error('generateInputFromSAW:BadInitialPhi', ...
        'Initial phi in S.meta.phi must be positive. Got %.8g.', phiInitial);
end
if ~isfinite(phiTarget) || phiTarget <= 0
    error('generateInputFromSAW:BadTargetPhi', ...
        'phiTarget must be positive. Got %.8g.', phiTarget);
end
if phiTarget > phiInitial
    error('generateInputFromSAW:TargetPhiTooLarge', ...
        ['This workflow treats the first deformation as swelling. ', ...
         'phiTarget (%.8g) cannot exceed phiInitial (%.8g).'], ...
         phiTarget, phiInitial);
end

fact = (phiInitial / phiTarget)^(1/3);
phiCheck = phiInitial / fact^3;


% Normalize loading-mode name

mode = lower(strtrim(deformationMode));
switch mode
    case {'tension','uniaxial','uniaxial_tension'}
        mode = 'tension';
        modeTitle = 'Uniaxial tension';
    case {'compression','uniaxial_compression'}
        mode = 'compression';
        modeTitle = 'Uniaxial compression';
    case {'pure_shear','shear','planar_extension'}
        mode = 'pure_shear';
        modeTitle = 'Pure shear / planar extension';
    otherwise
        error('generateInputFromSAW:BadDeformationMode', ...
            ['Unknown deformation mode "%s". Use tension, compression, ', ...
             'or pure_shear.'], deformationMode);
end


% Original SAW box and isotropically swollen reference box

xlo = 0; xhi = L;
ylo = 0; yhi = L;
zlo = 0; zhi = L;

% Symmetric swelling about original box center.
xmid = 0.5 * (xlo + xhi);
ymid = 0.5 * (ylo + yhi);
zmid = 0.5 * (zlo + zhi);

Lx_final = fact * (xhi - xlo);
Ly_final = fact * (yhi - ylo);
Lz_final = fact * (zhi - zlo);

xlo_final = xmid - 0.5 * Lx_final;
xhi_final = xmid + 0.5 * Lx_final;
ylo_final = ymid - 0.5 * Ly_final;
yhi_final = ymid + 0.5 * Ly_final;
zlo_final = zmid - 0.5 * Lz_final;
zhi_final = zmid + 0.5 * Lz_final;


% Files

inpDir  = fullfile(mainFolder,'Input');
outDump = fullfile(mainFolder,'Output','Dump');
if ~exist(inpDir,'dir'),  mkdir(inpDir);  end
if ~exist(outDump,'dir'), mkdir(outDump); end

topoFileFull = fullfile(inpDir,  sprintf('topology_%s.txt', fileTag));
inputFile    = fullfile(inpDir,  sprintf('input_%s.in',     fileTag));
atomDumpOut  = fullfile(outDump, sprintf('atoms_%s.dump',   fileTag));
bondDumpOut  = fullfile(outDump, sprintf('bonds_%s.dump',   fileTag));

if ~exist(topoFileFull,'file')
    error('generateInputFromSAW:MissingTopology', ...
        'Topology file missing: %s', topoFileFull);
end

% Filenames only because LAMMPS runs from Input/.
[~, topoName, topoExt] = fileparts(topoFileFull);
topFile = [topoName, topoExt];
[~, atomName, atomExt] = fileparts(atomDumpOut);
atomDump = [atomName, atomExt];
[~, bondName, bondExt] = fileparts(bondDumpOut);
bondDump = [bondName, bondExt];

fid = fopen(inputFile, 'w');
if fid < 0
    error('generateInputFromSAW:FileOpenFail','Cannot open: %s', inputFile);
end
cleanupObj = onCleanup(@() safeClose(fid)); %#ok<NASGU>


% LAMMPS header / force field / dynamics

fprintf(fid, '######### Input_%s #########\n', fileTag);
fprintf(fid, '# Loading mode      : %s\n', modeTitle);
fprintf(fid, '# Initial phi       : %.10g\n', phiInitial);
fprintf(fid, '# Target phi        : %.10g\n', phiTarget);
fprintf(fid, '# Swelling factor   : %.10g\n', fact);
fprintf(fid, '# phiInitial/fact^3 : %.10g\n\n', phiCheck);

fprintf(fid, 'units        lj\n');
fprintf(fid, 'boundary     p p p\n\n');

fprintf(fid, 'variable    kb      equal 1.380e-23\n');
fprintf(fid, 'variable    T       equal 1.000\n');
fprintf(fid, 'variable    kbT     equal 1.000\n\n');

fprintf(fid, 'variable    b       equal %.6g\n\n', b);

fprintf(fid, 'variable    eps_11  equal %.6g\n', epsLJ);
fprintf(fid, 'variable    eps_22  equal %.6g\n', epsLJ);
fprintf(fid, 'variable    eps_12  equal %.6g\n', epsLJ);
fprintf(fid, 'variable    sig_LJ  equal %.6g\n', b);
fprintf(fid, 'variable    sig_cut equal %.6g\n\n', 1.122*b);

fprintf(fid, 'variable    tau0    equal %.6g\n', tau0);
fprintf(fid, 'variable    dt      equal %.6g\n', dt);
fprintf(fid, 'variable    T0      equal 1.00\n');
fprintf(fid, 'variable    gamma   equal %.6g\n\n', gamma);

fprintf(fid, 'variable    ithermo equal %.6g\n', iout*10);
fprintf(fid, 'variable    iout    equal %.6g\n', iout);
fprintf(fid, 'variable    npull   equal %.6g\n', npull);
fprintf(fid, 'variable    nhold   equal %.6g\n', nhold);
fprintf(fid, 'variable    nhold0  equal %.6g\n\n', nhold0);

fprintf(fid, 'neighbor    %.6g bin\n', 1.122*b);
fprintf(fid, 'comm_modify cutoff  %.6g\n', 5*b);
fprintf(fid, 'neigh_modify delay 0 every 5 check yes\n\n');

fprintf(fid, 'atom_style   molecular\n');
fprintf(fid, 'bond_style   harmonic\n\n');

fprintf(fid, 'read_data %s &\n', topFile);
fprintf(fid, ' extra/bond/per/atom 1 extra/special/per/atom 10\n');
fprintf(fid, 'neigh_modify one 2000\n\n');

fprintf(fid, 'pair_style   lj/cut ${sig_cut}\n');
fprintf(fid, 'pair_coeff   1 1 ${eps_11} ${sig_LJ}\n');
fprintf(fid, 'pair_coeff   2 2 ${eps_22} ${sig_LJ}\n');
fprintf(fid, 'pair_coeff   1 2 ${eps_12} ${sig_LJ}\n');
fprintf(fid, 'special_bonds lj 0.0 1.0 1.0\n\n');

fprintf(fid, 'timestep ${dt}\n\n');

fprintf(fid, 'group  ends type 1\n');
fprintf(fid, 'group  ints type 2\n');
fprintf(fid, 'fix    free ints brownian ${T0} 6605 gamma_t ${gamma}\n\n');

fprintf(fid, 'compute  Pairs all property/local btype batom1 batom2\n');
fprintf(fid, 'compute  Bonds all bond/local dx dy dz dist force\n\n');

fprintf(fid, 'dump atomsdump all custom ${iout} %s id x y z vx vy vz type mol\n', atomDump);
fprintf(fid, 'dump bondsdump all local  ${iout} %s index c_Pairs[*] c_Bonds[*]\n\n', bondDump);

fprintf(fid, 'thermo_style  custom step time ebond epair vol press temp\n');
fprintf(fid, 'thermo        ${ithermo}\n');
fprintf(fid, 'thermo_modify flush yes\n');
fprintf(fid, 'thermo_modify lost warn\n\n');

fprintf(fid, 'timer full\n\n');


% Initial relaxation in phiInitial box

fprintf(fid, '# === Initial relaxation at phi_initial = %.10g ===\n', phiInitial);
fprintf(fid, 'run ${npull}\n\n');


% Isotropic swelling from phiInitial to phiTarget
% F_swell = fact * I

fprintf(fid, '# === Isotropic swelling ===\n');
fprintf(fid, '# phi_initial = %.10g, phi_target = %.10g, fact = %.10g\n', ...
    phiInitial, phiTarget, fact);
fprintf(fid, '# F_swell = diag(fact,fact,fact)\n');
fprintf(fid, 'fix EXPAND all deform 100 x final %.6f %.6f y final %.6f %.6f z final %.6f %.6f\n', ...
    xlo_final, xhi_final, ylo_final, yhi_final, zlo_final, zhi_final);
fprintf(fid, 'run ${npull}\n\n');
fprintf(fid, 'unfix EXPAND\n');
fprintf(fid, 'run ${nhold0}\n\n');


% Mechanical loading from the SWOLLEN reference configuration

Lx_exp = xhi_final - xlo_final;
Ly_exp = yhi_final - ylo_final;
Lz_exp = zhi_final - zlo_final;

x_mid = 0.5 * (xlo_final + xhi_final);
y_mid = 0.5 * (ylo_final + yhi_final);
z_mid = 0.5 * (zlo_final + zhi_final);

switch mode
    case 'tension'
        fprintf(fid, '# === Uniaxial tension in x after swelling ===\n');
        fprintf(fid, '# F = diag(lambda,lambda^(-1/2),lambda^(-1/2)); lambda=1+e\n');

    case 'compression'
        fprintf(fid, '# === Uniaxial compression in x after swelling ===\n');
        fprintf(fid, '# F = diag(lambda,lambda^(-1/2),lambda^(-1/2)); lambda=1-e\n');

    case 'pure_shear'
        fprintf(fid, '# === Pure shear / planar extension after swelling ===\n');
        fprintf(fid, '# F = diag(lambda,lambda^(-1),1); lambda=1+e\n');
end

for k = 1:numel(engStrains)
    e = engStrains(k);

    switch mode
        case 'tension'
            % Incompressible uniaxial tension:
            % F = diag(lambda, lambda^(-1/2), lambda^(-1/2))
            lambda = 1.0 + e;
            lx = lambda;
            ly = 1.0 / sqrt(lambda);
            lz = 1.0 / sqrt(lambda);

            % Preserve the original uniaxial-tension box convention:
            % x lower bound fixed; y and z centered.
            xlo_k = xlo_final;
            xhi_k = xlo_final + Lx_exp * lx;

            Ly_k = Ly_exp * ly;
            Lz_k = Lz_exp * lz;
            ylo_k = y_mid - 0.5 * Ly_k;
            yhi_k = y_mid + 0.5 * Ly_k;
            zlo_k = z_mid - 0.5 * Lz_k;
            zhi_k = z_mid + 0.5 * Lz_k;

        case 'compression'
            % Positive e means compression magnitude.
            % Incompressible uniaxial compression:
            % F = diag(lambda, lambda^(-1/2), lambda^(-1/2))
            % with lambda = 1-e < 1.
            lambda = 1.0 - e;
            if lambda <= 0
                error('generateInputFromSAW:BadCompressionRange', ...
                    'Compression strain gives lambda <= 0 at e=%.8g.', e);
            end
            lx = lambda;
            ly = 1.0 / sqrt(lambda);
            lz = 1.0 / sqrt(lambda);

            % Same x-boundary convention as the tension code.
            xlo_k = xlo_final;
            xhi_k = xlo_final + Lx_exp * lx;

            Ly_k = Ly_exp * ly;
            Lz_k = Lz_exp * lz;
            ylo_k = y_mid - 0.5 * Ly_k;
            yhi_k = y_mid + 0.5 * Ly_k;
            zlo_k = z_mid - 0.5 * Lz_k;
            zhi_k = z_mid + 0.5 * Lz_k;

        case 'pure_shear'
            % Incompressible pure shear / planar extension:
            % F = diag(lambda, 1/lambda, 1)
            lambda = 1.0 + e;
            lx = lambda;
            ly = 1.0 / lambda;
            lz = 1.0;

            Lx_k = Lx_exp * lx;
            Ly_k = Ly_exp * ly;
            Lz_k = Lz_exp * lz;

            % Preserve the previous pure-shear centered-box convention.
            xlo_k = x_mid - 0.5 * Lx_k;
            xhi_k = x_mid + 0.5 * Lx_k;
            ylo_k = y_mid - 0.5 * Ly_k;
            yhi_k = y_mid + 0.5 * Ly_k;
            zlo_k = z_mid - 0.5 * Lz_k;
            zhi_k = z_mid + 0.5 * Lz_k;
    end

    fprintf(fid, '\n# ---- mode=%s, engineering strain=%.6f, lambda=%.6f ----\n', ...
        mode, e, lambda);

    fprintf(fid, ...
        'fix DEF%d all deform 100 x final %.6f %.6f y final %.6f %.6f z final %.6f %.6f\n', ...
        k, xlo_k, xhi_k, ylo_k, yhi_k, zlo_k, zhi_k);

    fprintf(fid, 'run ${npull}\n');
    fprintf(fid, 'unfix DEF%d\n', k);
    fprintf(fid, 'run ${nhold}\n');
end

fclose(fid);
fid = -1; 
end

function safeClose(fid)
if isnumeric(fid) && isscalar(fid) && fid > 0
    try
        fclose(fid);
    catch
    end
end
end
