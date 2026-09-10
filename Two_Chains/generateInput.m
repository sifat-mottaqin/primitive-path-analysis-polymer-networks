function generateInput(N1, N2, fileTag, caseTag, mainFolder, dim, r_target, coords)
% create “input_FILETAG.in” in <mainFolder>\Input
% Uses globals tau0, dt
global b tau0 dt K eps gamma nhold npull 

% calculations

%  Chain 1 
r1_vec = coords(N1,:) - coords(1,:);           % end-to-end vector
r1_dist = norm(r1_vec);                        % end-to-end distance
r1 = r1_dist / ((N1-1) * b);                       % normalized stretch

%  Chain 2 
r2_vec = coords(N1+N2,:) - coords(N1+1,:);     % end-to-end vector
r2_dist = norm(r2_vec);
r2 = r2_dist / ((N2-1) * b);

% Choose current stretch and compute scale 
r_current = max(r1, r2);
scale = r_target / r_current;
scale = round(scale, 4);  

% compute max allowed scale based on N1, N2
max_scale = 10 - round(abs(N1 - N2) / 10);

% cap scale
scale = min(scale, max_scale);

Lx  = dim(1,2) - dim(1,1);
Ly  = dim(2,2) - dim(2,1);
Lz  = dim(3,2) - dim(3,1);
               
Lx_final = scale * Lx;
Ly_final = scale * Ly;
Lz_final = scale * Lz;

xlo_final = - 0.5*Lx_final;
xhi_final = 0.5*Lx_final;
ylo_final = - 0.5*Ly_final;
yhi_final =  0.5*Ly_final;
zlo_final =  - 0.5*Lz_final;
zhi_final =  0.5*Lz_final;


% filename 
topFile   = sprintf('topology_%s.txt', caseTag);
atomDump  = sprintf('atoms_%s.dump',   fileTag);
bondDump  = sprintf('bonds_%s.dump',   fileTag);
inputFile = fullfile(mainFolder,'Input',sprintf('input_%s.in',fileTag));

%% Start of writing
fid = fopen(inputFile,'w');

fprintf(fid,"######### Input_%s #########\n\n", fileTag);

fprintf(fid,"####################################\n");
fprintf(fid,"# DEFINE UNITS\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"units        lj\n");
fprintf(fid,"boundary p p p\n\n");

fprintf(fid,"####################################\n");
fprintf(fid,"# DEFINE LENGTH & FORCE SCALES\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"variable    kb      equal 1.380e-23\n");
fprintf(fid,"variable    T       equal 1.000\n");
fprintf(fid,"variable    kbT     equal 1.000\n\n");

fprintf(fid,"variable    b       equal %.2f\n\n", b);

fprintf(fid,"variable    eps_CC  equal %.2f\n", eps);
fprintf(fid,"variable    eps_AC  equal %.2f\n", eps);
fprintf(fid,"variable    eps_AA  equal %.2f\n", eps);
fprintf(fid,"variable    sig_LJ  equal %.2f\n", b);
fprintf(fid,"variable    sig_cut equal %.3f\n\n", 1.122*b);

fprintf(fid,"variable    tau0    equal %g\n", tau0);
fprintf(fid,"variable    dt      equal %g\n", dt);
fprintf(fid,"variable    T0      equal 1.00\n");
fprintf(fid,"variable    gamma   equal %g\n\n", gamma);

fprintf(fid,"variable    ithermo equal 50000\n");
fprintf(fid,"variable    iout    equal 50000\n");
fprintf(fid,"variable    npull   equal %.2e\n", npull);  % your convention
fprintf(fid,"variable    nhold   equal %.2e\n\n", nhold);

fprintf(fid,"####################################\n");
fprintf(fid,"# SIMULATION SPECIFICS\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"neighbor    %.2f bin\n", 5.0*b);
fprintf(fid,"neigh_modify delay  0 every 1 check yes\n\n");

fprintf(fid,"####################################\n");
fprintf(fid,"# INITIALIZE THE SYSTEM\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"atom_style   molecular\n");
fprintf(fid,"bond_style   nonlinear\n\n");
fprintf(fid,"read_data %s&\n", topFile);
fprintf(fid," extra/bond/per/atom 1 extra/special/per/atom 10\n");
fprintf(fid,"neigh_modify    one 10000\n\n");

fprintf(fid,"####################################\n");
fprintf(fid,"# SET ENVIRONMENT\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"pair_style lj/cut ${sig_cut}\n");
fprintf(fid,"pair_coeff 1 1 ${eps_CC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 1 3 ${eps_CC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 3 3 ${eps_CC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 2 2 ${eps_AA} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 2 4 ${eps_AA} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 4 4 ${eps_AA} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 1 2 ${eps_AC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 1 4 ${eps_AC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 3 2 ${eps_AC} ${sig_LJ}\n");
fprintf(fid,"pair_coeff 3 4 ${eps_AC} ${sig_LJ}\n");
fprintf(fid,"special_bonds lj 0.0 1.0 1.0\n\n");

fprintf(fid,"timestep ${dt}\n\n");
fprintf(fid,"group  ends    type 1 3\n");
fprintf(fid,"group  ints    type 2 4\n");
fprintf(fid,"fix    free    ints  brownian ${T0} 6605 gamma_t ${gamma}\n\n");

fprintf(fid,"compute  Pairs all property/local btype batom1 batom2\n");
fprintf(fid,"compute  Bonds all bond/local dx dy dz dist force\n\n");

fprintf(fid,"dump atomsdump all custom ${iout} %s id x y z vx vy vz type mol\n", atomDump);
fprintf(fid,"dump bondssdump all local  ${iout} %s index c_Pairs[*] c_Bonds[*]\n\n", bondDump);

fprintf(fid,"thermo_style    custom step time ebond epair vol press temp\n");
fprintf(fid,"thermo          ${ithermo}\n");
fprintf(fid,"thermo_modify   flush yes\n");
fprintf(fid,"thermo_modify   lost warn\n\n");

fprintf(fid,"####################################\n");
fprintf(fid,"# RUN-IN (optional)\n");
fprintf(fid,"####################################\n\n");
fprintf(fid,"run  ${npull}\n\n");

fprintf(fid, '\n# === Expansion section ===\n');
fprintf(fid, 'fix EXPAND all deform ${iout} x final %.6f %.6f y final %.6f %.6f z final %.6f %.6f\n',...
    xlo_final, xhi_final, ylo_final, yhi_final, zlo_final, zhi_final);
fprintf(fid, 'run ${npull}\n\n');
fprintf(fid, 'unfix EXPAND\n');

fprintf(fid, 'run 10000000\n\n');

fprintf(fid, 'run ${nhold}\n');

fclose(fid);
end
