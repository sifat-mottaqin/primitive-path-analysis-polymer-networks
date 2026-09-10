function generateTopologyFromSAW(S, fileTag, mainFolder)
% S must contain: S.chainPos (cell), S.meta (struct with L_grid, no_chains, no_beads)

K = 1500;        
chainPos  = S.chainPos;
meta      = S.meta;

no_chains = meta.no_chains;
no_beads  = meta.no_beads;
L         = meta.L_grid;

Natom  = no_chains * no_beads;
Nbonds = no_chains * (no_beads - 1);

xlo = 0; xhi = L;
ylo = 0; yhi = L;
zlo = 0; zhi = L;

% flatten coords into (Natom x 3)
xyz   = zeros(Natom,3);
chain = zeros(Natom,1);
atype = 2*ones(Natom,1);   % default internal bead type=2

id0 = 0;
for c = 1:no_chains
    P = double(chainPos{c});
    if size(P,1) ~= no_beads
        error('generateTopologyFromSAW:BadChainSize', 'Chain %d has %d beads; expected %d.', c, size(P,1), no_beads);
    end

    ids = (id0 + 1):(id0 + no_beads);

    xyz(ids,:)   = P;
    chain(ids)   = c;

    atype(ids(1))   = 1;      % chain end
    atype(ids(end)) = 1;      % chain end

    id0 = id0 + no_beads;
end

% bonds: connect (1-2-...-no_beads) within each chain
bond_i = zeros(Nbonds,1);
bond_j = zeros(Nbonds,1);

k0 = 0;
for c = 1:no_chains
    startId = (c-1)*no_beads + 1;
    ids_i   = (startId:(startId+no_beads-2))';
    ids_j   = ids_i + 1;

    nk = numel(ids_i);
    bond_i(k0 + (1:nk)) = ids_i;
    bond_j(k0 + (1:nk)) = ids_j;
    k0 = k0 + nk;
end

% write LAMMPS data file in Input/
outfile = fullfile(mainFolder,'Input',sprintf('topology_%s.txt',fileTag));
fid = fopen(outfile,'w');
if fid < 0
    error('generateTopologyFromSAW:FileOpenFail', 'Cannot open: %s', outfile);
end

fprintf(fid,'Topology SAW network: %d chains x %d beads = %d atoms\n\n', no_chains, no_beads, Natom);
fprintf(fid,'%d atoms\n',  Natom);
fprintf(fid,'%d bonds\n',  Nbonds);
fprintf(fid,'0 angles\n0 dihedrals\n0 impropers\n\n');
fprintf(fid,'2 atom types\n1 bond types\n\n');

fprintf(fid,'%.5e %.5e xlo xhi\n',xlo,xhi);
fprintf(fid,'%.5e %.5e ylo yhi\n',ylo,yhi);
fprintf(fid,'%.5e %.5e zlo zhi\n\n',zlo,zhi);

fprintf(fid,'Masses\n\n');
fprintf(fid,'1 1\n');
fprintf(fid,'2 1\n\n');

fprintf(fid,'Bond Coeffs\n\n');
fprintf(fid,'1 %d 1 \n\n', K);

fprintf(fid,'Atoms\n\n');
for id = 1:Natom
    fprintf(fid,'%d %d %d % .2f % .2f % .2f\n', ...
        id, chain(id), atype(id), xyz(id,1), xyz(id,2), xyz(id,3));
end

fprintf(fid,'\nVelocities\n\n');
for id = 1:Natom
    fprintf(fid,'%d 0 0 0\n', id);
end

fprintf(fid,'\nBonds\n\n');
for k = 1:Nbonds
    fprintf(fid,'%d 1 %d %d\n', k, bond_i(k), bond_j(k));
end

fclose(fid);

end
