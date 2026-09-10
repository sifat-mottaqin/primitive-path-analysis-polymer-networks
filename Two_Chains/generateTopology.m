function generateTopology(coords,N1, N2,fileTag,mainFolder, dim)

global K
xyz = coords;
Natom  = N1 + N2;
Nbonds = Natom - 2;

%% box limits  (± one segment beyond extents)
xlo = dim(1,1);   xhi = dim(1,2);
ylo = dim(2,1);   yhi = dim(2,2);
zlo = dim(3,1);   zhi = dim(3,2);

%% atom type & chain id
atype          = zeros(Natom,1);
atype(1)       = 1;                     % ends of chain-1
atype(N1)      = 1;
atype(2:N1-1)  = 2;                     % internal chain-1
atype(N1+1)    = 3;                     % ends of chain-2
atype(end)     = 3;
atype(N1+2:end-1) = 4;                  % internal chain-2

chain = [ones(N1,1); 2*ones(N2,1)];

%% bonds
bond_i = [(1:N1-1)'; (N1+1:N1+N2-1)'];
bond_j = bond_i + 1;

%% write file
outfile = fullfile(mainFolder,'Input',sprintf('topology_%s.txt',fileTag));
fid = fopen(outfile,'w');

fprintf(fid,'Topology 2 chains %d atoms\n\n',Natom);
fprintf(fid,'%d atoms\n',  Natom);
fprintf(fid,'%d bonds\n',  Nbonds);
fprintf(fid,'0 angles\n0 dihedrals\n0 impropers\n\n');
fprintf(fid,'4 atom types\n1 bond types\n\n');
fprintf(fid,'% .15e % .15e xlo xhi\n',xlo,xhi);
fprintf(fid,'% .15e % .15e ylo yhi\n',ylo,yhi);
fprintf(fid,'% .15e % .15e zlo zhi\n\n',zlo,zhi);

fprintf(fid,'Masses\n\n1 1\n2 1\n3 1\n4 1\n\n');
fprintf(fid,'Bond Coeffs\n\n1 %d 1 1\n\n', K);

fprintf(fid,'Atoms\n\n');
for id = 1:Natom
    fprintf(fid,'%d %d %d % .6f % .6f % .6f\n',...
            id, chain(id), atype(id), xyz(id,1), xyz(id,2), xyz(id,3));
end
fprintf(fid,'\nVelocities\n\n');
for id = 1:Natom
    fprintf(fid,'%d 0 0 0\n',id);
end

fprintf(fid,'\nBonds\n\n');
for k = 1:Nbonds
    fprintf(fid,'%d 1 %d %d\n',k,bond_i(k),bond_j(k));
end

fclose(fid);
end
