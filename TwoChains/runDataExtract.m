function runDataExtract(N1,N2,fileTag,mainFolder)
% Robust extractor for paired LAMMPS atom/bond dumps.
% Fast path matches your original logic exactly for aligned files.
% If a single mismatch appears, it transparently resynchronizes
% (skips the lagging frame) and continues without crashing.

% ----------- your original constants -----------
N_ATOMS   = N1+N2;
N_BONDS   = N_ATOMS-2;
chunkRows = 20000;
precision = 'single';

atomDump = fullfile(mainFolder,'Output','Dump',sprintf('atoms_%s.dump',fileTag));
bondDump = fullfile(mainFolder,'Output','Dump',sprintf('bonds_%s.dump',fileTag));

outDir   = fullfile(mainFolder,'Output','RawData');
outMat   = fullfile(outDir,sprintf('raw_%s.mat',fileTag));
varName  = sprintf('raw_%s',fileTag);

ROW_COLS = 1 + 3*N_ATOMS + 5*N_BONDS;

fidA = fopen(atomDump,'r');  if fidA==-1, error('Cannot open %s',atomDump); end
fidB = fopen(bondDump,'r');  if fidB==-1, error('Cannot open %s',bondDump); end

fprintf('Extracting Data from %s\n',fileTag);

if exist(outMat,'file'), delete(outMat); end
mf = matfile(outMat,'Writable',true);
mf.(varName) = zeros(chunkRows,ROW_COLS,precision);

% keep your original formats (expecting: id x y z ...  and bonds columns)
fmtAtom = '%f%f%f%f%*[^\n]';
fmtBond = '%*f%*f%f%*f%f%f%f%f%f';

rowBuf      = zeros(chunkRows,ROW_COLS,precision);
xyz_tmp     = zeros(N_ATOMS,3,precision);
bond_tmp    = zeros(N_BONDS,5,precision);

rowInChunk  = 0;
rowsWritten = 0;

% ----------- helpers (used only if/when desync happens) -----------
    function t = next_ts(fid)
        % Find next "ITEM: TIMESTEP" and return numeric value; NaN on EOF
        t = NaN;
        while true
            line = fgetl(fid);
            if ~ischar(line), return; end
            if strncmp(line,'ITEM: TIMESTEP',14)
                num = fgetl(fid);
                if ~ischar(num), return; end
                t = str2double(num);
                return;
            end
        end
    end

    function skip_atom_frame(fid, nAtoms)
        % Skip NUMBER OF ATOMS line + number, 3 box lines, ATOMS header, then nAtoms lines
        while ~strncmp(fgetl(fid),'ITEM: NUMBER OF ATOMS',21), end
        fgetl(fid);                 % number (ignored)
        fgetl(fid); fgetl(fid); fgetl(fid);   % 3 box lines
        while ~strncmp(fgetl(fid),'ITEM: ATOMS',11), end
        for i=1:nAtoms
            if ~ischar(fgetl(fid)), break; end
        end
    end

    function nB = peek_bond_n(fid)
        % From current position (after TIMESTEP), find NUMBER OF ... and read it
        while true
            line = fgetl(fid);
            if ~ischar(line), nB = 0; return; end
            if strncmp(line,'ITEM: NUMBER OF ENTRIES',23) || strncmp(line,'ITEM: NUMBER OF BONDS',21)
                nB = str2double(fgetl(fid));
                return;
            end
        end
    end

    function skip_bond_frame(fid, nB)
        % Skip ENTRIES/BONDS header and nB lines
        while true
            line = fgetl(fid);
            if ~ischar(line), return; end
            if strncmp(line,'ITEM: ENTRIES',13) || strncmp(line,'ITEM: BONDS',11)
                break;
            end
        end
        for i=1:nB
            if ~ischar(fgetl(fid)), break; end
        end
    end

% ----------- MAIN LOOP -----------
desynced = false;

while true
    if ~desynced
        % ---------- FAST PATH (your original logic) ----------
        line = '';
        while ischar(line)
            line = fgetl(fidA);
            if ~ischar(line), break; end
            if strncmp(line,'ITEM: TIMESTEP',14)
                ts = str2double(fgetl(fidA)); break;
            end
        end
        if ~ischar(line), break; end

        while ~strncmp(fgetl(fidA),'ITEM: NUMBER OF ATOMS',22), end
        fgetl(fidA);
        fgetl(fidA); fgetl(fidA); fgetl(fidA);
        while ~strncmp(fgetl(fidA),'ITEM: ATOMS',11), end

        blkA = textscan(fidA,fmtAtom,N_ATOMS,'Delimiter',' ','MultipleDelimsAsOne',true);
        if any(cellfun(@isempty,blkA)), warning('Truncated atom frame @ %d; stopping.', ts); break; end
        blkA = cell2mat(blkA);
        ids  = blkA(:,1);
        xyz  = blkA(:,2:4);

        % bonds: read TIMESTEP and compare
        while true
            lineB = fgetl(fidB);
            if ~ischar(lineB), warning('bond dump ended early'); break; end
            if strncmp(lineB,'ITEM: TIMESTEP',14)
                tsB = str2double(fgetl(fidB));

                if tsB~=ts
                    % switch to robust path; first, resync the lagging file
                    desynced = true;
                    if tsB < ts
                        % bonds behind -> skip this bond frame completely
                        nB = peek_bond_n(fidB);
                        skip_bond_frame(fidB, nB);
                        % next loop iteration will use robust path
                    else
                        % atoms behind -> skip this atom frame completely
                        skip_atom_frame(fidA, N_ATOMS);
                        % and continue; robust path will handle re-align
                    end
                    % restart main while; do not write this frame
                    break;
                end
                % matched timestep: proceed as original
                break;
            end
        end
        if desynced
            % restart main loop and enter robust path
            continue;
        end
        if ~ischar(lineB), break; end

        while ~strncmp(fgetl(fidB),'ITEM: NUMBER OF ENTRIES',23), end
        nBondsNow = str2double(fgetl(fidB));
        while ~strncmp(fgetl(fidB),'ITEM: ENTRIES',13), end

        blkB  = textscan(fidB,fmtBond,nBondsNow,'Delimiter',' ','MultipleDelimsAsOne',true);
        if any(cellfun(@isempty,blkB)), warning('Truncated bond frame @ %d; stopping.', ts); break; end
        blkB  = cell2mat(blkB);
        blkB  = sortrows(blkB,1);
        bond5 = blkB(:,2:6);

        % pack & write (unchanged)
        xyz_tmp(ids,:) = cast(xyz,precision);
        bond_tmp(:,:)  = 0;
        bond_tmp(1:size(bond5,1),:) = cast(bond5,precision);

        vec = rowBuf(1,:);
        vec(1) = ts;
        vec(2:1+3*N_ATOMS)         = reshape(xyz_tmp.',1,[]);
        vec(2+3*N_ATOMS:end)       = reshape(bond_tmp.',1,[]);

        rowInChunk = rowInChunk + 1;
        rowBuf(rowInChunk,:) = vec;

        if rowInChunk == chunkRows
            if rowsWritten+chunkRows > size(mf.(varName),1)
                mf.(varName)(rowsWritten+chunkRows,ROW_COLS) = cast(0,precision);
            end
            mf.(varName)(rowsWritten+1:rowsWritten+chunkRows,:) = rowBuf;
            rowsWritten = rowsWritten + chunkRows;
            rowInChunk  = 0;
        end

    else
        % ---------- ROBUST PATH (only after first mismatch) ----------
        % Read next timestep from each stream
        ts  = next_ts(fidA); if isnan(ts), break; end
        tsB = next_ts(fidB); if isnan(tsB), break; end

        % Resynchronize: advance lagging file until ts == tsB
        while ts ~= tsB
            if ts < tsB
                skip_atom_frame(fidA, N_ATOMS);
                ts = next_ts(fidA); if isnan(ts), break; end
            else
                nB = peek_bond_n(fidB);
                skip_bond_frame(fidB, nB);
                tsB = next_ts(fidB); if isnan(tsB), break; end
            end
        end
        if isnan(ts) || isnan(tsB), break; end

        % Now frames are aligned: parse them just like your original code
        while ~strncmp(fgetl(fidA),'ITEM: NUMBER OF ATOMS',22), end
        fgetl(fidA);
        fgetl(fidA); fgetl(fidA); fgetl(fidA);
        while ~strncmp(fgetl(fidA),'ITEM: ATOMS',11), end

        blkA = textscan(fidA,fmtAtom,N_ATOMS,'Delimiter',' ','MultipleDelimsAsOne',true);
        if any(cellfun(@isempty,blkA)), warning('Truncated atom frame @ %d; stopping.', ts); break; end
        blkA = cell2mat(blkA);
        ids  = blkA(:,1);
        xyz  = blkA(:,2:4);

        % BONDS
        while ~strncmp(fgetl(fidB),'ITEM: NUMBER OF ENTRIES',23)
            % accept alt spelling too (rare)
            pos = ftell(fidB);
            lineTmp = fgetl(fidB);
            if ~ischar(lineTmp), error('bond dump ended unexpectedly'); end
            if strncmp(lineTmp,'ITEM: NUMBER OF BONDS',21)
                % rewind one line so next fgetl reads the number
                fseek(fidB,pos,'bof'); break;
            end
        end
        nLine = fgetl(fidB);
        if ~ischar(nLine), warning('bond dump ended early'); break; end
        nBondsNow = str2double(nLine);

        % accept ENTRIES or BONDS header
        while true
            lineB = fgetl(fidB);
            if ~ischar(lineB), warning('bond dump ended early'); break; end
            if strncmp(lineB,'ITEM: ENTRIES',13) || strncmp(lineB,'ITEM: BONDS',11)
                break;
            end
        end
        if ~ischar(lineB), break; end

        blkB  = textscan(fidB,fmtBond,nBondsNow,'Delimiter',' ','MultipleDelimsAsOne',true);
        if any(cellfun(@isempty,blkB)), warning('Truncated bond frame @ %d; stopping.', ts); break; end
        blkB  = cell2mat(blkB);
        blkB  = sortrows(blkB,1);
        bond5 = blkB(:,2:6);

        % pack & write (unchanged)
        xyz_tmp(ids,:) = cast(xyz,precision);
        bond_tmp(:,:)  = 0;
        bond_tmp(1:size(bond5,1),:) = cast(bond5,precision);

        vec = rowBuf(1,:);
        vec(1) = ts;
        vec(2:1+3*N_ATOMS)         = reshape(xyz_tmp.',1,[]);
        vec(2+3*N_ATOMS:end)       = reshape(bond_tmp.',1,[]);

        rowInChunk = rowInChunk + 1;
        rowBuf(rowInChunk,:) = vec;

        if rowInChunk == chunkRows
            if rowsWritten+chunkRows > size(mf.(varName),1)
                mf.(varName)(rowsWritten+chunkRows,ROW_COLS) = cast(0,precision);
            end
            mf.(varName)(rowsWritten+1:rowsWritten+chunkRows,:) = rowBuf;
            rowsWritten = rowsWritten + chunkRows;
            rowInChunk  = 0;
        end
    end
end

% ----------- tail write & finalize (unchanged) -----------
if rowInChunk>0
    mf.(varName)(rowsWritten+1:rowsWritten+rowInChunk,:) = rowBuf(1:rowInChunk,:);
    rowsWritten = rowsWritten + rowInChunk;
end
mf.(varName) = mf.(varName)(1:rowsWritten,:);

meta = struct('nAtoms',N_ATOMS,'nBonds',N_BONDS,'nCols',ROW_COLS, ...
              'precision',precision,'rows',rowsWritten);
save(outMat,'meta','-append');

fclose(fidA); fclose(fidB);
end
