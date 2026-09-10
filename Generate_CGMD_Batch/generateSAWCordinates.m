function generateSAWCordinates(no_chains, no_beads, phi, fileTag, mainFolder)
if nargin < 5
    error('Need (no_chains,no_beads,phi,fileTag,mainFolder).');
end

coordFolder = fullfile(mainFolder, 'Coordinates');
if ~exist(coordFolder,'dir'); mkdir(coordFolder); end
coordOut = fullfile(coordFolder, ['coords_', fileTag, '.mat']);

tot_beads = no_chains * no_beads;
L_grid    = ceil((tot_beads/phi)^(1/3));

fprintf('[SAW] L_grid=%d  tot_beads=%d  phi=%.3f\n', L_grid, tot_beads, phi);

% knobs for dense packing 
maxStep          = 5;       % allow jumps up to this lattice distance
maxBacktracks    = 50000;   % safety cap per chain
maxStartTrials   = 2e5;     % try to find a free start
maxChainRestarts = 200;     % last-resort restarts per chain (should be rare with backtracking)

% Enforce: end beads (bead 1 and bead N) cannot lie on boundary planes
avoidEndBoundary = true;

% rng(1);

offsetsByR = precomputeOffsets(maxStep, true);  % Chebyshev shell

occ = false(L_grid^3, 1);
chainPos = cell(no_chains, 1);

for c = 1:no_chains
    %fprintf('[SAW] Chain %d/%d\n', c, no_chains);

    success = false;

    for restart = 1:maxChainRestarts

        tempPos = zeros(no_beads, 3, 'int32');
        tempLin = zeros(no_beads, 1, 'uint32');
        placedN = 0;

        candPos = cell(no_beads,1);  % candidates for bead i+1 from bead i
        candLin = cell(no_beads,1);

        % start bead 
        startFound = false;
        for t = 1:maxStartTrials
            x0 = randi(L_grid);
            y0 = randi(L_grid);
            z0 = randi(L_grid);

            p0   = int32([x0 y0 z0]);
            lin0 = sub2ind3(L_grid, x0, y0, z0);

            if ~occ(lin0)
                if ~avoidEndBoundary || ~isBoundarySite(p0, L_grid)
                    startFound = true;
                    break;
                end
            end
        end
        if ~startFound
            error('No free start site (respecting end-boundary rule). Increase L_grid or reduce phi.');
        end

        placedN = 1;
        tempPos(1,:) = int32([x0 y0 z0]);
        tempLin(1)   = uint32(lin0);
        occ(lin0)    = true;

        b = 2;
        backtracks = 0;

        while true
            if b > no_beads
                success = true;
                break;
            end

            prev = tempPos(b-1,:);

            % generate candidates for bead b from bead b-1 if not already
            if isempty(candPos{b-1})
                [Pcan, Lcan] = gatherCandidates(prev, L_grid, occ, offsetsByR);
                candPos{b-1} = Pcan;
                candLin{b-1} = Lcan;
            end

            if ~isempty(candPos{b-1})
                % take one candidate (pop last)
                Pcan = candPos{b-1};
                Lcan = candLin{b-1};
                nxt  = Pcan(end,:);
                lin  = Lcan(end);

                candPos{b-1}(end,:) = [];
                candLin{b-1}(end)   = [];

                % enforce end-bead boundary rule for the LAST bead only
                if avoidEndBoundary && (b == no_beads) && isBoundarySite(nxt, L_grid)
                    % reject this candidate; try another
                    continue;
                end

                if ~occ(lin)
                    tempPos(b,:) = nxt;
                    tempLin(b)   = uint32(lin);
                    occ(lin)     = true;
                    placedN = max(placedN, b);

                    candPos{b} = [];
                    candLin{b} = [];

                    b = b + 1;
                    continue;
                else
                    continue;
                end
            else
                % backtrack
                backtracks = backtracks + 1;
                if backtracks > maxBacktracks
                    break;
                end

                if b <= 2
                    % cannot backtrack past bead 1 -> restart chain
                    break;
                end

                % remove bead (b-1)
                linDel = double(tempLin(b-1));
                occ(linDel) = false;

                candPos{b-1} = [];
                candLin{b-1} = [];

                b = b - 1;
            end
        end

        if success
            chainPos{c} = tempPos;
            break;
        else
            % undo occupied sites for this failed restart
            for u = 1:placedN
                occ(double(tempLin(u))) = false;
            end
        end
    end

    if ~success
        error('Failed to build chain %d. Try increasing L_grid or maxStep, or lowering phi.', c);
    end
end

combined = zeros(tot_beads, 5, 'int32');
idx = 0;
for c = 1:no_chains
    P = chainPos{c};
    for b = 1:no_beads
        idx = idx + 1;
        combined(idx,:) = int32([c, b, P(b,1), P(b,2), P(b,3)]);
    end
end

meta = struct();
meta.no_chains = no_chains;
meta.no_beads  = no_beads;
meta.tot_beads = tot_beads;
meta.phi       = phi;
meta.L_grid    = L_grid;
meta.dim       = 3;
meta.pbc       = true;
meta.maxStep   = maxStep;
meta.fileTag   = fileTag;
meta.avoidEndBoundary = avoidEndBoundary;

save(coordOut, 'chainPos', 'combined', 'meta');
%fprintf('[SAW] Saved: %s\n', coordOut);

end

% candidate gathering 
function [Pcan, Lcan] = gatherCandidates(cur, L, occ, offsetsByR)
% returns candidates in random order, trying r=1..maxStep
Pall = zeros(0,3,'int32');
Lall = zeros(0,1,'uint32');

for r = 1:numel(offsetsByR)
    offs = offsetsByR{r};
    if isempty(offs), continue; end
    perm = randperm(size(offs,1));
    offs = offs(perm,:);

    for k = 1:size(offs,1)
        nxt = cur + offs(k,:);
        nxt = wrapPBC(nxt, L);
        lin = sub2ind3(L, nxt(1), nxt(2), nxt(3));
        if ~occ(lin)
            Pall(end+1,:) = nxt;         
            Lall(end+1,1) = uint32(lin); 
        end
    end

    if ~isempty(Pall)
        % stop at first radius that offers at least one move
        break;
    end
end

Pcan = Pall;
Lcan = Lall;
end

% offsets 
function offsetsByR = precomputeOffsets(maxStep, useChebyshevShell)
offsetsByR = cell(maxStep,1);

for r = 1:maxStep
    vals = -r:r;
    [DX,DY,DZ] = ndgrid(vals, vals, vals);
    dx = DX(:); dy = DY(:); dz = DZ(:);

    nz = ~(dx==0 & dy==0 & dz==0);

    if useChebyshevShell
        shell = max([abs(dx), abs(dy), abs(dz)], [], 2) == r;
    else
        shell = (abs(dx) + abs(dy) + abs(dz)) == r;
    end

    keep = nz & shell;
    offsetsByR{r} = int32([dx(keep), dy(keep), dz(keep)]);
end
end

% end-boundary test 
function tf = isBoundarySite(p, L)
% p is 1x3 int32
tf = (p(1)==1 || p(1)==L || p(2)==1 || p(2)==L || p(3)==1 || p(3)==L);
end

% PBC + indexing 
function nxt = wrapPBC(nxt, L)
nxt = mod(nxt-1, L) + 1;
end

function lin = sub2ind3(L, x, y, z)
lin = double(x) + (double(y)-1)*L + (double(z)-1)*L*L;
end

