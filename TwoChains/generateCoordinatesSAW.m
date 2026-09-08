function [LN_Out, coords, dim] = generateCoordinatesSAW(N1, N2, rho_target)
global b
% --- User parameters ---
%N1 = CN1;               % beads in chain 1
%N2 = CN2;               % beads in chain 2
%rho_target = 0.1;       % target density
%b  = bond;              % lattice spacing
%r  = rad;               % bead radius for plotting
maxTrials = 5e3;        % attempts before restart

% --- Box from density ---
Ntot = N1 + N2;
L_ideal  = (Ntot / rho_target)^(1/3);
M        = max(ceil(L_ideal / b), 6); % integer sites per axis
L_box    = M * b;
%rho_actual = Ntot / (L_box^3);

%fprintf('Target rho=%.6f, box L=%.3f, actual rho=%.6f, M=%d\n', ...
%    rho_target, L_box, rho_actual, M);

% --- Loop until linking found ---
attempt = 0;
while true
    attempt = attempt + 1;

    % Chain 1
    seed1 = randi(M,1,3)-1;
    [idx1, ok1] = grow_SAW_unbiased(N1, maxTrials, seed1, M, []);
    if ~ok1
        fprintf('Attempt %d: chain1 failed\n', attempt);
        continue;
    end

    % Chain 2, avoid chain 1
    seed2 = randi(M,1,3)-1;
    [idx2, ok2] = grow_SAW_unbiased(N2, maxTrials, seed2, M, idx1);
    if ~ok2
        fprintf('Attempt %d: chain2 failed\n', attempt);
        continue;
    end

    % Wrap to [0,L)
    C1_wrapped = (double(idx1) + 0.5) * b;
    C2_wrapped = (double(idx2) + 0.5) * b;

    % Unwrap
    C1_unwrapped = unwrap_path(idx1, M) * b;
    C2_unwrapped = unwrap_path(idx2, M) * b;

    % GLN
    [LN, ~] = calcGaussianLNandCoE(C1_unwrapped, C2_unwrapped);
    %fprintf('Attempt %d: GLN=%.3f\n', attempt, Lk);

    if abs(LN) > 1
        LN_Out = LN;
        fprintf('Accepted configuration with GLN = %.3f\n', LN);
        % Animate and save HD video
        %save_video(C1_wrapped, C2_wrapped, L_box, r, 'SAW_2chains_HD.mp4');
        % Final plot
        %plot_two_chains_pbc(C1_wrapped, C2_wrapped, L_box, r);
        break;
    end
end
coords = [C1_unwrapped; C2_unwrapped];
coords(:,1:3) = coords(:,1:3) - mean(coords(:,1:3),1);
%dim = compute_dim(coords, N1, N2, b);
dim = compute_dim_robust(coords, N1, N2, b, ...
    'StretchFactor',1.1, 'PadBeads',2);
end

% --- Growth function ---
function [idx, success] = grow_SAW_unbiased(N, maxTrials, startIdx, M, otherIdx)
success = false;
dirs = int32([ 1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1 ]);
for t = 1:maxTrials
    idx = zeros(N,3,'int32');
    idx(1,:) = wrap_idx(startIdx, M);
    occ = containers.Map('KeyType','char','ValueType','logical');
    occ(keyOf(idx(1,:))) = true;

    if ~isempty(otherIdx)
        for i = 1:size(otherIdx,1)
            occ(keyOf(wrap_idx(otherIdx(i,:),M))) = true;
        end
    end

    stuck = false;
    for k = 2:N
        cur = idx(k-1,:);
        nbrs = wrap_idx(bsxfun(@plus, double(cur), double(dirs)), M);

        free = [];
        for j = 1:6
            if ~isKey(occ, keyOf(nbrs(j,:)))
                free(end+1,:) = nbrs(j,:); %#ok<AGROW>
            end
        end
        if isempty(free), stuck = true; break; end

        next = free(randi(size(free,1)),:);
        idx(k,:) = int32(next);
        occ(keyOf(idx(k,:))) = true;
    end
    if ~stuck, success = true; return; end
end
idx = [];
end

% --- Unwrap indices ---
function U = unwrap_path(idx, M)
U = zeros(size(idx), 'double'); U(1,:) = double(idx(1,:));
for k = 2:size(idx,1)
    step = mi_delta(idx(k,:), idx(k-1,:), M);
    U(k,:) = U(k-1,:) + step;
end
end

% --- PBC helpers ---
function w = wrap_idx(p, M)
w = int32(mod(int32(p), int32(M)));
end

function d = mi_delta(a, b, M)
d = double(a) - double(b);
d = d - M * round(d / M);
end

function s = keyOf(p)
s = sprintf('%d,%d,%d', p(1), p(2), p(3));
end

%%
function dim = compute_dim_robust(coords, N1, N2, b, varargin)
% COMPUTE_DIM_ROBUST
%   Returns a *cubic* LAMMPS box that safely contains two chains for
%   any conformation (coiled ? fully extended).
%
%   coords : (N x 3) atom positions (both chains)
%   N1, N2 : beads per chain
%   b      : bead size (e.g., sigma or bond length unit)
%
%   Name-value (optional):
%     'StretchFactor' (default 1.1)  % allow bond stretch beyond contour
%     'PadBeads'      (default 2)    % extra padding in units of b
%     'Cutoff'        (default 2.5*b)% LJ cutoff (typical)
%     'Skin'          (default 0.3*b)% neighbor skin

p = inputParser;
addParameter(p,'StretchFactor',1.1);
addParameter(p,'PadBeads',2);
addParameter(p,'Cutoff',2.5*b);
addParameter(p,'Skin',0.3*b);
parse(p,varargin{:});
s     = p.Results.StretchFactor;
pad   = p.Results.PadBeads * b;
rc    = p.Results.Cutoff;
skin  = p.Results.Skin;

% center for symmetric box placement
ctr = mean(coords,1);
C   = coords - ctr;

% current half-extent (make it cubic)
half_now = max(max(abs(C),[],1));  % scalar

% theoretical half-extent (fully extended; allow stretch)
Lc1 = (N1-1) * b;
Lc2 = (N2-1) * b;
half_theory = 0.5 * s * max(Lc1, Lc2);

% safety margin: padding + interaction buffer
safety = pad + rc + skin;

% final cubic half-length
half = max(half_now, half_theory) + safety;

% cubic box centered at ctr
dim = [ ctr(1)-half, ctr(1)+half; ...
    ctr(2)-half, ctr(2)+half; ...
    ctr(3)-half, ctr(3)+half ];
end

