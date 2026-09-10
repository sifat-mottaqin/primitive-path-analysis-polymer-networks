function visSAWNetwork(chainPos, meta, fileTag, mainFolder, doSave, figPng, figFig)
% visSAWNetwork
% Makes TWO figures:
%   (1) Original PBC-wrapped coordinates in [1..L]
%   (2) Unwrapped for visualization (continuous chain, may go outside [1..L])
%
% Beads are rendered as spheres with diameter = 1.
%
% Inputs:
%   chainPos  : cell{nChains}, each (nBeads x 3) int coords in [1..L]
%   meta      : struct (expects meta.L_grid)
%   fileTag   : string used in title
%   mainFolder: root path (for default Figures folder)
%   doSave    : true/false to save figure(s)
%   figPng    : base png path (optional). If provided, we'll create:
%               <name>_wrapped.png and <name>_unwrapped.png
%   figFig    : base fig path (optional). If provided, we'll create:
%               <name>_wrapped.fig and <name>_unwrapped.fig

    if nargin < 5
        doSave = false;
    end

    L = meta.L_grid;

    % % ---- Build unwrapped coordinates (per chain)
    % chainPosUnwrap = cell(size(chainPos));
    % for c = 1:numel(chainPos)
    %     P = double(chainPos{c});
    %     chainPosUnwrap{c} = unwrapChainPBC(P, L);
    % end

    % Determine save paths (if requested)
    if doSave
        if nargin < 6 || isempty(figPng) || nargin < 7 || isempty(figFig)
            figFolder = fullfile(mainFolder, 'Figures');
            if ~exist(figFolder,'dir'); mkdir(figFolder); end
            basePng = fullfile(figFolder, ['visSAW_', fileTag, '.png']);
            baseFig = fullfile(figFolder, ['visSAW_', fileTag, '.fig']);
        else
            basePng = figPng;
            baseFig = figFig;
        end

        [pngDir, pngName, ~] = fileparts(basePng);
        [figDir, figName, ~] = fileparts(baseFig);

        wrappedPng   = fullfile(pngDir, [pngName, '_wrapped.png']);
        %unwrappedPng = fullfile(pngDir, [pngName, '_unwrapped.png']);
        wrappedFig   = fullfile(figDir, [figName, '_wrapped.fig']);
        %unwrappedFig = fullfile(figDir, [figName, '_unwrapped.fig']);
    end

    %% FIGURE 1: Wrapped (original)  
    f1 = figure('Color','w');
    ax1 = axes('NextPlot','add'); hold(ax1,'on'); grid(ax1,'on'); axis(ax1,'equal'); view(ax1,3);
    camlight(ax1,'headlight'); lighting(ax1,'gouraud');

    drawNetworkWithSpheres(ax1, chainPos, 1.0);

    xlim(ax1,[1 L]); ylim(ax1,[1 L]); zlim(ax1,[1 L]);
    xlabel(ax1,'x'); ylabel(ax1,'y'); zlabel(ax1,'z');
    title(ax1, sprintf('SAW Network (WRAPPED/PBC) | %s | L=%d | bead diam=1', fileTag, L), 'Interpreter','none');
    drawBoxWireframe(L, ax1);

    if doSave
        savefig(f1, wrappedFig);
        exportgraphics(f1, wrappedPng, 'Resolution', 300);
        fprintf('[visSAW] Saved WRAPPED:\n  %s\n  %s\n', wrappedFig, wrappedPng);
    end

    % % FIGURE 2: Unwrapped (continuous)
    % 
    % f2 = figure('Color','w');
    % ax2 = axes('NextPlot','add'); hold(ax2,'on'); grid(ax2,'on'); axis(ax2,'equal'); view(ax2,3);
    % camlight(ax2,'headlight'); lighting(ax2,'gouraud');
    % 
    % drawNetworkWithSpheres(ax2, chainPosUnwrap, 1.0);
    % 
    % % Autoscale limits for unwrapped plot
    % allP = cell2mat(cellfun(@(P) double(P), chainPosUnwrap, 'UniformOutput', false));
    % pad = 2; % small padding
    % xlim(ax2,[min(allP(:,1))-pad, max(allP(:,1))+pad]);
    % ylim(ax2,[min(allP(:,2))-pad, max(allP(:,2))+pad]);
    % zlim(ax2,[min(allP(:,3))-pad, max(allP(:,3))+pad]);
    % 
    % xlabel(ax2,'x'); ylabel(ax2,'y'); zlabel(ax2,'z');
    % title(ax2, sprintf('SAW Network (UNWRAPPED for visualization) | %s | L=%d | bead diam=1', fileTag, L), ...
    %     'Interpreter','none');
    % 
    % if doSave
    %     savefig(f2, unwrappedFig);
    %     exportgraphics(f2, unwrappedPng, 'Resolution', 300);
    %     fprintf('[visSAW] Saved UNWRAPPED:\n  %s\n  %s\n', unwrappedFig, unwrappedPng);
    % end
end

% Helper: draw chains as polyline + spheres (diameter = beadDiam)
function drawNetworkWithSpheres(ax, chainPosCell, beadDiam)
    beadR   = beadDiam/2;
    nSphere = 18;
    [sx,sy,sz] = sphere(nSphere);

    nChains = numel(chainPosCell);
    cmap = parula(nChains);   % one color per chain

    for c = 1:nChains
        P = double(chainPosCell{c});
        col = cmap(c,:);      % fixed color for this chain

        % backbone
        plot3(ax, P(:,1), P(:,2), P(:,3), '-', ...
            'LineWidth', 1.4, 'Color', col);

        % beads
        for b = 1:size(P,1)
            cx = P(b,1); cy = P(b,2); cz = P(b,3);
            surf(ax, ...
                beadR*sx + cx, beadR*sy + cy, beadR*sz + cz, ...
                'FaceColor', col, ...
                'EdgeColor','none', ...
                'FaceLighting','gouraud');
        end
    end
end


% Helper: unwrap chain coordinates so steps are continuous across PBC
% Assumes nearest-neighbor lattice steps (±1 along one axis).
% Output starts at original bead1 position and then accumulates minimal-image steps.
% function Puw = unwrapChainPBC(P, L)
%     n = size(P,1);
%     Puw = zeros(n,3);
%     Puw(1,:) = P(1,:);
% 
%     for i = 2:n
%         d = P(i,:) - P(i-1,:);
% 
%         % minimal-image correction on each component
%         % If d is large positive, we likely crossed from 1 -> L (wrap), so subtract L
%         % If d is large negative, we likely crossed from L -> 1 (wrap), so add L
%         for k = 1:3
%             if d(k) >  L/2
%                 d(k) = d(k) - L;
%             elseif d(k) < -L/2
%                 d(k) = d(k) + L;
%             end
%         end
% 
%         Puw(i,:) = Puw(i-1,:) + d;
%     end
% end

% Helper: box wireframe (wrapped plot)
function drawBoxWireframe(L, ax)
    corners = [ ...
        1 1 1;
        L 1 1;
        L L 1;
        1 L 1;
        1 1 L;
        L 1 L;
        L L L;
        1 L L];

    edges = [ ...
        1 2; 2 3; 3 4; 4 1; ...
        5 6; 6 7; 7 8; 8 5; ...
        1 5; 2 6; 3 7; 4 8];

    for i = 1:size(edges,1)
        a = corners(edges(i,1),:);
        b = corners(edges(i,2),:);
        plot3(ax, ...
            [a(1) b(1)], [a(2) b(2)], [a(3) b(3)], ...
            'k-', 'LineWidth', 1.2);   % BLACK box
    end
end

