function PostProcessData(Package,ToggleDynamics,LC,DC,FC,BSOM,CM,CF,OF,BT) %#ok<INUSD>
% Force-vs-r* post-processing for the bead-spring model only.
% r* = r/(N*b), where r is the chain end-to-end distance.

global LengthConversion ForceConversion BeadSpringOrMeso BondType...
    CurrentFolder OutputFolder ForceDataFileName TimeStretchDataFileName

LengthConversion = LC;
ForceConversion = FC;
BeadSpringOrMeso = 0;
BondType = BT;
CurrentFolder = CF;
OutputFolder = OF;

if ~isfolder('Output Plots')
    mkdir('Output Plots')
end
if ~isfolder('Force Data')
    mkdir('Force Data')
end

Nps = unique(Package(:,2));
Ds = unique(Package(:,3));
N_Kuhns = unique(Package(:,4));
stiffnesses = unique(Package(:,5));
kbTs = unique(Package(:,6));
bs = unique(Package(:,7));
dts = unique(Package(:,8));

Np = Nps(1);
D = Ds(1);
kbT = kbTs(1);
b = bs(1);
dt = dts(1);

FontSize = 20;
figure(3); clf; hold on

% Keep the ORIGINAL force-plot colors and marker styles.
color1 = [0 0 0];
color2 = [1 0 0];
colors_stf = DefinePlotColors(color1,color2,length(stiffnesses));
marker_styles = {'o','^','sq','>','d','v'};
legendEntries = cell(length(stiffnesses),1);
h = gobjects(length(stiffnesses),1);
AllData = table();

for nk = 1:length(N_Kuhns)
    N_Kuhn = N_Kuhns(nk);

    for stf = 1:length(stiffnesses)
        stiffness = stiffnesses(stf);

        % Resolve the compiled bead-spring filenames.
        InputScript(1,Np,D,N_Kuhn,stiffness,kbT,b,dt);
        SetDirAndFileNames;

        if ~isfile(TimeStretchDataFileName) || ~isfile(ForceDataFileName)
            warning('Skipping N=%d, stiffness=%.4g: no successfully compiled pull data.',...
                N_Kuhn,stiffness);
            continue
        end

        datR = load(TimeStretchDataFileName,'-mat');
        datF = load(ForceDataFileName,'-mat');
        if isfield(datF,'successful_samples')
            fprintf('Plotting N=%d, stiffness=%.4g from successful sample(s): %s\n',...
                N_Kuhn,stiffness,mat2str(datF.successful_samples));
        end

        rstarRaw = datR.stretch(:);   % stores r* = r/(N*b)
        fx = datF.fx;

        % Keep only the requested range.
        valid = isfinite(rstarRaw) & rstarRaw >= 0 & rstarRaw <= 1.15 + 1e-8;
        rstarRaw = rstarRaw(valid);
        fx = fx(valid,:,:);

        % Same 0.01 grouping logic used by the original force plot.
        rstarBins = unique(round(rstarRaw,2));
        meanForceN = nan(size(rstarBins));
        semForceN = nan(size(rstarBins));

        for ir = 1:length(rstarBins)
            r0 = rstarBins(ir);
            idx = find(abs(rstarRaw-r0) < 0.01);
            if isempty(idx)
                continue
            end

            % Preserve the original sign convention and bond averaging.
            fSubset = -fx(idx,:,:);
            fAcrossBonds = mean(fSubset,2,'omitnan');

            nTimes = size(fAcrossBonds,1);
            nSamples = size(fAcrossBonds,3);
            fAcrossBonds = reshape(fAcrossBonds,nTimes,nSamples);

            % Use the second half of each hold, as in the original code.
            firstSteady = max(1,round(nTimes/2));
            fPerSample = mean(fAcrossBonds(firstSteady:end,:),1,'omitnan');
            fPerSample = fPerSample(isfinite(fPerSample));

            if ~isempty(fPerSample)
                meanForceN(ir) = mean(fPerSample);
                if numel(fPerSample) > 1
                    semForceN(ir) = std(fPerSample,0)/sqrt(numel(fPerSample));
                else
                    semForceN(ir) = 0;
                end
            end
        end

        keep = isfinite(meanForceN);
        rstarBins = rstarBins(keep);
        meanForceN = meanForceN(keep);
        semForceN = semForceN(keep);
        [rstarBins,ord] = sort(rstarBins);
        meanForceN = meanForceN(ord);
        semForceN = semForceN(ord);

        % % Preserve the original plotted force scaling.
        % meanForce = meanForceN/ForceConversion;
        % semForce = semForceN/ForceConversion;

        % Dimensionless force: f* = f b/(kBT)
        b_SI = b*LengthConversion;

        meanForce = meanForceN*b_SI/kbT;
        semForce  = semForceN*b_SI/kbT;

        % Save exactly the force-vs-r* data.
        T = table(rstarBins,meanForce,semForce,meanForceN,semForceN,...
            'VariableNames',{'r_star','force_mean','force_sem','force_mean_N','force_sem_N'});
        csvName = fullfile('Force Data',sprintf('Force_vs_rstar.N%d.Stiff%.2f.csv',N_Kuhn,stiffness));
        writetable(T,csvName);

        Tcombined = addvars(T,repmat(stiffness,height(T),1),...
            'Before',1,'NewVariableNames','stiffness');
        if width(AllData)==0
            AllData = Tcombined;
        else
            AllData = [AllData; Tcombined]; %#ok<AGROW>
        end

        % Only requested plot: force vs r*.
        % Plot appearance is kept the same as the ORIGINAL force-vs-stretch plot.
        color_stf = colors_stf(stf,:);
        marker_style = marker_styles{stf};
        h(stf) = errorbar(rstarBins,meanForce,semForce);
        h(stf).Marker = marker_style;
        h(stf).MarkerFaceColor = color_stf;
        h(stf).MarkerEdgeColor = 'k';
        h(stf).Color = 'k';
        h(stf).LineStyle = 'none';
        legendEntries{stf} = ['$K=$ ',num2str(stiffness),' $k_b T/b^2$'];
    end
end

if height(AllData)==0
    warning('No successful pull data were available. No force-vs-r* plot or CSV was created.');
    close(figure(3));
    return
end

writetable(AllData,fullfile('Force Data',sprintf('Force_vs_rstar.N%d.ALL.csv',N_Kuhns(1))));

%%
% Pade approximation
rstarPade = linspace(0.001,0.99,500);
forcePade = rstarPade.*(3-rstarPade.^2)./(1-rstarPade.^2);
hPade = plot(rstarPade,forcePade,'k--','LineWidth',1.5);

set(gca,'FontSize',FontSize/1.5)
set(gcf,'color','w')
xlabel('$r^*=r/(Nb)$','FontSize',FontSize,'Interpreter','latex')
ylabel('$\bar{f}/(k_b T/b)$','FontSize',FontSize,'Interpreter','latex')

% validHandles = isgraphics(h);
% l = legend(h(validHandles),legendEntries(validHandles));

validHandles = isgraphics(h);

l = legend([h(validHandles); hPade], ...
    [legendEntries(validHandles); {'Padé approximation'}]);

l.FontSize = FontSize/1.75;
l.Location = 'Northwest';
l.Interpreter = 'latex';
pbaspect([1 1 1])
xlim([0 1.15])
ylim([0 35])
title(['$N=$',num2str(N_Kuhns(1))],'FontSize',FontSize/2,'Interpreter','latex')

saveas(gcf,fullfile('Output Plots',sprintf('Force_vs_rstar.N%d.png',N_Kuhns(1))))
saveas(gcf,fullfile('Output Plots',sprintf('Force_vs_rstar.N%d.fig',N_Kuhns(1))))

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function colors = DefinePlotColors(color1,color2,ncolors)

rinterp = (linspace(color1(1),color2(1),ncolors))';
ginterp = (linspace(color1(2),color2(2),ncolors))';
binterp = (linspace(color1(3),color2(3),ncolors))';
colors = [rinterp ginterp binterp];

end
