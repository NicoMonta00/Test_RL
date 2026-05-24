% =========================================================
% test_all_agents_random_training_domain.m
% Test all SAC agents found in the repository on 100 random
% initial conditions sampled from the same IC domain used in main_SAC.m.
%
% Training IC domain replicated from localResetFcn in main_SAC.m:
%   X0 = [-650; 0; -450] + rho*d, rho in [0,10], d random unit vector
%   V0 = [8*cos(psi); 8*sin(psi); 5]
%   accepted only if angle(V0_xy, target_xy - X0_xy) <= 75 deg
%   target = [0; 0; 0]
%
% Output expected from Simulink:
%   The final position is read from the To Workspace variable "xe".
%   If "xe" is not found, the script also tries common fallback names.
%
% The script saves:
%   test_results/all_agents_random_training_domain_results_*.csv
%   test_results/all_agents_random_training_domain_results_*.mat
%   test_results/all_agents_random_training_domain_summary_*.csv
%   test_results/landing_error_all_cases_*.png/.fig
%   test_results/landing_error_cdf_*.png/.fig
% =========================================================

clc;
close all;

%% ================================================================
%% CONFIG
%% ================================================================
N_TESTS = 100;
R_IC = 10;
MAX_ANGLE_DEG = 75;

RESULTS_DIR = "test_results";

mdl = "SAC_simulink";
agentblk = mdl + "/RL Agent";

% Simulation settings coherent with main_SAC.m.
% The model itself now stops when landing occurs.
Tf = 160;
Ts = 150;
maxsteps = 1;

% Deterministic seed: all agents are tested on the same 100 ICs.
rng(1, "twister");

%% ================================================================
%% SETUP BASE WORKSPACE, MODEL AND ENVIRONMENT
%% ================================================================
setupBaseWorkspace(Tf, Ts);

if ~bdIsLoaded(mdl)
    open_system(mdl);
end

obsInfo = rlNumericSpec([3 1], ...
    LowerLimit = -inf*ones(3,1), ...
    UpperLimit =  inf*ones(3,1));
obsInfo.Name = "observations";

actInfo = rlNumericSpec([1 1], ...
    LowerLimit = 0.88, ...
    UpperLimit = 1.07);
actInfo.Name = "k_L";

env = rlSimulinkEnv(mdl, agentblk, obsInfo, actInfo);
simOptions = rlSimulationOptions(MaxSteps=maxsteps);

if ~exist(RESULTS_DIR, "dir")
    mkdir(RESULTS_DIR);
end

%% ================================================================
%% FIND ALL SAVED AGENTS RECURSIVELY
%% ================================================================
agentFiles = findAllAgentFilesRecursive(pwd, RESULTS_DIR);

if isempty(agentFiles)
    fprintf("\nTrovati 0 agenti da testare.\n");
    fprintf("Controllo atteso: file Agent*.mat dentro sottocartelle della repo.\n");
    return;
end

fprintf("\nTrovati %d agenti da testare:\n", numel(agentFiles));
for iAgent = 1:numel(agentFiles)
    fprintf("  %2d) %s\n", iAgent, agentFiles(iAgent).relpath);
end

%% ================================================================
%% GENERATE FIXED RANDOM TEST SET
%% ================================================================
X0_center = [-650; 0; -450];
V0_ref    = [8; 0; 5];
target    = [0; 0; 0];

IC = struct([]);
for iTest = 1:N_TESTS
    [X0_rand, V0_rand, theta0_ph, heading_err_deg, attempts] = sampleTrainingIC( ...
        X0_center, V0_ref, target, R_IC, MAX_ANGLE_DEG);

    r0 = norm(target(1:2) - X0_rand(1:2));
    h0 = X0_rand(3);

    IC(iTest).test_id         = iTest;
    IC(iTest).X0              = X0_rand;
    IC(iTest).V0              = V0_rand;
    IC(iTest).r0              = r0;
    IC(iTest).h0              = h0;
    IC(iTest).theta0_ph       = theta0_ph;
    IC(iTest).heading_err_deg = heading_err_deg;
    IC(iTest).attempts        = attempts;
    IC(iTest).state_r_h_alpha = sprintf("[%.6f %.6f %.6f]", r0, h0, heading_err_deg);
    IC(iTest).X0_xyz          = sprintf("[%.6f %.6f %.6f]", X0_rand(1), X0_rand(2), X0_rand(3));
end

save(fullfile(RESULTS_DIR, "random_training_domain_IC_set.mat"), "IC");

%% ================================================================
%% TEST LOOP: ALL AGENTS x ALL INITIAL CONDITIONS
%% ================================================================
emptyRow = makeEmptyResultRow();
rows = repmat(emptyRow, 0, 1);

for iAgent = 1:numel(agentFiles)
    agentPath = agentFiles(iAgent).fullpath;
    agentName = agentFiles(iAgent).name;
    agentLabel = agentFiles(iAgent).label;

    fprintf("\n[%d/%d] Carico agente: %s\n", iAgent, numel(agentFiles), agentPath);
    agent = loadAgentFromMat(agentPath);

    for iTest = 1:N_TESTS
        fprintf("  Test %3d/%3d... ", iTest, N_TESTS);

        % Remove old logged variables so every test reads only current data.
        clearLoggedWorkspaceVariables();

        % Assign IC to base workspace because the Simulink model reads them there.
        assignin("base", "X0",        IC(iTest).X0);
        assignin("base", "V0",        IC(iTest).V0);
        assignin("base", "w0",        [0; 0; 0]);
        assignin("base", "euler",     deg2rad([0; 0; 0]));
        assignin("base", "theta0_ph", IC(iTest).theta0_ph);
        assignin("base", "target",    target);

        row = emptyRow;
        row.agent_file = string(agentName);
        row.agent_label = string(agentLabel);
        row.agent_path = string(agentPath);
        row.test_id = iTest;

        % Initial condition, both raw and state-space representation.
        row.initial_X0_xyz = string(IC(iTest).X0_xyz);
        row.initial_state_r_h_alpha = string(IC(iTest).state_r_h_alpha);
        row.r0_m = IC(iTest).r0;
        row.h0_m = IC(iTest).h0;
        row.alpha0_deg = IC(iTest).heading_err_deg;
        row.X0_N_m = IC(iTest).X0(1);
        row.X0_E_m = IC(iTest).X0(2);
        row.X0_D_m = IC(iTest).X0(3);
        row.V0_N_mps = IC(iTest).V0(1);
        row.V0_E_mps = IC(iTest).V0(2);
        row.V0_D_mps = IC(iTest).V0(3);
        row.theta0_ph_rad = IC(iTest).theta0_ph;
        row.ic_sampling_attempts = IC(iTest).attempts;

        try
            experience = sim(env, agent, simOptions);

            row.k_L = extractActionValue(experience);
            [final_pos, usedVarName] = extractFinalPositionFromWorkspace(target);

            row.final_position_source = string(usedVarName);
            row.final_X_m = final_pos(1);
            row.final_Y_m = final_pos(2);
            row.final_Z_m = final_pos(3);
            row.landing_error_radius_m = norm(final_pos(1:2) - target(1:2));
            row.landing_error_3d_m = norm(final_pos - target);
            row.landing_error_percent_of_initial_range = ...
                100 * row.landing_error_radius_m / max(row.r0_m, eps);
            row.reward = extractRewardValue(experience);
            row.status = "ok";

            fprintf("OK | err_xy = %.3f m | err%% = %.3f | k_L = %.4f\n", ...
                row.landing_error_radius_m, ...
                row.landing_error_percent_of_initial_range, ...
                row.k_L);
        catch ME
            row.status = "error";
            row.error_message = string(ME.message);
            fprintf("ERRORE: %s\n", ME.message);
        end

        rows(end+1, 1) = row; %#ok<SAGROW>
    end
end

if isempty(rows)
    fprintf("\nNessun risultato prodotto.\n");
    return;
end

resultsTable = struct2table(rows);

%% ================================================================
%% SUMMARY, PLOTS, SAVE RESULTS
%% ================================================================
timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));

summaryTable = buildSummaryTable(resultsTable);

matFile = fullfile(RESULTS_DIR, "all_agents_random_training_domain_results_" + timestamp + ".mat");
csvFile = fullfile(RESULTS_DIR, "all_agents_random_training_domain_results_" + timestamp + ".csv");
summaryCsvFile = fullfile(RESULTS_DIR, "all_agents_random_training_domain_summary_" + timestamp + ".csv");

save(matFile, "resultsTable", "summaryTable", "IC", "agentFiles");
writetable(resultsTable, csvFile);
writetable(summaryTable, summaryCsvFile);

plotFiles = makeLandingErrorPlots(resultsTable, RESULTS_DIR, timestamp);

fprintf("\n=========================================================\n");
fprintf("Test completato.\n");
fprintf("Risultati MAT:     %s\n", matFile);
fprintf("Risultati CSV:     %s\n", csvFile);
fprintf("Summary CSV:       %s\n", summaryCsvFile);
fprintf("Grafico casi PNG:  %s\n", plotFiles.scatterPng);
fprintf("Grafico CDF PNG:   %s\n", plotFiles.cdfPng);
fprintf("=========================================================\n\n");

disp(summaryTable);

%% ================================================================
%% LOCAL FUNCTIONS
%% ================================================================
function setupBaseWorkspace(Tf, Ts)
    if exist('dataFoil.m', 'file')
        par_phys = dataFoil;
    else
        par_phys = struct;
    end

    par_phys.W = [0; 0; 0];

    assignin('base', 'par_phys',               par_phys);
    assignin('base', 'delta',                  0.03);
    assignin('base', 'b0',                     20.8064);
    assignin('base', 'wc',                     0.57626);
    assignin('base', 'k',                      5.35);
    assignin('base', 'delay',                  0);
    assignin('base', 'T',                      12.5471);
    assignin('base', 'Delta_e',                5.0);
    assignin('base', 'Gamma',                  1.0);
    assignin('base', 'Kc',                     0.5);
    assignin('base', 't_step',                 0.5);
    assignin('base', 't1_min_rad',             -pi);
    assignin('base', 't1_max_rad',              pi);
    assignin('base', 'k0_target',              0.0);
    assignin('base', 'k1_target',              0.0);
    assignin('base', 'meters_end_zf',          15);
    assignin('base', 's_f_time',               2);
    assignin('base', 'mean_vertical_velocity', 7.9);
    assignin('base', 'r_min',                  0);
    assignin('base', 'r_max',                  1200);
    assignin('base', 'h_min',                  -550);
    assignin('base', 'h_max',                  550);
    assignin('base', 'yaw_min',               -20);
    assignin('base', 'yaw_max',               +20);
    assignin('base', 'delta_a_max',            0.03);
    assignin('base', 'max_idx',               200);
    assignin('base', 't_stop',                 Tf);
    assignin('base', 'Ts',                     Ts);
    assignin('base', 'Tf',                     Tf);
    assignin('base', 'rigeneration_step_s',    300);
end

function row = makeEmptyResultRow()
    row.agent_file = "";
    row.agent_label = "";
    row.agent_path = "";
    row.test_id = NaN;

    row.initial_X0_xyz = "";
    row.initial_state_r_h_alpha = "";
    row.r0_m = NaN;
    row.h0_m = NaN;
    row.alpha0_deg = NaN;
    row.X0_N_m = NaN;
    row.X0_E_m = NaN;
    row.X0_D_m = NaN;
    row.V0_N_mps = NaN;
    row.V0_E_mps = NaN;
    row.V0_D_mps = NaN;
    row.theta0_ph_rad = NaN;
    row.ic_sampling_attempts = NaN;

    row.k_L = NaN;
    row.final_position_source = "";
    row.final_X_m = NaN;
    row.final_Y_m = NaN;
    row.final_Z_m = NaN;
    row.landing_error_radius_m = NaN;
    row.landing_error_3d_m = NaN;
    row.landing_error_percent_of_initial_range = NaN;
    row.reward = NaN;
    row.status = "not_run";
    row.error_message = "";
end

function agentFiles = findAllAgentFilesRecursive(rootDir, resultsDir)
    listing = dir(fullfile(rootDir, "**", "Agent*.mat"));

    % Fallback for old MATLAB versions that do not support ** in dir().
    if isempty(listing)
        listing = recursiveDirAgentMat(rootDir);
    end

    agentFiles = struct('name', {}, 'folder', {}, 'fullpath', {}, 'relpath', {}, 'label', {}, 'episode', {});

    rootDirChar = char(rootDir);
    for iFile = 1:numel(listing)
        fullpath = string(fullfile(listing(iFile).folder, listing(iFile).name));
        folderStr = string(listing(iFile).folder);

        if contains(folderStr, string(filesep) + string(resultsDir))
            continue;
        end

        ep = parseEpisodeNumber(listing(iFile).name);
        relpath = erase(fullpath, string(rootDirChar) + string(filesep));
        [parentFolder, ~, ~] = fileparts(fullpath);
        [~, label, ~] = fileparts(parentFolder);

        agentFiles(end+1).name = string(listing(iFile).name); %#ok<AGROW>
        agentFiles(end).folder = folderStr;
        agentFiles(end).fullpath = fullpath;
        agentFiles(end).relpath = relpath;
        agentFiles(end).label = string(label);
        agentFiles(end).episode = ep;
    end

    if ~isempty(agentFiles)
        [~, idx] = sortrows([[agentFiles.episode]' (1:numel(agentFiles))']);
        agentFiles = agentFiles(idx);
    end
end

function listing = recursiveDirAgentMat(rootDir)
    listing = [];
    content = dir(rootDir);

    for i = 1:numel(content)
        name = content(i).name;
        if content(i).isdir
            if strcmp(name, ".") || strcmp(name, "..") || strcmp(name, ".git")
                continue;
            end
            sub = recursiveDirAgentMat(fullfile(content(i).folder, name));
            listing = [listing; sub]; %#ok<AGROW>
        else
            if ~isempty(regexp(name, '^Agent.*\.mat$', 'once'))
                listing = [listing; content(i)]; %#ok<AGROW>
            end
        end
    end
end

function ep = parseEpisodeNumber(fileName)
    tok = regexp(fileName, 'Agent(\d+)\.mat', 'tokens');
    if ~isempty(tok)
        ep = str2double(tok{1}{1});
        return;
    end

    tokFinal = regexp(fileName, 'AgentFINAL_ep(\d+)_', 'tokens');
    if ~isempty(tokFinal)
        ep = str2double(tokFinal{1}{1});
        return;
    end

    ep = inf;
end

function agent = loadAgentFromMat(agentPath)
    S = load(agentPath);

    if isfield(S, "saved_agent")
        agent = S.saved_agent;
    elseif isfield(S, "agent")
        agent = S.agent;
    else
        vars = fieldnames(S);
        error("Nel file %s non trovo variabili 'saved_agent' o 'agent'. Variabili presenti: %s", ...
            agentPath, strjoin(vars, ", "));
    end
end

function [X0_rand, V0_rand, theta0_ph, heading_err_deg, attempts] = sampleTrainingIC( ...
        X0_center, V0_ref, target_pos, R, maxAngleDeg)

    valid_ic = false;
    attempts = 0;

    while ~valid_ic && attempts < 1000
        attempts = attempts + 1;

        d = randn(3,1);
        d = d / norm(d);
        rho = R * rand()^(1/3);
        X0_rand = X0_center + rho * d;

        psi = 2*pi*rand();
        Vh = norm(V0_ref(1:2));
        V0_rand = [Vh*cos(psi); Vh*sin(psi); V0_ref(3)];

        dir_xy = target_pos(1:2) - X0_rand(1:2);
        cos_a = dot(V0_rand(1:2), dir_xy) / (norm(V0_rand(1:2))*norm(dir_xy));
        heading_err_deg = rad2deg(acos(max(-1, min(1, cos_a))));

        valid_ic = heading_err_deg <= maxAngleDeg;
    end

    if ~valid_ic
        error("Impossibile generare una IC valida dopo %d tentativi.", attempts);
    end

    theta0_ph = atan2(V0_rand(2), V0_rand(1));
end

function clearLoggedWorkspaceVariables()
    varsToClear = ["xe", "x", "X", "pos", "position", "k_L", "kL", "action", "act"];
    for iVar = 1:numel(varsToClear)
        evalin("base", sprintf("if exist('%s','var'); clear('%s'); end", varsToClear(iVar), varsToClear(iVar)));
    end
end

function [finalPos, usedVarName] = extractFinalPositionFromWorkspace(target)
    candidateNames = ["xe", "x", "X", "pos", "position"];

    for iName = 1:numel(candidateNames)
        varName = candidateNames(iName);
        existsVar = evalin("base", sprintf("exist('%s','var')", varName));
        if existsVar
            raw = evalin("base", varName);
            try
                finalPos = extractFinalPositionFromLoggedData(raw, target);
                usedVarName = varName;
                return;
            catch
                % Try next candidate.
            end
        end
    end

    error("Non riesco a leggere la posizione finale. Nessuna variabile valida trovata tra: %s", ...
        strjoin(candidateNames, ", "));
end

function finalPos = extractFinalPositionFromLoggedData(raw, target)
    data = raw;

    if isa(data, "timeseries")
        data = data.Data;
    elseif isstruct(data)
        if isfield(data, "signals") && isfield(data.signals, "values")
            data = data.signals.values;
        elseif isfield(data, "Data")
            data = data.Data;
        elseif isfield(data, "Values")
            data = data.Values;
        else
            error("Formato struct non riconosciuto.");
        end
    end

    if istimetable(data) || istable(data)
        data = table2array(data);
    end

    data = squeeze(double(data));

    if isempty(data) || all(isnan(data(:)))
        error("Dati posizione vuoti o tutti NaN.");
    end

    if isvector(data)
        vec = data(:);
        if numel(vec) < 3
            error("Vettore posizione con meno di 3 componenti.");
        end
        finalPos = vec(end-2:end);
        finalPos = finalPos(:);
        return;
    end

    % Remove rows/columns that are entirely NaN.
    data = data(~all(isnan(data), 2), :);
    data = data(:, ~all(isnan(data), 1));

    [nRows, nCols] = size(data);

    if nCols >= 4 && isLikelyTimeColumn(data(:,1))
        posMat = data(:, 2:4);
        finalPos = lastValidRow(posMat).';
    elseif nCols >= 3
        % Most common To Workspace Array format for a 3D vector signal: N x 3.
        posMat = data(:, 1:3);
        finalPos = lastValidRow(posMat).';
    elseif nRows >= 3
        % Transposed case: 3 x N or 4 x N with first row time.
        dataT = data.';
        if size(dataT, 2) >= 4 && isLikelyTimeColumn(dataT(:,1))
            posMat = dataT(:, 2:4);
        else
            posMat = dataT(:, 1:3);
        end
        finalPos = lastValidRow(posMat).';
    else
        error("Formato numerico posizione non interpretabile.");
    end

    finalPos = finalPos(:);

    % If the signal contains extra channels and this heuristic selected an invalid
    % triplet, fail explicitly instead of silently computing a wrong error.
    if numel(finalPos) ~= 3 || any(~isfinite(finalPos))
        error("Posizione finale non valida.");
    end

    % Keep target in signature to make the interpretation explicit.
    %#ok<NASGU>
end

function tf = isLikelyTimeColumn(col)
    col = col(:);
    col = col(isfinite(col));
    if numel(col) < 2
        tf = false;
        return;
    end
    tf = all(diff(col) >= -1e-12) && col(1) >= 0;
end

function row = lastValidRow(mat)
    valid = all(isfinite(mat), 2);
    idx = find(valid, 1, "last");
    if isempty(idx)
        error("Nessuna riga valida nella matrice posizione.");
    end
    row = mat(idx, :);
end

function rewardVal = extractRewardValue(experience)
    rewardVal = NaN;

    try
        if isprop(experience, "Reward") || isfield(experience, "Reward")
            rewardVal = extractLastNumericValue(experience.Reward);
        end
    catch
        rewardVal = NaN;
    end
end

function actionVal = extractActionValue(experience)
    actionVal = NaN;

    try
        if isprop(experience, "Action") || isfield(experience, "Action")
            actionVal = extractLastNumericValue(experience.Action);
            return;
        end
    catch
        actionVal = NaN;
    end

    % Fallback: read action from common workspace variables if available.
    candidateNames = ["k_L", "kL", "action", "act"];
    for iName = 1:numel(candidateNames)
        varName = candidateNames(iName);
        existsVar = evalin("base", sprintf("exist('%s','var')", varName));
        if existsVar
            try
                raw = evalin("base", varName);
                actionVal = extractLastNumericValue(raw);
                return;
            catch
            end
        end
    end
end

function val = extractLastNumericValue(x)
    if iscell(x)
        vals = [];
        for i = 1:numel(x)
            try
                vals(end+1) = extractLastNumericValue(x{i}); %#ok<AGROW>
            catch
            end
        end
        if isempty(vals)
            error("Nessun valore numerico in cell.");
        end
        val = vals(end);
        return;
    end

    if isa(x, "timeseries")
        x = x.Data;
    elseif isstruct(x)
        if isfield(x, "signals") && isfield(x.signals, "values")
            x = x.signals.values;
        elseif isfield(x, "Data")
            x = x.Data;
        elseif isfield(x, "Values")
            x = x.Values;
        else
            error("Struct non numerica.");
        end
    end

    x = squeeze(double(x));
    x = x(isfinite(x));

    if isempty(x)
        error("Nessun valore numerico finito.");
    end

    val = x(end);
end

function summaryTable = buildSummaryTable(resultsTable)
    ok = resultsTable.status == "ok";
    T = resultsTable(ok, :);

    if isempty(T)
        summaryTable = table();
        return;
    end

    labels = unique(T.agent_label, "stable");

    summaryRows = struct([]);
    for i = 1:numel(labels)
        label = labels(i);
        Ti = T(T.agent_label == label, :);
        e = Ti.landing_error_radius_m;
        ep = Ti.landing_error_percent_of_initial_range;
        valid = isfinite(e);
        e = e(valid);
        ep = ep(valid);

        s.agent_label = label;
        s.n_ok = numel(e);
        s.mean_error_m = mean(e, "omitnan");
        s.median_error_m = median(e, "omitnan");
        s.p75_error_m = percentileLocal(e, 75);
        s.p90_error_m = percentileLocal(e, 90);
        s.p95_error_m = percentileLocal(e, 95);
        s.max_error_m = max(e, [], "omitnan");
        s.mean_error_percent = mean(ep, "omitnan");
        s.p75_error_percent = percentileLocal(ep, 75);
        s.p90_error_percent = percentileLocal(ep, 90);
        s.p95_error_percent = percentileLocal(ep, 95);
        s.mean_k_L = mean(Ti.k_L, "omitnan");
        s.min_k_L = min(Ti.k_L, [], "omitnan");
        s.max_k_L = max(Ti.k_L, [], "omitnan");

        summaryRows = [summaryRows; s]; %#ok<AGROW>
    end

    summaryTable = struct2table(summaryRows);
end

function p = percentileLocal(x, q)
    x = x(isfinite(x));
    if isempty(x)
        p = NaN;
        return;
    end

    x = sort(x(:));
    n = numel(x);
    pos = 1 + (q/100) * (n - 1);
    lo = floor(pos);
    hi = ceil(pos);

    if lo == hi
        p = x(lo);
    else
        p = x(lo) + (pos - lo) * (x(hi) - x(lo));
    end
end

function plotFiles = makeLandingErrorPlots(resultsTable, resultsDir, timestamp)
    ok = resultsTable.status == "ok" & isfinite(resultsTable.landing_error_radius_m);
    T = resultsTable(ok, :);

    plotFiles.scatterPng = "";
    plotFiles.cdfPng = "";

    if isempty(T)
        return;
    end

    labels = unique(T.agent_label, "stable");

    fig1 = figure("Name", "Landing error - all cases");
    hold on;
    grid on;
    for i = 1:numel(labels)
        label = labels(i);
        Ti = T(T.agent_label == label, :);
        plot(Ti.test_id, Ti.landing_error_radius_m, "o-", "DisplayName", label);
    end
    xlabel("Test ID");
    ylabel("Errore di atterraggio nel piano XY [m]");
    title("Errore di atterraggio per tutte le condizioni iniziali");
    legend("Location", "best", "Interpreter", "none");
    hold off;

    scatterPng = fullfile(resultsDir, "landing_error_all_cases_" + timestamp + ".png");
    scatterFig = fullfile(resultsDir, "landing_error_all_cases_" + timestamp + ".fig");
    saveas(fig1, scatterPng);
    savefig(fig1, scatterFig);

    fig2 = figure("Name", "Landing error CDF");
    hold on;
    grid on;
    for i = 1:numel(labels)
        label = labels(i);
        Ti = T(T.agent_label == label, :);
        e = sort(Ti.landing_error_radius_m(isfinite(Ti.landing_error_radius_m)));
        F = (1:numel(e)) / numel(e);
        plot(e, F, "LineWidth", 1.5, "DisplayName", label);
    end
    xlabel("Errore di atterraggio nel piano XY [m]");
    ylabel("Frazione cumulativa dei test [-]");
    title("CDF errore di atterraggio: soglie 75%, 90%, 95%");
    yline(0.75, "--", "75%");
    yline(0.90, "--", "90%");
    yline(0.95, "--", "95%");
    legend("Location", "best", "Interpreter", "none");
    hold off;

    cdfPng = fullfile(resultsDir, "landing_error_cdf_" + timestamp + ".png");
    cdfFig = fullfile(resultsDir, "landing_error_cdf_" + timestamp + ".fig");
    saveas(fig2, cdfPng);
    savefig(fig2, cdfFig);

    plotFiles.scatterPng = scatterPng;
    plotFiles.cdfPng = cdfPng;
end
