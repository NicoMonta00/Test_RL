% =========================================================
% test_all_agents_random_training_domain.m
% Test all saved SAC agents on 100 random initial conditions
% sampled from the same initial-state domain used in main_SAC.m.
%
% Training IC domain replicated from localResetFcn in main_SAC.m:
%   X0 = [-650; 0; -450] + rho*d, rho in [0,10], d random unit vector
%   V0 = [8*cos(psi); 8*sin(psi); 5]
%   accepted only if angle(V0_xy, target_xy - X0_xy) <= 75 deg
%   target = [0; 0; 0]
%
% The script:
%   1. Finds all Agent*.mat files in savedAgents/ and savedAgents/previous_agents/
%   2. Generates N_TESTS random ICs in the training domain
%   3. Simulates every loaded agent on every IC
%   4. Saves a results table in test_results/
% =========================================================

clc;
close all;

%% ================================================================
%% CONFIG
%% ================================================================
N_TESTS = 100;
R_IC = 10;
MAX_ANGLE_DEG = 75;

CHECKPOINT_DIR = "savedAgents";
PREV_DIR = fullfile(CHECKPOINT_DIR, "previous_agents");
RESULTS_DIR = "test_results";

mdl = "SAC_simulink";
agentblk = mdl + "/RL Agent";

% Use the same simulation horizon / step setting used in main_SAC.m
Tf = 160;
Ts = 150;
maxsteps = 1;

% Deterministic random seed for repeatable test set
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
%% FIND ALL SAVED AGENTS
%% ================================================================
agentFiles = findAllAgentFiles(CHECKPOINT_DIR, PREV_DIR);

if isempty(agentFiles)
    error("Nessun file Agent*.mat trovato in %s o %s.", CHECKPOINT_DIR, PREV_DIR);
end

fprintf("\nTrovati %d agenti da testare.\n", numel(agentFiles));

%% ================================================================
%% GENERATE FIXED RANDOM TEST SET
%% ================================================================
X0_center = [-650; 0; -450];
V0_ref    = [8; 0; 5];
target    = [0; 0; 0];

IC = struct([]);
for iTest = 1:N_TESTS
    [X0_rand, V0_rand, theta0_ph, angle_deg, attempts] = sampleTrainingIC( ...
        X0_center, V0_ref, target, R_IC, MAX_ANGLE_DEG);

    IC(iTest).test_id   = iTest;
    IC(iTest).X0        = X0_rand;
    IC(iTest).V0        = V0_rand;
    IC(iTest).theta0_ph = theta0_ph;
    IC(iTest).angle_deg = angle_deg;
    IC(iTest).attempts  = attempts;
end

save(fullfile(RESULTS_DIR, "random_training_domain_IC_set.mat"), "IC");

%% ================================================================
%% TEST LOOP: ALL AGENTS x ALL INITIAL CONDITIONS
%% ================================================================
rows = [];
rowIdx = 0;

for iAgent = 1:numel(agentFiles)
    agentPath = agentFiles(iAgent).fullpath;
    agentName = agentFiles(iAgent).name;

    fprintf("\n[%d/%d] Carico agente: %s\n", iAgent, numel(agentFiles), agentPath);
    agent = loadAgentFromMat(agentPath);

    for iTest = 1:N_TESTS
        fprintf("  Test %3d/%3d... ", iTest, N_TESTS);

        % Assign IC to base workspace because the Simulink model reads them there
        assignin("base", "X0",        IC(iTest).X0);
        assignin("base", "V0",        IC(iTest).V0);
        assignin("base", "w0",        [0; 0; 0]);
        assignin("base", "euler",     deg2rad([0; 0; 0]));
        assignin("base", "theta0_ph", IC(iTest).theta0_ph);
        assignin("base", "target",    target);

        rowIdx = rowIdx + 1;
        row.agent_file = string(agentName);
        row.agent_path = string(agentPath);
        row.test_id = iTest;
        row.X0_N = IC(iTest).X0(1);
        row.X0_E = IC(iTest).X0(2);
        row.X0_D = IC(iTest).X0(3);
        row.V0_N = IC(iTest).V0(1);
        row.V0_E = IC(iTest).V0(2);
        row.V0_D = IC(iTest).V0(3);
        row.theta0_ph = IC(iTest).theta0_ph;
        row.initial_heading_error_deg = IC(iTest).angle_deg;
        row.ic_sampling_attempts = IC(iTest).attempts;
        row.reward = NaN;
        row.action_kL = NaN;
        row.status = "not_run";
        row.error_message = "";

        try
            experience = sim(env, agent, simOptions);
            [rewardVal, actionVal] = extractRewardAndAction(experience);

            row.reward = rewardVal;
            row.action_kL = actionVal;
            row.status = "ok";
            fprintf("OK | R = %.4g | k_L = %.4f\n", row.reward, row.action_kL);
        catch ME
            row.status = "error";
            row.error_message = string(ME.message);
            fprintf("ERRORE: %s\n", ME.message);
        end

        rows = [rows; row]; %#ok<AGROW>
    end
end

resultsTable = struct2table(rows);

%% ================================================================
%% SAVE RESULTS
%% ================================================================
timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
matFile = fullfile(RESULTS_DIR, "all_agents_random_training_domain_results_" + timestamp + ".mat");
csvFile = fullfile(RESULTS_DIR, "all_agents_random_training_domain_results_" + timestamp + ".csv");

save(matFile, "resultsTable", "IC", "agentFiles");
writetable(resultsTable, csvFile);

fprintf("\n=========================================================\n");
fprintf("Test completato.\n");
fprintf("Risultati MAT: %s\n", matFile);
fprintf("Risultati CSV: %s\n", csvFile);
fprintf("=========================================================\n");

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

function agentFiles = findAllAgentFiles(checkpointDir, previousDir)
    folders = string.empty;
    if exist(checkpointDir, "dir")
        folders(end+1) = string(checkpointDir); %#ok<AGROW>
    end
    if exist(previousDir, "dir")
        folders(end+1) = string(previousDir); %#ok<AGROW>
    end

    agentFiles = struct('name', {}, 'folder', {}, 'fullpath', {}, 'episode', {});

    for iFolder = 1:numel(folders)
        listing = dir(fullfile(folders(iFolder), "Agent*.mat"));
        for iFile = 1:numel(listing)
            fullpath = fullfile(listing(iFile).folder, listing(iFile).name);
            ep = parseEpisodeNumber(listing(iFile).name);

            agentFiles(end+1).name = string(listing(iFile).name); %#ok<AGROW>
            agentFiles(end).folder = string(listing(iFile).folder);
            agentFiles(end).fullpath = string(fullpath);
            agentFiles(end).episode = ep;
        end
    end

    if ~isempty(agentFiles)
        [~, idx] = sort([agentFiles.episode]);
        agentFiles = agentFiles(idx);
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

function [X0_rand, V0_rand, theta0_ph, angle_deg, attempts] = sampleTrainingIC( ...
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
        angle_deg = rad2deg(acos(max(-1, min(1, cos_a))));

        valid_ic = angle_deg <= maxAngleDeg;
    end

    if ~valid_ic
        error("Impossibile generare una IC valida dopo %d tentativi.", attempts);
    end

    theta0_ph = atan2(V0_rand(2), V0_rand(1));
end

function [rewardVal, actionVal] = extractRewardAndAction(experience)
    rewardVal = NaN;
    actionVal = NaN;

    try
        if isprop(experience, "Reward") || isfield(experience, "Reward")
            r = experience.Reward;
            if iscell(r)
                rewardVal = sum(cellfun(@double, r));
            else
                rewardVal = sum(double(r(:)));
            end
        end
    catch
        rewardVal = NaN;
    end

    try
        if isprop(experience, "Action") || isfield(experience, "Action")
            a = experience.Action;
            if iscell(a)
                actionVal = double(a{1}(end));
            else
                actionVal = double(a(end));
            end
        end
    catch
        actionVal = NaN;
    end
end
