% =========================================================
% main_SAC.m
% Train SAC Agent for Path-Following Control (Parafoil)
% Parallel training – async – con checkpoint automatico
% e resume da crash.
%
% Architettura directory (INVARIANTE garantita da reorganize/cleanup):
%   savedAgents/                 ← SOLO file relativi del run in corso
%   savedAgents/previous_agents/ ← SOLO file storici a numerazione assoluta
%   savedAgents/AgentFINAL*.mat  ← salvataggio finale con trainingStats
%
% Flusso normale:
%   1. reorganize_checkpoints  → archivia run precedente (se crashato)
%   2. Carica resume da previous_agents/ (non da savedAgents/)
%   3. Training → MATLAB scrive Agent150.mat, Agent300.mat... in savedAgents/
%   4. cleanup_after_training  → archivia run corrente in previous_agents/
%
% Per riprendere dopo crash: imposta RESUME = true
% =========================================================
%% ================================================================
%% PULIZIA INIZIALE
%% ================================================================
clc
% clear all % <-- COMMENTATO: rimosso per permettere riavvii senza svuotare il workspace
close all
%% ================================================================
%% CONFIG PRINCIPALE – modifica solo questa sezione
%% ================================================================
VERBOSE_COMPACT   = true;
RESUME            = true;       % true = riprendi da ultimo checkpoint
CHECKPOINT_DIR    = "savedAgents";
CHECKPOINT_FREQ   = 150;        % salva i pesi ogni N episodi
N_WORKERS         = 8;
maxepisodes       = 120000;
% --- NUOVE FLAG PER TRAINING A LOOP ---
ENABLE_CHUNK_LOOP = true;       % Se true, fa ripartire il training a blocchi
CHUNK_SIZE        = 1500;       % Numero di episodi per ogni interazione del loop
%% ================================================================
%% PRE-TRAINING: archivia checkpoint del run precedente se crashato.
%% Sposta i file relativi da savedAgents/ → previous_agents/ rinominati.
%% savedAgents/ rimane pulita prima di ogni run.
%% ================================================================
reorganize_checkpoints(CHECKPOINT_DIR);
%% ================================================================
%% 1. SETUP WORKSPACE SIMULINK
%% ================================================================
clc
Simulink.sdi.setRecordData(false);
Simulink.sdi.setArchiveRunLimit(1);
warning('off', 'SDI:sdi:SimWithRecordOff');
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
V0 = [8; 0; 5];
assignin('base', 'X0',                   [-650; 0; -450]);
assignin('base', 'V0',                    V0);
assignin('base', 'w0',                   [0; 0; 0]);
assignin('base', 'euler',                 deg2rad([0; 0; 0]));
assignin('base', 'theta0_ph',             atan2(V0(2), V0(1)));
assignin('base', 'target',               [0; 0; 0]);
Tf  = 160;
Ts  = 150;
rigeneration_step_s = 300;
assignin('base', 't_stop',                Tf);
assignin('base', 'Ts',                    Ts);
assignin('base', 'Tf',                    Tf);
assignin('base', 'rigeneration_step_s',   rigeneration_step_s);
previousRngState = rng(0, "twister");
%% ================================================================
%% 2. PARALLEL POOL (con pulizia preventiva worker)
%% ================================================================

% --- Pulizia worker: blocco 1 (parcluster Jobs) ---
try
    myCluster = parcluster('local');
    myCluster.IdleTimeout = Inf;
    p_tmp = gcp('nocreate');
    if isempty(p_tmp)
        p_tmp = parpool(myCluster, N_WORKERS);
    elseif p_tmp.NumWorkers ~= N_WORKERS
        delete(p_tmp);
        p_tmp = parpool(myCluster, N_WORKERS);
    end
catch
    % Nessun job da pulire o profilo non trovato – ignorato
end

% --- Pulizia worker: blocco 2 (cartella job storage) ---
try
    myCluster = parcluster('Processes');
    jobFolder = fullfile(fileparts(prefdir), 'local_cluster_jobs');
    fprintf('Cartella job: %s\n', jobFolder);
    if exist(jobFolder, 'dir')
        rmdir(jobFolder, 's');
        fprintf('[OK] Cartella cancellata.\n');
    else
        fprintf('[INFO] Cartella non trovata – già pulita.\n');
    end
catch
    % Profilo 'Processes' non trovato o cartella non cancellabile – ignorato
end

% --- Setup pool principale ---
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(N_WORKERS, 'IdleTimeout', Inf);
    fprintf('Parallel pool aperta: %d worker.\n', pool.NumWorkers);
elseif pool.NumWorkers ~= N_WORKERS
    delete(pool);
    pool = parpool(N_WORKERS, 'IdleTimeout', Inf);
    fprintf('Parallel pool riaperta: %d worker.\n', pool.NumWorkers);
else
    pool.IdleTimeout = Inf;
    fprintf('Parallel pool già attiva: %d worker. IdleTimeout = Inf.\n', pool.NumWorkers);
end
%% ================================================================
%% 3. ENVIRONMENT
%% ================================================================
action_limits_lower = 0.88;
action_limits_upper = 1.07;
mdl      = "SAC_simulink";
open_system(mdl)
agentblk = mdl + "/RL Agent";
obsInfo = rlNumericSpec([3 1], ...
    LowerLimit = -inf*ones(3,1), UpperLimit = inf*ones(3,1));
obsInfo.Name = "observations";
actInfo = rlNumericSpec([1 1], ...
    LowerLimit = action_limits_lower, UpperLimit = action_limits_upper);
actInfo.Name = "k_L";
X0_center_val = [-650; 0; -450];
V0_ref_val    = [8; 0; 5];
w0_ref_val    = [0; 0; 0];
euler_ref_val = deg2rad([0; 0; 0]);
env = rlSimulinkEnv(mdl, agentblk, obsInfo, actInfo);
env.ResetFcn = @(in) localResetFcn(in, X0_center_val, V0_ref_val, ...
    w0_ref_val, euler_ref_val);
%% ================================================================
%% 4. AGENT – build nuovo o carica da checkpoint
%% ================================================================
L    = 400;
nObs = prod(obsInfo.Dimension);
nAct = prod(actInfo.Dimension);
episodeOffset = 0;
PREV_DIR = fullfile(CHECKPOINT_DIR, "previous_agents");
if RESUME
    if ~exist(PREV_DIR, 'dir')
        error('[RESUME] La cartella previous_agents/ non esiste.\nImposta RESUME=false per partire da zero.');
    end
    listing = dir(fullfile(PREV_DIR, "Agent*.mat"));
    if isempty(listing)
        error('[RESUME] Nessun checkpoint in previous_agents/.\nImposta RESUME=false per partire da zero.');
    end
    epNums = zeros(numel(listing), 1);
    for ii = 1:numel(listing)
        tok = regexp(listing(ii).name, 'Agent(\d+)\.mat', 'tokens');
        if ~isempty(tok)
            epNums(ii) = str2double(tok{1}{1});
        end
    end
    [episodeOffset, bestIdx] = max(epNums);
    checkpointFile = fullfile(PREV_DIR, listing(bestIdx).name);
    fprintf('\n[RESUME] Carico: previous_agents/%s (ep abs %d)\n\n', ...
        listing(bestIdx).name, episodeOffset);
    
    loaded = load(checkpointFile, 'saved_agent');
    agent  = loaded.saved_agent;
    maxepisodes = maxepisodes - episodeOffset;
    if maxepisodes <= 0
        fprintf('Training già completato (ep %d). Uscita.\n', episodeOffset);
        return;
    end
else
    rng(0, "twister");
    criticNet1 = buildCriticNetwork(nObs, nAct, L);
    criticNet2 = buildCriticNetwork(nObs, nAct, L);
    critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, ...
        ObservationInputNames="obsInLyr", ActionInputNames="actInLyr");
    critic2 = rlQValueFunction(criticNet2, obsInfo, actInfo, ...
        ObservationInputNames="obsInLyr", ActionInputNames="actInLyr");
    
    commonPath = [
        featureInputLayer(nObs,   Name="observation")
        fullyConnectedLayer(L,    Name="comFC1")
        reluLayer(                Name="comRelu1")
        fullyConnectedLayer(L,    Name="comFC2")
        reluLayer(                Name="commonPath")
    ];
    meanPath = [
        fullyConnectedLayer(L,    Name="meanFC")
        reluLayer(                Name="meanRelu")
        fullyConnectedLayer(nAct, Name="actionMean")
    ];
    stdPath = [
        fullyConnectedLayer(nAct, Name="stdFC")
        reluLayer(                Name="stdRelu")
        softplusLayer(            Name="actionStd")
    ];
    
    actorNet = dlnetwork();
    actorNet = addLayers(actorNet, commonPath);
    actorNet = addLayers(actorNet, meanPath);
    actorNet = addLayers(actorNet, stdPath);
    actorNet = connectLayers(actorNet, "commonPath", "meanFC/in");
    actorNet = connectLayers(actorNet, "commonPath", "stdFC/in");
    actorNet = initialize(actorNet);
    summary(actorNet)
    
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        ObservationInputNames              = "observation", ...
        ActionMeanOutputNames              = "actionMean",  ...
        ActionStandardDeviationOutputNames = "actionStd");
        
    agentOpts = rlSACAgentOptions( ...
        SampleTime             = Ts,   ...
        DiscountFactor         = 0,    ...
        ExperienceBufferLength = 240, ...
        MiniBatchSize          = 120,  ...
        TargetSmoothFactor     = 0.02, ...
        TargetUpdateFrequency  = 1);
        
    agentOpts.ActorOptimizerOptions.Algorithm         = "adam";
    agentOpts.ActorOptimizerOptions.LearnRate         = 3e-3;
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1;
    for ct = 1:2
        agentOpts.CriticOptimizerOptions(ct).Algorithm         = "adam";
        agentOpts.CriticOptimizerOptions(ct).LearnRate         = 1e-3;
        agentOpts.CriticOptimizerOptions(ct).GradientThreshold = 1;
    end
    
    rng(0, "twister");
    agent = rlSACAgent(actor, [critic1, critic2], agentOpts);
end
%% ================================================================
%% 5. TRAINING OPTIONS (Configurazione base)
%% ================================================================
maxsteps = 1;
if ~exist(CHECKPOINT_DIR, 'dir'); mkdir(CHECKPOINT_DIR); end
trainingOpts = rlTrainingOptions( ...
    MaxEpisodes           = maxepisodes,          ...
    MaxStepsPerEpisode    = maxsteps,             ...
    Verbose               = false,                ...
    Plots                 = "training-progress",  ...
    StopTrainingCriteria  = "EvaluationStatistic",...
    StopTrainingValue     = 1600,                 ...
    StopOnError           = "off",                ...
    SimulationStorageType = "none",               ...
    UseParallel           = true,                 ...
    SaveAgentCriteria     = "EpisodeFrequency",   ...
    SaveAgentValue        = CHECKPOINT_FREQ,      ...
    SaveAgentDirectory    = CHECKPOINT_DIR);
trainingOpts.ParallelizationOptions.Mode                           = "async";
trainingOpts.ParallelizationOptions.WorkerRandomSeeds              = -1;
trainingOpts.ParallelizationOptions.TransferBaseWorkspaceVariables = "on";
%% ================================================================
%% 6. LOGGER
%% ================================================================
evl    = rlEvaluator(EvaluationFrequency=64, NumEpisodes=10);
logger = rlDataLogger();
%% ================================================================
%% 7. TRAINING
%% ================================================================
if RESUME
    rng('shuffle'); % Se riprendiamo, genera scenari totalmente nuovi
else
    rng(0, "twister"); % Se partiamo da zero, manteniamo il run riproducibile
end
% Variabile per tracciare quanti episodi abbiamo fatto realmente in questo run
total_run_eps = 0; 
if ENABLE_CHUNK_LOOP
    fprintf('\n=== Inizio Training a LOOP: %d ep rimanenti totali, chunk da %d ep ===\n\n', maxepisodes, CHUNK_SIZE);
    episodes_done = 0;
    
    while episodes_done < maxepisodes
        current_chunk = min(CHUNK_SIZE, maxepisodes - episodes_done);
        current_offset = episodeOffset + episodes_done;
        
        fprintf('\n--- Avvio Iterazione Loop: %d episodi (Episodi totali salvati finora: %d) ---\n', ...
            current_chunk, current_offset);

        % --------------------------------------------------------
        % PULIZIA WORKER all'inizio di ogni chunk
        % I try-catch silenziosi ignorano gli errori "nulla da pulire"
        % --------------------------------------------------------
        try
            myCluster = parcluster('local');
            myCluster.IdleTimeout = Inf;
            p_clean = gcp('nocreate');
            if isempty(p_clean)
                p_clean = parpool(myCluster, N_WORKERS);
            elseif p_clean.NumWorkers ~= N_WORKERS
                delete(p_clean);
                p_clean = parpool(myCluster, N_WORKERS);
            end
        catch
            % Nessun job da pulire – ignorato
        end
        try
            myCluster = parcluster('Processes');
            jobFolder = fullfile(fileparts(prefdir), 'local_cluster_jobs');
            if exist(jobFolder, 'dir')
                rmdir(jobFolder, 's');
                fprintf('[Cleanup] Cartella job rimossa.\n');
            end
        catch
            % Profilo Processes non trovato o cartella già pulita – ignorato
        end
        % --------------------------------------------------------

        loop_logger = rlDataLogger();
        loop_logger.EpisodeFinishedFcn = @(data) episodeDebugFcn(data, agent, ...
            VERBOSE_COMPACT, current_chunk, current_offset);
            
        trainingOpts.MaxEpisodes = current_chunk;
        Simulink.sdi.clear;
            
        trainingStats = train(agent, env, trainingOpts, Evaluator=evl, Logger=loop_logger);
        
        % --- CONTROLLO STOP MANUALE O OBIETTIVO RAGGIUNTO ---
        actual_episodes = length(trainingStats.EpisodeIndex);
        
        % Salvataggio dell'ultimo agente effettivo del blocco
        if actual_episodes > 0
            chunk_last_file = fullfile(CHECKPOINT_DIR, sprintf("Agent%d.mat", actual_episodes));
            if ~exist(chunk_last_file, 'file')
                saved_agent = agent;
                save(chunk_last_file, "saved_agent");
            end
        end
        
        episodes_done = episodes_done + actual_episodes;
        total_run_eps = episodes_done;
        
        cleanup_after_training(CHECKPOINT_DIR, current_offset);
        
        % --- CHIUSURA E RIAPERTURA POOL tra chunk ---
        if episodes_done < maxepisodes
            fprintf('\n--- Chiusura pool per pulizia worker... ---\n');
            try
                p_end = gcp('nocreate');
                if ~isempty(p_end)
                    delete(p_end);
                    fprintf('[OK] Pool chiusa.\n');
                end
            catch
                % Pool già chiusa o non esistente – ignorato
            end

            % Pulizia cartella job dopo shutdown del pool
            try
                myCluster = parcluster('local');
                myCluster.IdleTimeout = Inf;
                p_clean2 = gcp('nocreate');
                if isempty(p_clean2)
                    p_clean2 = parpool(myCluster, N_WORKERS);
                elseif p_clean2.NumWorkers ~= N_WORKERS
                    delete(p_clean2);
                    p_clean2 = parpool(myCluster, N_WORKERS);
                end
            catch
                % Ignorato
            end
            try
                myCluster2 = parcluster('Processes');
                jobFolder2 = fullfile(fileparts(prefdir), 'local_cluster_jobs');
                if exist(jobFolder2, 'dir')
                    rmdir(jobFolder2, 's');
                    fprintf('[Cleanup post-shutdown] Cartella job rimossa.\n');
                end
            catch
                % Ignorato
            end

            fprintf('--- Raffreddamento (5 sec)... ---\n');
            pause(5);

            % Riapertura pool con worker freschi
            try
                pool = parpool(N_WORKERS, 'IdleTimeout', Inf);
                fprintf('[OK] Pool riaperta: %d worker.\n', pool.NumWorkers);
            catch ME
                fprintf('[WARN] Riapertura pool fallita: %s\n', ME.message);
            end
        end
        % --- FINE GESTIONE POOL ---

        % Se ha fatto meno episodi di quelli richiesti, ferma il loop while!
        if actual_episodes < current_chunk
            fprintf('\n[!] Training interrotto in anticipo (Stop manuale o Reward raggiunto).\n');
            fprintf('    Uscita definitiva dal ciclo a blocchi...\n');
            break; 
        end
        
    end
else
    fprintf('\n=== Training Standard: %d ep rimanenti | offset=%d | checkpoint ogni %d ep ===\n\n', ...
        maxepisodes, episodeOffset, CHECKPOINT_FREQ);
        
    Simulink.sdi.clear;
    logger.EpisodeFinishedFcn = @(data) episodeDebugFcn(data, agent, ...
        VERBOSE_COMPACT, maxepisodes, episodeOffset);
        
    trainingStats = train(agent, env, trainingOpts, Evaluator=evl, Logger=logger);
    
    actual_episodes = length(trainingStats.EpisodeIndex);
    total_run_eps = actual_episodes;
    
    if actual_episodes > 0
        last_file = fullfile(CHECKPOINT_DIR, sprintf("Agent%d.mat", actual_episodes));
        if ~exist(last_file, 'file')
            saved_agent = agent;
            save(last_file, "saved_agent");
        end
    end
    
    cleanup_after_training(CHECKPOINT_DIR, episodeOffset);
end
%% ================================================================
%% POST-TRAINING: Salvataggio finale
%% ================================================================
if ~exist(PREV_DIR, 'dir'); mkdir(PREV_DIR); end
final_ep_absolute = episodeOffset + total_run_eps;
finalFile = fullfile(PREV_DIR, sprintf("Agent%d.mat", final_ep_absolute));
saved_agent = agent; %#ok<NASGU>
save(finalFile, "saved_agent", "trainingStats");
fprintf('\n[OK] Training terminato del tutto! Agent conclusivo salvato in: %s\n', finalFile);
%% ================================================================
%% POST-TRAINING: Salvataggio finale (con timestamp)
%% ================================================================
finalFile = fullfile(CHECKPOINT_DIR, ...
    sprintf("AgentFINAL_ep%d_%s.mat", ...
    episodeOffset + maxepisodes, datestr(now, 'yyyymmdd_HHMMSS')));
saved_agent = agent; %#ok<NASGU>
save(finalFile, "saved_agent", "trainingStats");
fprintf('\n[OK] Agent finale salvato: %s\n', finalFile);
%% ================================================================
%% 8. SIMULAZIONE FINALE
%% ================================================================
rng(0, "twister");
simOptions = rlSimulationOptions(MaxSteps=maxsteps);
experience = sim(env, agent, simOptions);
rng(previousRngState);
%% ================================================================
%% FUNZIONI LOCALI
%% ================================================================
function net = buildCriticNetwork(nObs, nAct, L)
    obsPath = [
        featureInputLayer(nObs,    Name="obsInLyr")
        concatenationLayer(1, 2,   Name="concat")
        fullyConnectedLayer(L,     Name="fc1")
        reluLayer(                 Name="relu1")
        fullyConnectedLayer(L,     Name="fc2")
        reluLayer(                 Name="relu2")
        fullyConnectedLayer(1,     Name="QValLyr")
    ];
    actPath = featureInputLayer(nAct, Name="actInLyr");
    net = dlnetwork();
    net = addLayers(net, obsPath);
    net = addLayers(net, actPath);
    net = connectLayers(net, "actInLyr", "concat/in2");
    net = initialize(net);
end
function in = localResetFcn(in, X0_center, V0_ref, w0_ref, euler_ref)
    R = 10; MAX_ANGLE_DEG = 75;
    target_pos = [0; 0; 0];
    valid_ic = false; attempts = 0;
    while ~valid_ic && attempts < 1000
        attempts = attempts + 1;
        d       = randn(3,1); d = d / norm(d);
        rho     = R * rand()^(1/3);
        X0_rand = X0_center + rho * d;
        psi     = 2*pi*rand();
        Vh      = norm(V0_ref(1:2));
        V0_rand = [Vh*cos(psi); Vh*sin(psi); V0_ref(3)];
        dir_xy  = target_pos(1:2) - X0_rand(1:2);
        cos_a   = dot(V0_rand(1:2), dir_xy) / (norm(V0_rand(1:2))*norm(dir_xy));
        if rad2deg(acos(max(-1, min(1, cos_a)))) <= MAX_ANGLE_DEG
            valid_ic = true;
        end
    end
    if attempts >= 1000
        warning('Impossibile trovare IC valida dopo 1000 tentativi.');
    end
    in = setVariable(in, "X0",        X0_rand);
    in = setVariable(in, "V0",        V0_rand);
    in = setVariable(in, "w0",        w0_ref);
    in = setVariable(in, "euler",     euler_ref);
    in = setVariable(in, "theta0_ph", atan2(V0_rand(2), V0_rand(1)));
    in = setVariable(in, "target",    target_pos);
end
function dataToLog = episodeDebugFcn(data, agent, verbose, maxEp, epOffset)
    ep    = data.EpisodeCount;
    rew   = data.EpisodeInfo.CumulativeReward;
    epAbs = ep + epOffset;
    if verbose
        lastExp = data.Experience(end);
        act_val = lastExp.Action{1};
        obs_val = lastExp.Observation{1};
        mu      = getAction(getActor(agent), {obs_val});
        mu      = mu{1};
        fprintf('Ep: %d/%d (abs:%d) | R: %.2f | act=%.4f (mu=%.4f)\n', ...
            ep, maxEp, epAbs, rew, act_val, mu);
    end
    dataToLog.EpisodeReward   = rew;
    dataToLog.EpisodeAbsolute = epAbs;
end