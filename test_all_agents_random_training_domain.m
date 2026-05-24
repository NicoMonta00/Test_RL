% =========================================================
% test_all_agents_random_training_domain.m
%
% Batch test delle policy SAC salvate nella repository.
%
% Regole operative:
%   1. Il modello simulato è sempre SAC_simulink.
%   2. Il blocco Policy mantiene sempre lo stesso file: blockAgentData.mat.
%   3. Per ogni Agent*.mat viene rigenerato blockAgentData.mat.
%   4. Ogni agente viene testato su 100 condizioni iniziali random nel dominio
%      di training.
%   5. Prima di simulare si diagnostica la dimensione observation della policy.
%      Se la policy richiede più di 3 osservazioni, vengono aggiunti zeri.
%   6. Se una qualunque simulazione fallisce, il programma si ferma subito.
%      Viene salvato un CSV parziale e poi l'errore viene rilanciato.
%   7. Il k_L scelto dalla Policy viene salvato tramite un To Workspace
%      automatico collegato all'uscita del blocco Policy.
% =========================================================

clc;
close all;

%% CONFIG
N_TESTS = 100;
R_IC = 10;
MAX_ANGLE_DEG = 75;
MODEL_OBSERVATION_DIM = 3;

RESULTS_DIR = "test_results";
POLICY_DATA_FILE = "blockAgentData.mat";
ACTION_TO_WORKSPACE_VAR = "k_L_policy";

mdl = "SAC_simulink";
policyBlock = mdl + "/Policy";

Tf = 160;
Ts = 150;
rng(1,"twister");

if ~exist(RESULTS_DIR,"dir")
    mkdir(RESULTS_DIR);
end

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));
partialCsvFile = fullfile(RESULTS_DIR,"PARTIAL_all_agents_random_training_domain_results_" + timestamp + ".csv");
finalCsvFile   = fullfile(RESULTS_DIR,"all_agents_random_training_domain_results_" + timestamp + ".csv");
summaryCsvFile = fullfile(RESULTS_DIR,"all_agents_random_training_domain_summary_" + timestamp + ".csv");
matFile        = fullfile(RESULTS_DIR,"all_agents_random_training_domain_results_" + timestamp + ".mat");
fatalReportFile = fullfile(RESULTS_DIR,"FATAL_batch_report_" + timestamp + ".txt");

%% SETUP MODEL
setupBaseWorkspace(Tf,Ts);

if ~bdIsLoaded(mdl)
    open_system(mdl);
end

prepare_sac_simulink_batch_model(mdl);
policyBlock = findPolicyBlock(mdl, policyBlock);
ensurePolicyActionToWorkspace(mdl, policyBlock, ACTION_TO_WORKSPACE_VAR);

fprintf("Modello usato: %s\n", mdl);
fprintf("Policy block usato: %s\n", policyBlock);
fprintf("Policy MAT file fisso: %s\n", POLICY_DATA_FILE);
fprintf("Variabile To Workspace per k_L: %s\n", ACTION_TO_WORKSPACE_VAR);

%% FIND AGENTS
agentFiles = findAllAgentFilesRecursive(pwd, RESULTS_DIR);
if isempty(agentFiles)
    error("Nessun Agent*.mat trovato nelle sottocartelle della repository.");
end

fprintf("\nTrovati %d agenti da testare:\n", numel(agentFiles));
for iAgent = 1:numel(agentFiles)
    fprintf("  %2d) %s\n", iAgent, agentFiles(iAgent).relpath);
end

%% GENERATE SAME 100 INITIAL CONDITIONS FOR ALL AGENTS
X0_center = [-650;0;-450];
V0_ref = [8;0;5];
target = [0;0;0];

IC = struct([]);
for iTest = 1:N_TESTS
    [X0_rand,V0_rand,theta0_ph,alpha_deg,attempts] = sampleTrainingIC( ...
        X0_center,V0_ref,target,R_IC,MAX_ANGLE_DEG);

    r0 = norm(target(1:2)-X0_rand(1:2));
    h0 = X0_rand(3);

    IC(iTest).test_id = iTest;
    IC(iTest).X0 = X0_rand;
    IC(iTest).V0 = V0_rand;
    IC(iTest).theta0_ph = theta0_ph;
    IC(iTest).r0 = r0;
    IC(iTest).h0 = h0;
    IC(iTest).alpha_deg = alpha_deg;
    IC(iTest).attempts = attempts;
    IC(iTest).state_r_h_alpha = sprintf("[%.6f %.6f %.6f]",r0,h0,alpha_deg);
    IC(iTest).X0_xyz = sprintf("[%.6f %.6f %.6f]",X0_rand(1),X0_rand(2),X0_rand(3));
end
save(fullfile(RESULTS_DIR,"random_training_domain_IC_set.mat"),"IC");

%% TEST LOOP
rows = repmat(makeEmptyRow(),0,1);

try
    for iAgent = 1:numel(agentFiles)
        agentPath = agentFiles(iAgent).fullpath;
        agentName = agentFiles(iAgent).name;
        agentLabel = agentFiles(iAgent).label;

        fprintf("\n[%d/%d] Agente: %s\n",iAgent,numel(agentFiles),agentPath);
        agent = loadAgentFromMat(agentPath);

        [policyObsDim,obsDescription] = diagnose_policy_observation_dimension(agent);
        nPad = configure_policy_observation_padding(mdl,policyBlock,policyObsDim,MODEL_OBSERVATION_DIM);
        fprintf("  Observation policy: %d | modello: %d | padding zeri: %d | %s\n", ...
            policyObsDim,MODEL_OBSERVATION_DIM,nPad,obsDescription);

        regenerateFixedPolicyDataFile(agent,POLICY_DATA_FILE,mdl);
        prepare_sac_simulink_batch_model(mdl);
        ensurePolicyActionToWorkspace(mdl, policyBlock, ACTION_TO_WORKSPACE_VAR);

        for iTest = 1:N_TESTS
            fprintf("  Test %3d/%3d... ",iTest,N_TESTS);

            clearLoggedWorkspaceVariables(ACTION_TO_WORKSPACE_VAR);
            assignInitialCondition(IC(iTest),target);

            simOut = sim(mdl,"StopTime",num2str(Tf),"ReturnWorkspaceOutputs","on");

            [finalPos,sourceName] = extractFinalPosition(simOut);
            kL = extractActionValue(simOut,ACTION_TO_WORKSPACE_VAR);
            errXY = norm(finalPos(1:2)-target(1:2));
            err3D = norm(finalPos-target);
            errPercent = 100*errXY/max(IC(iTest).r0,eps);

            row = makeEmptyRow();
            row.agent_file = string(agentName);
            row.agent_label = string(agentLabel);
            row.agent_path = string(agentPath);
            row.policy_data_file = string(POLICY_DATA_FILE);
            row.policy_observation_dim = policyObsDim;
            row.model_observation_dim = MODEL_OBSERVATION_DIM;
            row.zero_padding_dim = nPad;
            row.observation_description = string(obsDescription);
            row.test_id = iTest;
            row.initial_X0_xyz = string(IC(iTest).X0_xyz);
            row.initial_state_r_h_alpha = string(IC(iTest).state_r_h_alpha);
            row.r0_m = IC(iTest).r0;
            row.h0_m = IC(iTest).h0;
            row.alpha0_deg = IC(iTest).alpha_deg;
            row.X0_N_m = IC(iTest).X0(1);
            row.X0_E_m = IC(iTest).X0(2);
            row.X0_D_m = IC(iTest).X0(3);
            row.V0_N_mps = IC(iTest).V0(1);
            row.V0_E_mps = IC(iTest).V0(2);
            row.V0_D_mps = IC(iTest).V0(3);
            row.theta0_ph_rad = IC(iTest).theta0_ph;
            row.ic_sampling_attempts = IC(iTest).attempts;
            row.k_L = kL;
            row.final_position_source = string(sourceName);
            row.final_X_m = finalPos(1);
            row.final_Y_m = finalPos(2);
            row.final_Z_m = finalPos(3);
            row.landing_error_radius_m = errXY;
            row.landing_error_3d_m = err3D;
            row.landing_error_percent_of_initial_range = errPercent;
            row.status = "ok";

            rows(end+1,1) = row; %#ok<SAGROW>
            writePartialResults(rows,partialCsvFile);

            fprintf("OK | errXY = %.3f m | err%% = %.3f | k_L = %.5g\n",errXY,errPercent,kL);
        end
    end
catch ME
    fprintf("\n[ERRORE FATALE] Batch interrotto alla prima simulazione fallita.\n");
    fprintf("Messaggio: %s\n",ME.message);
    writeFatalReport(ME,fatalReportFile);
    fprintf("Report errore salvato in: %s\n",fatalReportFile);
    if ~isempty(rows)
        writePartialResults(rows,partialCsvFile);
        fprintf("Risultati parziali salvati in: %s\n",partialCsvFile);
    end
    rethrow(ME);
end

%% SAVE FINAL RESULTS
resultsTable = struct2table(rows);
summaryTable = buildSummaryTable(resultsTable);

save(matFile,"resultsTable","summaryTable","IC","agentFiles");
writetable(resultsTable,finalCsvFile);
writetable(summaryTable,summaryCsvFile);
makeLandingErrorPlots(resultsTable,RESULTS_DIR,timestamp);

fprintf("\n=========================================================\n");
fprintf("Test completato senza crash.\n");
fprintf("Risultati MAT: %s\n",matFile);
fprintf("Risultati CSV: %s\n",finalCsvFile);
fprintf("Summary CSV:   %s\n",summaryCsvFile);
fprintf("=========================================================\n");
disp(summaryTable);

%% LOCAL FUNCTIONS
function setupBaseWorkspace(Tf,Ts)
    if exist('dataFoil.m','file')
        par_phys = dataFoil;
    else
        par_phys = struct;
    end
    par_phys.W = [0;0;0];

    assignin('base','par_phys',par_phys);
    assignin('base','delta',0.03);
    assignin('base','b0',20.8064);
    assignin('base','wc',0.57626);
    assignin('base','k',5.35);
    assignin('base','delay',0);
    assignin('base','T',12.5471);
    assignin('base','Delta_e',5.0);
    assignin('base','Gamma',1.0);
    assignin('base','Kc',0.5);
    assignin('base','t_step',0.5);
    assignin('base','t1_min_rad',-pi);
    assignin('base','t1_max_rad',pi);
    assignin('base','k0_target',0.0);
    assignin('base','k1_target',0.0);
    assignin('base','meters_end_zf',15);
    assignin('base','s_f_time',2);
    assignin('base','mean_vertical_velocity',7.9);
    assignin('base','r_min',0);
    assignin('base','r_max',1200);
    assignin('base','h_min',-550);
    assignin('base','h_max',550);
    assignin('base','yaw_min',-20);
    assignin('base','yaw_max',20);
    assignin('base','delta_a_max',0.03);
    assignin('base','max_idx',200);
    assignin('base','t_stop',Tf);
    assignin('base','Ts',Ts);
    assignin('base','Tf',Tf);
    assignin('base','rigeneration_step_s',300);
end

function blockPath = findPolicyBlock(mdl,preferred)
    if getSimulinkBlockHandle(preferred) > 0
        blockPath = char(preferred);
        return;
    end
    blocks = find_system(mdl,"LookUnderMasks","all","FollowLinks","on","Type","Block");
    idx = find(contains(string(blocks),"Policy","IgnoreCase",true),1,"first");
    if isempty(idx)
        error("Blocco Policy non trovato in %s.",mdl);
    end
    blockPath = char(blocks{idx});
end

function ensurePolicyActionToWorkspace(mdl,policyBlock,varName)
    blockName = mdl + "/AUTO_policy_action_to_workspace";

    ph = get_param(policyBlock,"PortHandles");
    if isempty(ph.Outport)
        error("Il blocco Policy non ha una porta di uscita.");
    end
    policyOut = ph.Outport(1);

    if getSimulinkBlockHandle(blockName) <= 0
        pos = get_param(policyOut,"Position");
        add_block("simulink/Sinks/To Workspace",blockName, ...
            "Position",[pos(1)+80 pos(2)-15 pos(1)+180 pos(2)+15]);
    end

    set_param(blockName,"VariableName",char(varName));
    set_param(blockName,"SaveFormat","Array");
    set_param(blockName,"MaxDataPoints","inf");
    set_param(blockName,"Decimation","1");
    set_param(blockName,"SampleTime","-1");

    phSink = get_param(blockName,"PortHandles");
    sinkLine = get_param(phSink.Inport(1),"Line");
    if sinkLine > 0
        try, delete_line(sinkLine); catch, end
    end

    try
        add_line(mdl,policyOut,phSink.Inport(1),"autorouting","on");
    catch
        % If the branch already exists, leave it unchanged.
    end

    try
        set_param(mdl,"SimulationCommand","update");
    catch
    end
end

function files = findAllAgentFilesRecursive(rootDir,resultsDir)
    listing = dir(fullfile(rootDir,"**","Agent*.mat"));
    files = struct('name',{},'folder',{},'fullpath',{},'relpath',{},'label',{},'episode',{});
    rootStr = string(rootDir);
    for i = 1:numel(listing)
        fullpath = string(fullfile(listing(i).folder,listing(i).name));
        folder = string(listing(i).folder);
        if contains(folder,string(filesep)+string(resultsDir))
            continue;
        end
        [parent,~,~] = fileparts(fullpath);
        [~,label,~] = fileparts(parent);
        ep = parseEpisode(listing(i).name);
        relpath = erase(fullpath,rootStr+string(filesep));
        files(end+1).name = string(listing(i).name); %#ok<AGROW>
        files(end).folder = folder;
        files(end).fullpath = fullpath;
        files(end).relpath = relpath;
        files(end).label = string(label);
        files(end).episode = ep;
    end
    if ~isempty(files)
        [~,idx] = sort([files.episode]);
        files = files(idx);
    end
end

function ep = parseEpisode(fileName)
    tok = regexp(fileName,'Agent(\d+)\.mat','tokens');
    if isempty(tok)
        ep = inf;
    else
        ep = str2double(tok{1}{1});
    end
end

function agent = loadAgentFromMat(path)
    S = load(path);
    if isfield(S,'saved_agent')
        agent = S.saved_agent;
    elseif isfield(S,'agent')
        agent = S.agent;
    else
        error("Il file %s non contiene saved_agent o agent.",path);
    end
end

function regenerateFixedPolicyDataFile(agent,policyDataFile,mdl)
    beforeModels = string(find_system("Type","block_diagram"));
    if exist(policyDataFile,"file")
        delete(policyDataFile);
    end

    generatePolicyBlock(agent);

    if exist(policyDataFile,"file") ~= 2
        error("generatePolicyBlock non ha creato %s.",policyDataFile);
    end

    afterModels = string(find_system("Type","block_diagram"));
    newModels = setdiff(afterModels,beforeModels);
    for k = 1:numel(newModels)
        if newModels(k) ~= string(mdl)
            try, close_system(char(newModels(k)),0); catch, end
        end
    end
end

function [X0,V0,theta0,alphaDeg,attempts] = sampleTrainingIC(X0c,V0ref,target,R,maxAngleDeg)
    attempts = 0;
    while attempts < 1000
        attempts = attempts + 1;
        d = randn(3,1); d = d/norm(d);
        rho = R*rand()^(1/3);
        X0 = X0c + rho*d;
        psi = 2*pi*rand();
        Vh = norm(V0ref(1:2));
        V0 = [Vh*cos(psi); Vh*sin(psi); V0ref(3)];
        dirXY = target(1:2)-X0(1:2);
        c = dot(V0(1:2),dirXY)/(norm(V0(1:2))*norm(dirXY));
        alphaDeg = rad2deg(acos(max(-1,min(1,c))));
        if alphaDeg <= maxAngleDeg
            theta0 = atan2(V0(2),V0(1));
            return;
        end
    end
    error("Impossibile generare una condizione iniziale valida.");
end

function assignInitialCondition(ic,target)
    assignin("base","X0",ic.X0);
    assignin("base","V0",ic.V0);
    assignin("base","w0",[0;0;0]);
    assignin("base","euler",deg2rad([0;0;0]));
    assignin("base","theta0_ph",ic.theta0_ph);
    assignin("base","target",target);
end

function clearLoggedWorkspaceVariables(actionVarName)
    vars = ["xe","x","X","pos","position","k_L","kL","action","act","logsout",string(actionVarName)];
    for i = 1:numel(vars)
        evalin("base",sprintf("if exist('%s','var'); clear('%s'); end",vars(i),vars(i)));
    end
end

function [finalPos,sourceName] = extractFinalPosition(simOut)
    names = ["xe","x","X","pos","position"];
    for i = 1:numel(names)
        try
            raw = simOut.get(char(names(i)));
            finalPos = finalPositionFromRaw(raw);
            sourceName = "simOut." + names(i);
            return;
        catch
        end
    end
    for i = 1:numel(names)
        if evalin("base",sprintf("exist('%s','var')",names(i)))
            raw = evalin("base",names(i));
            finalPos = finalPositionFromRaw(raw);
            sourceName = "base." + names(i);
            return;
        end
    end
    error("Variabile di posizione finale non trovata. Attesa xe.");
end

function p = finalPositionFromRaw(raw)
    if isa(raw,'timeseries')
        data = raw.Data;
    elseif isstruct(raw) && isfield(raw,'signals') && isfield(raw.signals,'values')
        data = raw.signals.values;
    elseif isstruct(raw) && isfield(raw,'Data')
        data = raw.Data;
    else
        data = raw;
    end
    data = squeeze(double(data));
    if isvector(data)
        v = data(:);
        if numel(v) < 3, error("xe non contiene almeno 3 componenti."); end
        p = v(end-2:end);
        return;
    end
    data = data(~all(isnan(data),2),:);
    data = data(:,~all(isnan(data),1));
    if size(data,2) >= 4 && all(diff(data(:,1)) >= -1e-12)
        p = data(find(all(isfinite(data(:,2:4)),2),1,'last'),2:4).';
    elseif size(data,2) >= 3
        p = data(find(all(isfinite(data(:,1:3)),2),1,'last'),1:3).';
    elseif size(data,1) >= 3
        data = data.';
        p = data(find(all(isfinite(data(:,1:3)),2),1,'last'),1:3).';
    else
        error("Formato xe non interpretabile.");
    end
    p = p(:);
end

function kL = extractActionValue(simOut,actionVarName)
    kL = NaN;
    names = [string(actionVarName),"k_L_policy","k_L","kL","action","act"];

    for i = 1:numel(names)
        try
            kL = lastNumeric(simOut.get(char(names(i))));
            return;
        catch
        end
    end

    for i = 1:numel(names)
        if evalin("base",sprintf("exist('%s','var')",names(i)))
            kL = lastNumeric(evalin("base",names(i)));
            return;
        end
    end
end

function v = lastNumeric(raw)
    if isa(raw,'timeseries')
        raw = raw.Data;
    elseif isstruct(raw) && isfield(raw,'signals') && isfield(raw.signals,'values')
        raw = raw.signals.values;
    elseif isstruct(raw) && isfield(raw,'Data')
        raw = raw.Data;
    end
    raw = double(squeeze(raw));
    raw = raw(isfinite(raw));
    if isempty(raw)
        v = NaN;
    else
        v = raw(end);
    end
end

function row = makeEmptyRow()
    row.agent_file = "";
    row.agent_label = "";
    row.agent_path = "";
    row.policy_data_file = "";
    row.policy_observation_dim = NaN;
    row.model_observation_dim = NaN;
    row.zero_padding_dim = NaN;
    row.observation_description = "";
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
    row.status = "not_run";
end

function writePartialResults(rows,fileName)
    if isempty(rows), return; end
    writetable(struct2table(rows),fileName);
end

function writeFatalReport(ME,fileName)
    fid = fopen(fileName,'w');
    if fid < 0
        return;
    end
    cleanupObj = onCleanup(@() fclose(fid));
    fprintf(fid,"Fatal batch error report\n");
    fprintf(fid,"Generated: %s\n\n",string(datetime("now")));
    fprintf(fid,"Identifier: %s\n",ME.identifier);
    fprintf(fid,"Message: %s\n\n",ME.message);
    fprintf(fid,"Extended report:\n%s\n",getReport(ME,'extended','hyperlinks','off'));
    clear cleanupObj;
end

function summary = buildSummaryTable(T)
    labels = unique(T.agent_label,"stable");
    S = struct([]);
    for i = 1:numel(labels)
        Ti = T(T.agent_label == labels(i),:);
        e = Ti.landing_error_radius_m;
        ep = Ti.landing_error_percent_of_initial_range;
        s.agent_label = labels(i);
        s.policy_observation_dim = Ti.policy_observation_dim(1);
        s.zero_padding_dim = Ti.zero_padding_dim(1);
        s.n_ok = height(Ti);
        s.mean_error_m = mean(e,'omitnan');
        s.median_error_m = median(e,'omitnan');
        s.p75_error_m = percentileLocal(e,75);
        s.p90_error_m = percentileLocal(e,90);
        s.p95_error_m = percentileLocal(e,95);
        s.mean_error_percent = mean(ep,'omitnan');
        s.p75_error_percent = percentileLocal(ep,75);
        s.p90_error_percent = percentileLocal(ep,90);
        s.p95_error_percent = percentileLocal(ep,95);
        s.mean_k_L = mean(Ti.k_L,'omitnan');
        S = [S; s]; %#ok<AGROW>
    end
    summary = struct2table(S);
end

function p = percentileLocal(x,q)
    x = sort(x(isfinite(x)));
    if isempty(x), p = NaN; return; end
    pos = 1 + (q/100)*(numel(x)-1);
    lo = floor(pos); hi = ceil(pos);
    if lo == hi
        p = x(lo);
    else
        p = x(lo) + (pos-lo)*(x(hi)-x(lo));
    end
end

function makeLandingErrorPlots(T,resultsDir,timestamp)
    labels = unique(T.agent_label,"stable");

    f1 = figure("Name","Landing error all cases"); hold on; grid on;
    for i = 1:numel(labels)
        Ti = T(T.agent_label == labels(i),:);
        plot(Ti.test_id,Ti.landing_error_radius_m,"o-","DisplayName",labels(i));
    end
    xlabel("Test ID"); ylabel("Errore atterraggio XY [m]");
    title("Errore di atterraggio per i 100 casi random");
    legend("Location","best","Interpreter","none");
    saveas(f1,fullfile(resultsDir,"landing_error_all_cases_"+timestamp+".png"));
    savefig(f1,fullfile(resultsDir,"landing_error_all_cases_"+timestamp+".fig"));

    f2 = figure("Name","Landing error CDF"); hold on; grid on;
    for i = 1:numel(labels)
        Ti = T(T.agent_label == labels(i),:);
        e = sort(Ti.landing_error_radius_m);
        F = (1:numel(e))/numel(e);
        plot(e,F,"LineWidth",1.5,"DisplayName",labels(i));
    end
    yline(0.75,"--","75%"); yline(0.90,"--","90%"); yline(0.95,"--","95%");
    xlabel("Errore atterraggio XY [m]"); ylabel("Frazione cumulativa");
    title("CDF errore atterraggio");
    legend("Location","best","Interpreter","none");
    saveas(f2,fullfile(resultsDir,"landing_error_cdf_"+timestamp+".png"));
    savefig(f2,fullfile(resultsDir,"landing_error_cdf_"+timestamp+".fig"));
end
