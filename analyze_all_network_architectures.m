% =========================================================
% analyze_all_network_architectures.m
%
% Analisi architetturale completa delle reti SAC salvate nella repository.
%
% Il file cerca ricorsivamente tutti gli Agent*.mat e, per ogni agente,
% estrae informazioni su:
%   - Actor
%   - Critic 1, Critic 2, ...
%   - dimensione observation/action, quando disponibile
%   - numero di layer
%   - numero di hidden layer fully-connected
%   - numero di neuroni per layer
%   - numero totale di neuroni nei layer fully-connected
%   - activation functions presenti
%   - numero di parametri learnable, quando disponibile
%
% Output prodotti in:
%   network_architecture_results/
%       network_architecture_summary_*.csv
%       network_architecture_layers_detail_*.csv
%       network_architecture_report_*.md
%       network_architecture_results_*.mat
%
% Nota:
%   Lo script usa introspezione robusta, perche' le classi MATLAB RL Toolbox
%   possono cambiare tra versioni diverse. Se un'informazione non e'
%   recuperabile, viene salvata come NaN oppure "unknown".
% =========================================================

clc;
close all;

%% CONFIG
RESULTS_DIR = "network_architecture_results";
AGENT_PATTERN = "Agent*.mat";

timestamp = string(datetime("now","Format","yyyyMMdd_HHmmss"));

if ~exist(RESULTS_DIR,"dir")
    mkdir(RESULTS_DIR);
end

summaryCsvFile = fullfile(RESULTS_DIR,"network_architecture_summary_" + timestamp + ".csv");
layersCsvFile  = fullfile(RESULTS_DIR,"network_architecture_layers_detail_" + timestamp + ".csv");
reportMdFile   = fullfile(RESULTS_DIR,"network_architecture_report_" + timestamp + ".md");
matFile        = fullfile(RESULTS_DIR,"network_architecture_results_" + timestamp + ".mat");

%% FIND AGENTS
agentFiles = findAllAgentFilesRecursive(pwd, RESULTS_DIR, AGENT_PATTERN);

if isempty(agentFiles)
    error("Nessun %s trovato nella repository.", AGENT_PATTERN);
end

fprintf("Trovati %d agenti da analizzare:\n", numel(agentFiles));
for i = 1:numel(agentFiles)
    fprintf("  %2d) %s\n", i, agentFiles(i).relpath);
end

%% ANALYSIS LOOP
summaryRows = repmat(makeEmptySummaryRow(),0,1);
layerRows   = repmat(makeEmptyLayerRow(),0,1);

for iAgent = 1:numel(agentFiles)
    agentPath  = agentFiles(iAgent).fullpath;
    agentName  = agentFiles(iAgent).name;
    agentLabel = agentFiles(iAgent).label;

    fprintf("\n[%d/%d] Analizzo agente: %s\n", iAgent, numel(agentFiles), agentPath);

    agent = loadAgentFromMat(agentPath);

    [obsDim, obsDescription] = safeDiagnoseObservation(agent);
    [actDim, actDescription] = safeDiagnoseAction(agent);

    components = getAgentNetworkComponents(agent);

    for iComp = 1:numel(components)
        comp = components(iComp);
        fprintf("  - %s\n", comp.component_name);

        [net, netSource] = extractNetworkFromRepresentation(comp.object);
        netInfo = analyzeNetwork(net);

        s = makeEmptySummaryRow();
        s.agent_file = string(agentName);
        s.agent_label = string(agentLabel);
        s.agent_path = string(agentPath);
        s.agent_episode = agentFiles(iAgent).episode;
        s.component_name = string(comp.component_name);
        s.component_class = string(class(comp.object));
        s.network_class = string(class(net));
        s.network_source = string(netSource);
        s.observation_dim = obsDim;
        s.observation_description = string(obsDescription);
        s.action_dim = actDim;
        s.action_description = string(actDescription);
        s.total_layers = netInfo.total_layers;
        s.input_layers = netInfo.input_layers;
        s.fully_connected_layers = netInfo.fully_connected_layers;
        s.hidden_fully_connected_layers = netInfo.hidden_fully_connected_layers;
        s.output_fully_connected_layers = netInfo.output_fully_connected_layers;
        s.hidden_neurons_by_layer = string(netInfo.hidden_neurons_by_layer);
        s.all_fc_neurons_by_layer = string(netInfo.all_fc_neurons_by_layer);
        s.total_fc_neurons = netInfo.total_fc_neurons;
        s.total_hidden_fc_neurons = netInfo.total_hidden_fc_neurons;
        s.activation_functions = string(netInfo.activation_functions);
        s.learnable_parameters = netInfo.learnable_parameters;
        s.layer_names = string(netInfo.layer_names);
        s.status = string(netInfo.status);
        s.error_message = string(netInfo.error_message);

        summaryRows(end+1,1) = s; %#ok<SAGROW>

        for j = 1:numel(netInfo.layer_details)
            d = netInfo.layer_details(j);
            r = makeEmptyLayerRow();
            r.agent_file = string(agentName);
            r.agent_label = string(agentLabel);
            r.agent_path = string(agentPath);
            r.agent_episode = agentFiles(iAgent).episode;
            r.component_name = string(comp.component_name);
            r.component_class = string(class(comp.object));
            r.network_class = string(class(net));
            r.layer_index = d.layer_index;
            r.layer_name = string(d.layer_name);
            r.layer_class = string(d.layer_class);
            r.layer_type = string(d.layer_type);
            r.output_size = d.output_size;
            r.num_neurons = d.num_neurons;
            r.is_input_layer = d.is_input_layer;
            r.is_fully_connected = d.is_fully_connected;
            r.is_activation = d.is_activation;
            r.is_hidden_fc = d.is_hidden_fc;
            r.is_output_fc = d.is_output_fc;
            r.activation_function = string(d.activation_function);
            r.learnable_parameters_layer = d.learnable_parameters_layer;
            layerRows(end+1,1) = r; %#ok<SAGROW>
        end
    end
end

summaryTable = struct2table(summaryRows);
layersTable  = struct2table(layerRows);

writetable(summaryTable, summaryCsvFile);
writetable(layersTable, layersCsvFile);
save(matFile, "summaryTable", "layersTable", "agentFiles");
writeMarkdownReport(reportMdFile, summaryTable, layersTable);

fprintf("\n=========================================================\n");
fprintf("Analisi architetture completata.\n");
fprintf("Summary CSV: %s\n", summaryCsvFile);
fprintf("Layers CSV:  %s\n", layersCsvFile);
fprintf("Report MD:   %s\n", reportMdFile);
fprintf("MAT:         %s\n", matFile);
fprintf("=========================================================\n");

disp(summaryTable);

%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================
function files = findAllAgentFilesRecursive(rootDir, resultsDir, pattern)
    listing = dir(fullfile(rootDir,"**",pattern));

    % Fallback per versioni MATLAB vecchie senza supporto ** in dir().
    if isempty(listing)
        listing = recursiveDirPattern(rootDir, pattern);
    end

    files = struct('name',{},'folder',{},'fullpath',{},'relpath',{},'label',{},'episode',{});
    rootStr = string(rootDir);

    for i = 1:numel(listing)
        fullpath = string(fullfile(listing(i).folder, listing(i).name));
        folder = string(listing(i).folder);

        if contains(folder, string(filesep) + string(resultsDir))
            continue;
        end

        [parent,~,~] = fileparts(fullpath);
        [~,label,~] = fileparts(parent);

        files(end+1).name = string(listing(i).name); %#ok<AGROW>
        files(end).folder = folder;
        files(end).fullpath = fullpath;
        files(end).relpath = erase(fullpath, rootStr + string(filesep));
        files(end).label = string(label);
        files(end).episode = parseEpisodeNumber(listing(i).name);
    end

    if ~isempty(files)
        [~,idx] = sortrows([[files.episode]' (1:numel(files))']);
        files = files(idx);
    end
end

function listing = recursiveDirPattern(rootDir, pattern)
    listing = [];
    content = dir(rootDir);
    for i = 1:numel(content)
        name = content(i).name;
        if content(i).isdir
            if strcmp(name,".") || strcmp(name,"..") || strcmp(name,".git")
                continue;
            end
            sub = recursiveDirPattern(fullfile(content(i).folder,name), pattern);
            listing = [listing; sub]; %#ok<AGROW>
        else
            if ~isempty(regexp(name, wildcardToRegexp(pattern), 'once'))
                listing = [listing; content(i)]; %#ok<AGROW>
            end
        end
    end
end

function expr = wildcardToRegexp(pattern)
    expr = regexptranslate('wildcard', pattern);
end

function ep = parseEpisodeNumber(fileName)
    tok = regexp(fileName,'Agent(\d+)\.mat','tokens');
    if ~isempty(tok)
        ep = str2double(tok{1}{1});
        return;
    end
    tok = regexp(fileName,'AgentFINAL_ep(\d+)_','tokens');
    if ~isempty(tok)
        ep = str2double(tok{1}{1});
        return;
    end
    ep = inf;
end

function agent = loadAgentFromMat(path)
    S = load(path);
    if isfield(S,'saved_agent')
        agent = S.saved_agent;
    elseif isfield(S,'agent')
        agent = S.agent;
    else
        vars = fieldnames(S);
        error("Il file %s non contiene saved_agent o agent. Variabili presenti: %s", ...
            path, strjoin(vars, ", "));
    end
end

function [obsDim, description] = safeDiagnoseObservation(agent)
    try
        [obsDim, description] = diagnose_policy_observation_dimension(agent);
    catch ME
        obsDim = NaN;
        description = "unknown: " + string(ME.message);
    end
end

function [actDim, description] = safeDiagnoseAction(agent)
    try
        actInfo = getActionInfo(agent);
        [actDim, description] = numericSpecDescription(actInfo);
    catch
        try
            actor = getActor(agent);
            actInfo = getActionInfo(actor);
            [actDim, description] = numericSpecDescription(actInfo);
        catch ME
            actDim = NaN;
            description = "unknown: " + string(ME.message);
        end
    end
end

function [dim, description] = numericSpecDescription(info)
    if ~iscell(info)
        info = {info};
    end
    dims = zeros(numel(info),1);
    names = strings(numel(info),1);
    for i = 1:numel(info)
        try
            dims(i) = prod(info{i}.Dimension);
        catch
            dims(i) = NaN;
        end
        try
            names(i) = string(info{i}.Name);
        catch
            names(i) = "spec" + i;
        end
    end
    dim = sum(dims,'omitnan');
    parts = strings(numel(info),1);
    for i = 1:numel(info)
        parts(i) = names(i) + ":" + string(dims(i));
    end
    description = strjoin(parts, ", ");
end

function components = getAgentNetworkComponents(agent)
    components = struct('component_name',{},'object',{});

    try
        actor = getActor(agent);
        components(end+1).component_name = "Actor"; %#ok<AGROW>
        components(end).object = actor;
    catch ME
        warning("Impossibile estrarre Actor: %s", ME.message);
    end

    critics = [];
    try
        critics = getCritic(agent);
    catch
        try
            critics = agent.Critic;
        catch
            critics = [];
        end
    end

    if ~isempty(critics)
        for i = 1:numel(critics)
            components(end+1).component_name = "Critic_" + string(i); %#ok<AGROW>
            components(end).object = critics(i);
        end
    end
end

function [net, source] = extractNetworkFromRepresentation(rep)
    source = "unknown";
    net = [];

    try
        net = getModel(rep);
        source = "getModel";
        return;
    catch
    end

    candidateProps = ["Model", "Network", "Net", "dlnetwork", "LayerGraph"];
    for i = 1:numel(candidateProps)
        p = candidateProps(i);
        try
            net = rep.(p);
            source = "." + p;
            return;
        catch
        end
    end

    % Alcuni oggetti contengono una struttura Model.Network.
    try
        m = rep.Model;
        if isstruct(m) && isfield(m,'Network')
            net = m.Network;
            source = ".Model.Network";
            return;
        end
    catch
    end

    error("Non riesco a estrarre la rete da un oggetto di classe %s.", class(rep));
end

function info = analyzeNetwork(net)
    info = struct();
    info.status = "ok";
    info.error_message = "";
    info.total_layers = NaN;
    info.input_layers = NaN;
    info.fully_connected_layers = NaN;
    info.hidden_fully_connected_layers = NaN;
    info.output_fully_connected_layers = NaN;
    info.hidden_neurons_by_layer = "";
    info.all_fc_neurons_by_layer = "";
    info.total_fc_neurons = NaN;
    info.total_hidden_fc_neurons = NaN;
    info.activation_functions = "";
    info.learnable_parameters = NaN;
    info.layer_names = "";
    info.layer_details = repmat(makeEmptyLayerDetail(),0,1);

    try
        layers = getLayersFromNetwork(net);
        info.total_layers = numel(layers);

        learnableByLayer = countLearnablesByLayer(net);

        details = repmat(makeEmptyLayerDetail(),0,1);
        for i = 1:numel(layers)
            L = layers(i);
            d = makeEmptyLayerDetail();
            d.layer_index = i;
            d.layer_name = getLayerName(L,i);
            d.layer_class = string(class(L));
            d.layer_type = classifyLayerType(L);
            d.output_size = getLayerOutputSize(L);
            d.num_neurons = getLayerNeuronCount(L);
            d.is_input_layer = isInputLayer(L);
            d.is_fully_connected = isFullyConnectedLayer(L);
            d.is_activation = isActivationLayer(L);
            d.activation_function = getActivationFunctionName(L);
            d.is_output_fc = d.is_fully_connected && isOutputLikeLayer(L,i,numel(layers));
            d.is_hidden_fc = d.is_fully_connected && ~d.is_output_fc;
            d.learnable_parameters_layer = getLearnablesForLayer(learnableByLayer, d.layer_name);
            details(end+1,1) = d; %#ok<AGROW>
        end

        info.layer_details = details;
        info.input_layers = sum([details.is_input_layer]);
        info.fully_connected_layers = sum([details.is_fully_connected]);
        info.hidden_fully_connected_layers = sum([details.is_hidden_fc]);
        info.output_fully_connected_layers = sum([details.is_output_fc]);

        fc = details([details.is_fully_connected]);
        hiddenFc = details([details.is_hidden_fc]);
        acts = details([details.is_activation]);

        info.all_fc_neurons_by_layer = joinLayerNeuronString(fc);
        info.hidden_neurons_by_layer = joinLayerNeuronString(hiddenFc);
        info.total_fc_neurons = sum([fc.num_neurons], 'omitnan');
        info.total_hidden_fc_neurons = sum([hiddenFc.num_neurons], 'omitnan');
        info.activation_functions = joinUniqueStrings(string({acts.activation_function}));
        info.layer_names = strjoin(string({details.layer_name}), " -> ");
        info.learnable_parameters = countTotalLearnables(net, details);
    catch ME
        info.status = "error";
        info.error_message = string(ME.message);
    end
end

function layers = getLayersFromNetwork(net)
    if isempty(net)
        error("Rete vuota.");
    end

    try
        layers = net.Layers;
        return;
    catch
    end

    try
        layers = net.LayerGraph.Layers;
        return;
    catch
    end

    if isa(net,'nnet.cnn.Layer') || contains(class(net),'Layer')
        layers = net;
        return;
    end

    error("Non riesco a leggere la proprieta' Layers dalla rete di classe %s.", class(net));
end

function name = getLayerName(layer,idx)
    try
        name = string(layer.Name);
        if strlength(name) == 0
            name = "layer_" + idx;
        end
    catch
        name = "layer_" + idx;
    end
end

function typ = classifyLayerType(layer)
    c = lower(string(class(layer)));
    n = lower(getLayerName(layer,0));

    if contains(c,"featureinput") || contains(c,"imageinput") || contains(c,"sequenceinput")
        typ = "input";
    elseif contains(c,"fullyconnected")
        typ = "fully_connected";
    elseif contains(c,"relulayer") || contains(c,"relu")
        typ = "relu";
    elseif contains(c,"softplus")
        typ = "softplus";
    elseif contains(c,"tanh")
        typ = "tanh";
    elseif contains(c,"sigmoid")
        typ = "sigmoid";
    elseif contains(c,"leakyrelu")
        typ = "leaky_relu";
    elseif contains(c,"elu")
        typ = "elu";
    elseif contains(c,"batchnormalization")
        typ = "batch_normalization";
    elseif contains(c,"dropout")
        typ = "dropout";
    elseif contains(c,"concatenation") || contains(c,"concat") || contains(n,"concat")
        typ = "concatenation";
    elseif contains(c,"addition")
        typ = "addition";
    else
        typ = string(class(layer));
    end
end

function out = getLayerOutputSize(layer)
    props = ["OutputSize", "NumHiddenUnits", "InputSize"];
    out = NaN;
    for i = 1:numel(props)
        try
            val = layer.(props(i));
            if isnumeric(val)
                out = double(prod(val));
                return;
            end
        catch
        end
    end
end

function n = getLayerNeuronCount(layer)
    if isFullyConnectedLayer(layer)
        try
            n = double(layer.OutputSize);
            return;
        catch
            n = NaN;
            return;
        end
    end

    if isInputLayer(layer)
        try
            n = double(prod(layer.InputSize));
            return;
        catch
            n = NaN;
            return;
        end
    end

    n = NaN;
end

function tf = isInputLayer(layer)
    c = lower(string(class(layer)));
    tf = contains(c,"input");
end

function tf = isFullyConnectedLayer(layer)
    c = lower(string(class(layer)));
    tf = contains(c,"fullyconnected");
end

function tf = isActivationLayer(layer)
    c = lower(string(class(layer)));
    tf = contains(c,"relu") || contains(c,"softplus") || contains(c,"tanh") || ...
         contains(c,"sigmoid") || contains(c,"elu");
end

function a = getActivationFunctionName(layer)
    if ~isActivationLayer(layer)
        a = "";
        return;
    end
    typ = classifyLayerType(layer);
    a = typ;
end

function tf = isOutputLikeLayer(layer,idx,nLayers)
    name = lower(getLayerName(layer,idx));
    n = NaN;
    try
        n = double(layer.OutputSize);
    catch
    end

    outputKeywords = ["output", "actionmean", "actionstd", "qval", "qvalue", "q_val", "q", "mean", "std"];
    tf = false;
    for k = 1:numel(outputKeywords)
        if contains(name, outputKeywords(k))
            tf = true;
            return;
        end
    end

    % Euristica: FC molto piccolo alla fine della rete o vicino alla fine.
    if idx >= nLayers-1 && isfinite(n) && n <= 10
        tf = true;
        return;
    end
end

function txt = joinLayerNeuronString(details)
    if isempty(details)
        txt = "";
        return;
    end
    parts = strings(numel(details),1);
    for i = 1:numel(details)
        parts(i) = string(details(i).layer_name) + ":" + string(details(i).num_neurons);
    end
    txt = strjoin(parts, "; ");
end

function txt = joinUniqueStrings(vals)
    vals = vals(strlength(vals) > 0);
    vals = unique(vals,'stable');
    if isempty(vals)
        txt = "";
    else
        txt = strjoin(vals, "; ");
    end
end

function total = countTotalLearnables(net, details)
    total = NaN;
    try
        L = net.Learnables;
        total = 0;
        for i = 1:height(L)
            total = total + numel(L.Value{i});
        end
        return;
    catch
    end

    vals = [details.learnable_parameters_layer];
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        total = sum(vals);
    end
end

function tableMap = countLearnablesByLayer(net)
    tableMap = containers.Map('KeyType','char','ValueType','double');
    try
        L = net.Learnables;
        for i = 1:height(L)
            layerName = char(string(L.Layer(i)));
            n = numel(L.Value{i});
            if isKey(tableMap, layerName)
                tableMap(layerName) = tableMap(layerName) + n;
            else
                tableMap(layerName) = n;
            end
        end
    catch
        % Se Learnables non esiste, rimane vuota.
    end
end

function n = getLearnablesForLayer(tableMap, layerName)
    n = NaN;
    try
        key = char(string(layerName));
        if isKey(tableMap,key)
            n = tableMap(key);
        end
    catch
    end
end

function d = makeEmptyLayerDetail()
    d.layer_index = NaN;
    d.layer_name = "";
    d.layer_class = "";
    d.layer_type = "";
    d.output_size = NaN;
    d.num_neurons = NaN;
    d.is_input_layer = false;
    d.is_fully_connected = false;
    d.is_activation = false;
    d.is_hidden_fc = false;
    d.is_output_fc = false;
    d.activation_function = "";
    d.learnable_parameters_layer = NaN;
end

function row = makeEmptySummaryRow()
    row.agent_file = "";
    row.agent_label = "";
    row.agent_path = "";
    row.agent_episode = NaN;
    row.component_name = "";
    row.component_class = "";
    row.network_class = "";
    row.network_source = "";
    row.observation_dim = NaN;
    row.observation_description = "";
    row.action_dim = NaN;
    row.action_description = "";
    row.total_layers = NaN;
    row.input_layers = NaN;
    row.fully_connected_layers = NaN;
    row.hidden_fully_connected_layers = NaN;
    row.output_fully_connected_layers = NaN;
    row.hidden_neurons_by_layer = "";
    row.all_fc_neurons_by_layer = "";
    row.total_fc_neurons = NaN;
    row.total_hidden_fc_neurons = NaN;
    row.activation_functions = "";
    row.learnable_parameters = NaN;
    row.layer_names = "";
    row.status = "";
    row.error_message = "";
end

function row = makeEmptyLayerRow()
    row.agent_file = "";
    row.agent_label = "";
    row.agent_path = "";
    row.agent_episode = NaN;
    row.component_name = "";
    row.component_class = "";
    row.network_class = "";
    row.layer_index = NaN;
    row.layer_name = "";
    row.layer_class = "";
    row.layer_type = "";
    row.output_size = NaN;
    row.num_neurons = NaN;
    row.is_input_layer = false;
    row.is_fully_connected = false;
    row.is_activation = false;
    row.is_hidden_fc = false;
    row.is_output_fc = false;
    row.activation_function = "";
    row.learnable_parameters_layer = NaN;
end

function writeMarkdownReport(fileName, summaryTable, layersTable)
    fid = fopen(fileName,'w');
    if fid < 0
        warning("Impossibile creare report markdown: %s", fileName);
        return;
    end
    cleanupObj = onCleanup(@() fclose(fid));

    fprintf(fid,"# Report architetture reti SAC\n\n");
    fprintf(fid,"Generato: %s\n\n", string(datetime("now")));
    fprintf(fid,"## Sintesi componenti\n\n");

    for i = 1:height(summaryTable)
        s = summaryTable(i,:);
        fprintf(fid,"### %s — %s\n\n", s.agent_label, s.component_name);
        fprintf(fid,"- File agente: `%s`\n", s.agent_file);
        fprintf(fid,"- Classe componente: `%s`\n", s.component_class);
        fprintf(fid,"- Classe rete: `%s`\n", s.network_class);
        fprintf(fid,"- Observation dim: `%g` — %s\n", s.observation_dim, s.observation_description);
        fprintf(fid,"- Action dim: `%g` — %s\n", s.action_dim, s.action_description);
        fprintf(fid,"- Numero totale layer: `%g`\n", s.total_layers);
        fprintf(fid,"- Fully-connected layer: `%g`\n", s.fully_connected_layers);
        fprintf(fid,"- Hidden fully-connected layer: `%g`\n", s.hidden_fully_connected_layers);
        fprintf(fid,"- Neuroni hidden per layer: `%s`\n", s.hidden_neurons_by_layer);
        fprintf(fid,"- Neuroni FC totali: `%g`\n", s.total_fc_neurons);
        fprintf(fid,"- Neuroni hidden FC totali: `%g`\n", s.total_hidden_fc_neurons);
        fprintf(fid,"- Activation functions: `%s`\n", s.activation_functions);
        fprintf(fid,"- Parametri learnable: `%g`\n\n", s.learnable_parameters);

        idx = layersTable.agent_path == s.agent_path & layersTable.component_name == s.component_name;
        L = layersTable(idx,:);
        fprintf(fid,"| # | Layer | Tipo | Neuroni | Activation | Parametri |\n");
        fprintf(fid,"|---:|---|---|---:|---|---:|\n");
        for j = 1:height(L)
            fprintf(fid,"| %g | `%s` | `%s` | %g | `%s` | %g |\n", ...
                L.layer_index(j), L.layer_name(j), L.layer_type(j), L.num_neurons(j), ...
                L.activation_function(j), L.learnable_parameters_layer(j));
        end
        fprintf(fid,"\n");
    end

    clear cleanupObj;
end
