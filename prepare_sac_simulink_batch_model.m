function prepare_sac_simulink_batch_model(mdl)
%PREPARE_SAC_SIMULINK_BATCH_MODEL Prepare SAC_simulink for automated batch simulation.
%
% This helper solves two practical issues during scripted Policy-block tests:
%   1. Open mask/block dialogs with unapplied changes can block sim().
%   2. generatePolicyBlock can create temporary untitled models that are not used.
%
% The function keeps the target model open, applies/sets the To Workspace
% logging block used for the landing position, and closes temporary untitled
% block diagrams.

    if nargin < 1 || strlength(string(mdl)) == 0
        mdl = "SAC_simulink";
    end
    mdl = char(mdl);

    close_open_simulink_dialogs_safely();

    if ~bdIsLoaded(mdl)
        open_system(mdl);
    end

    configure_xe_to_workspace_block(mdl);
    close_extra_untitled_models(mdl);

    try
        set_param(mdl, "SimulationCommand", "update");
    catch
        % Do not fail here: sim() will report the real model error if present.
    end
end

function close_open_simulink_dialogs_safely()
    try
        root = DAStudio.ToolRoot;
        dialogs = root.getOpenDialogs;
        for k = 1:numel(dialogs)
            try
                dialogs(k).apply;
            catch
            end
            try
                dialogs(k).delete;
            catch
            end
        end
    catch
        % DAStudio dialog API can differ across releases; ignore if unavailable.
    end
end

function configure_xe_to_workspace_block(mdl)
    preferredBlock = [mdl '/GNC & Environment/Environment/To Workspace'];

    if getSimulinkBlockHandle(preferredBlock) > 0
        blocks = {preferredBlock};
    else
        blocks = find_system(mdl, ...
            "LookUnderMasks", "all", ...
            "FollowLinks", "on", ...
            "BlockType", "ToWorkspace");
    end

    for i = 1:numel(blocks)
        blk = blocks{i};
        try
            currentName = string(get_param(blk, "VariableName"));
        catch
            currentName = "";
        end

        % The landing-position block is expected to write xe. If the preferred
        % path exists, force it. If we are scanning all To Workspace blocks,
        % only touch blocks already named xe or x-like position variables.
        shouldConfigure = strcmp(blk, preferredBlock) || any(currentName == ["xe", "x", "X", "pos", "position"]);

        if shouldConfigure
            try_set_param(blk, "VariableName", "xe");
            try_set_param(blk, "SaveFormat", "Array");
            try_set_param(blk, "MaxDataPoints", "inf");
            try_set_param(blk, "Decimation", "1");
            try_set_param(blk, "SampleTime", "-1");
        end
    end
end

function close_extra_untitled_models(mdl)
    try
        diagrams = string(find_system("Type", "block_diagram"));
    catch
        return;
    end

    for i = 1:numel(diagrams)
        name = diagrams(i);
        if name == string(mdl)
            continue;
        end

        if startsWith(name, "untitled", "IgnoreCase", true)
            try
                close_system(char(name), 0);
            catch
            end
        end
    end
end

function try_set_param(blockPath, paramName, paramValue)
    try
        set_param(blockPath, paramName, paramValue);
    catch
    end
end
