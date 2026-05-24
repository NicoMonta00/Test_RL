function nPad = configure_policy_observation_padding(mdl, policyBlock, policyObsDim, modelObsDim)
%CONFIGURE_POLICY_OBSERVATION_PADDING Append zero components before a Policy block.
%
%   nPad = configure_policy_observation_padding(mdl, policyBlock, policyObsDim, modelObsDim)
%
% If policyObsDim > modelObsDim, the function creates/updates two blocks:
%   AUTO_obs_pad_zeros   : Constant block with zeros(nPad,1)
%   AUTO_obs_pad_concat  : Vector Concatenate block
%
% The original observation line is routed through the concatenate block:
%   obs_policy = [obs_model; zeros(nPad,1)]
%
% If policyObsDim == modelObsDim, the function removes the automatic adapter
% if it exists and restores the original direct connection.

    padVar = "obs_policy_extra_zeros";
    nPad = policyObsDim - modelObsDim;

    if nPad < 0
        error("Policy expects %d observations, but the model provides %d.", policyObsDim, modelObsDim);
    end

    assignin("base", padVar, zeros(max(nPad,0),1));
    assignin("base", "policy_expected_observation_dim", policyObsDim);

    concatBlock = mdl + "/AUTO_obs_pad_concat";
    constBlock  = mdl + "/AUTO_obs_pad_zeros";

    if nPad == 0
        restore_direct_policy_connection(mdl, policyBlock, concatBlock, constBlock);
        return;
    end

    if getSimulinkBlockHandle(concatBlock) > 0 && getSimulinkBlockHandle(constBlock) > 0
        set_param(constBlock, "Value", char(padVar));
        try
            set_param(mdl, "SimulationCommand", "update");
        catch
        end
        return;
    end

    phPolicy = get_param(policyBlock, "PortHandles");
    policyIn = phPolicy.Inport(1);
    oldLine = get_param(policyIn, "Line");

    if oldLine < 0
        error("The Policy block observation input is not connected.");
    end

    srcPort = get_param(oldLine, "SrcPortHandle");
    srcPos = get_param(srcPort, "Position");
    dstPos = get_param(policyIn, "Position");

    delete_line(oldLine);

    xMid = round((srcPos(1) + dstPos(1))/2);
    yMid = round((srcPos(2) + dstPos(2))/2);

    try
        add_block("simulink/Signal Routing/Vector Concatenate", concatBlock, ...
            "Position", [xMid-20 yMid-25 xMid+30 yMid+25]);
    catch
        add_block("simulink/Math Operations/Concatenate", concatBlock, ...
            "Position", [xMid-20 yMid-25 xMid+30 yMid+25]);
    end

    try_set_param(concatBlock, "NumInputs", "2");
    try_set_param(concatBlock, "Mode", "Vector");
    try_set_param(concatBlock, "ConcatenateDimension", "1");

    add_block("simulink/Sources/Constant", constBlock, ...
        "Value", char(padVar), ...
        "Position", [xMid-115 yMid+45 xMid-40 yMid+75]);
    try_set_param(constBlock, "VectorParams1D", "on");

    phConcat = get_param(concatBlock, "PortHandles");
    phConst  = get_param(constBlock, "PortHandles");

    add_line(mdl, srcPort, phConcat.Inport(1), "autorouting", "on");
    add_line(mdl, phConst.Outport(1), phConcat.Inport(2), "autorouting", "on");
    add_line(mdl, phConcat.Outport(1), policyIn, "autorouting", "on");

    try
        set_param(mdl, "SimulationCommand", "update");
    catch
    end
end

function restore_direct_policy_connection(mdl, policyBlock, concatBlock, constBlock)
    if getSimulinkBlockHandle(concatBlock) <= 0
        return;
    end

    phPolicy = get_param(policyBlock, "PortHandles");
    policyIn = phPolicy.Inport(1);
    phConcat = get_param(concatBlock, "PortHandles");

    inLine = get_param(phConcat.Inport(1), "Line");
    outLine = get_param(phConcat.Outport(1), "Line");

    srcPort = -1;
    if inLine > 0
        srcPort = get_param(inLine, "SrcPortHandle");
    end

    if outLine > 0
        delete_line(outLine);
    end

    for i = 1:numel(phConcat.Inport)
        ln = get_param(phConcat.Inport(i), "Line");
        if ln > 0
            delete_line(ln);
        end
    end

    if getSimulinkBlockHandle(concatBlock) > 0
        delete_block(concatBlock);
    end
    if getSimulinkBlockHandle(constBlock) > 0
        delete_block(constBlock);
    end

    if srcPort > 0
        add_line(mdl, srcPort, policyIn, "autorouting", "on");
    end
end

function try_set_param(blockPath, paramName, paramValue)
    try
        set_param(blockPath, paramName, paramValue);
    catch
    end
end
