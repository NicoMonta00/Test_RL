function [nObs, description] = diagnose_policy_observation_dimension(agent)
%DIAGNOSE_POLICY_OBSERVATION_DIMENSION Return the observation-vector size used by a trained RL agent.
%
%   [nObs, description] = diagnose_policy_observation_dimension(agent)
%
% The function first tries getObservationInfo(agent). If the MATLAB release
% does not support that call, it falls back to getObservationInfo(getActor(agent)).

    obsInfo = [];
    source = "agent";

    try
        obsInfo = getObservationInfo(agent);
    catch
        try
            actor = getActor(agent);
            obsInfo = getObservationInfo(actor);
            source = "actor";
        catch ME
            error("Unable to diagnose observation dimension: %s", ME.message);
        end
    end

    if ~iscell(obsInfo)
        obsInfo = {obsInfo};
    end

    dims = zeros(numel(obsInfo), 1);
    names = strings(numel(obsInfo), 1);

    for i = 1:numel(obsInfo)
        dims(i) = prod(obsInfo{i}.Dimension);
        try
            names(i) = string(obsInfo{i}.Name);
        catch
            names(i) = "obs" + i;
        end
    end

    nObs = sum(dims);

    parts = strings(numel(dims), 1);
    for i = 1:numel(dims)
        parts(i) = names(i) + ":" + string(dims(i));
    end

    description = source + " | " + strjoin(parts, ", ");
end
