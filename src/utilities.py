""" Utilities. """


def init_params(model, scip_limits, scip_params):
    """
    :param model: scip.Model(), model instantiation
    :param scip_limits: dict, specifying SCIP parameter limits
    :param scip_params: dict, specifying SCIP parameter setting
    :return: -
        Initialize SCIP parameters for the model.
    """

    model.setIntParam('display/verblevel', 0)

    # limits
    model.setLongintParam('limits/nodes', scip_limits['node_limit'])
    model.setRealParam('limits/time', scip_limits['time_limit'])

    # enable presolve and cuts (as in default)
    model.setIntParam('presolving/maxrounds', -1)       # 0: off, -1: unlimited
    model.setIntParam('separating/maxrounds', -1)       # 0 to disable local separation
    model.setIntParam('separating/maxroundsroot', -1)   # 0 to disable root separation

    # disable reoptimization (as in default)
    model.setBoolParam('reoptimization/enable', False)

    # cutoff value is eventually set in env.run_episode
    # other parameters to be disabled in 'sandbox' setting
    model.setBoolParam('conflict/usesb', scip_params['conflict_usesb'])
    model.setBoolParam('branching/fullstrong/probingbounds', scip_params['probing_bounds'])
    model.setBoolParam('branching/relpscost/probingbounds', scip_params['probing_bounds'])
    model.setBoolParam('branching/checksol', scip_params['checksol'])
    model.setLongintParam('branching/fullstrong/reevalage', scip_params['reevalage'])

    # primal heuristics (54 total, 14 of which are disabled in default setting as well)
    if not scip_params['heuristics']:
        model.setIntParam('heuristics/actconsdiving/freq', -1)          # disabled at default
        model.setIntParam('heuristics/bound/freq', -1)                  # disabled at default
        model.setIntParam('heuristics/clique/freq', -1)
        model.setIntParam('heuristics/coefdiving/freq', -1)
        model.setIntParam('heuristics/completesol/freq', -1)
        model.setIntParam('heuristics/conflictdiving/freq', -1)         # disabled at default
        model.setIntParam('heuristics/crossover/freq', -1)
        model.setIntParam('heuristics/dins/freq', -1)                   # disabled at default
        model.setIntParam('heuristics/distributiondiving/freq', -1)
        model.setIntParam('heuristics/dualval/freq', -1)                # disabled at default
        model.setIntParam('heuristics/farkasdiving/freq', -1)
        model.setIntParam('heuristics/feaspump/freq', -1)
        model.setIntParam('heuristics/fixandinfer/freq', -1)            # disabled at default
        model.setIntParam('heuristics/fracdiving/freq', -1)
        model.setIntParam('heuristics/gins/freq', -1)
        model.setIntParam('heuristics/guideddiving/freq', -1)
        model.setIntParam('heuristics/zeroobj/freq', -1)                # disabled at default
        model.setIntParam('heuristics/indicator/freq', -1)
        model.setIntParam('heuristics/intdiving/freq', -1)              # disabled at default
        model.setIntParam('heuristics/intshifting/freq', -1)
        model.setIntParam('heuristics/linesearchdiving/freq', -1)
        model.setIntParam('heuristics/localbranching/freq', -1)         # disabled at default
        model.setIntParam('heuristics/locks/freq', -1)
        model.setIntParam('heuristics/lpface/freq', -1)
        model.setIntParam('heuristics/alns/freq', -1)
        model.setIntParam('heuristics/nlpdiving/freq', -1)
        model.setIntParam('heuristics/mutation/freq', -1)               # disabled at default
        model.setIntParam('heuristics/multistart/freq', -1)
        model.setIntParam('heuristics/mpec/freq', -1)
        model.setIntParam('heuristics/objpscostdiving/freq', -1)
        model.setIntParam('heuristics/octane/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/ofins/freq', -1)
        model.setIntParam('heuristics/oneopt/freq', -1)
        model.setIntParam('heuristics/proximity/freq', -1)              # disabled at default
        model.setIntParam('heuristics/pscostdiving/freq', -1)
        model.setIntParam('heuristics/randrounding/freq', -1)
        model.setIntParam('heuristics/rens/freq', -1)
        model.setIntParam('heuristics/reoptsols/freq', -1)
        model.setIntParam('heuristics/repair/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/rins/freq', -1)
        model.setIntParam('heuristics/rootsoldiving/freq', -1)
        model.setIntParam('heuristics/rounding/freq', -1)
        model.setIntParam('heuristics/shiftandpropagate/freq', -1)
        model.setIntParam('heuristics/shifting/freq', -1)
        model.setIntParam('heuristics/simplerounding/freq', -1)
        model.setIntParam('heuristics/subnlp/freq', -1)
        model.setIntParam('heuristics/trivial/freq', -1)
        model.setIntParam('heuristics/trivialnegation/freq', -1)
        model.setIntParam('heuristics/trysol/freq', -1)
        model.setIntParam('heuristics/twoopt/freq', -1)                 # disabled at default
        model.setIntParam('heuristics/undercover/freq', -1)
        model.setIntParam('heuristics/vbounds/freq', -1)
        model.setIntParam('heuristics/veclendiving/freq', -1)
        model.setIntParam('heuristics/zirounding/freq', -1)
