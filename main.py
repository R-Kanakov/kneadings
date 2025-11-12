import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/computation_template'))

import src.computing.workers_kneadings_fbpo as kneadings_wrk
import src.computing.workers_symbolic_representation as symbolic_wrk
import src.computing.engines as engines
from lib.computation_template.engine import workflow, getConfiguration, parseArguments

ENGINE_REGISTRY = {
    'kneadings_fbpo'         : engines.general_engine,
    'symbolic_representation': engines.general_engine
}

WORKER_REGISTRY = {
    'kneadings_fbpo'         : kneadings_wrk,
    'symbolic_representation': symbolic_wrk
}

if __name__ == "__main__":
    # Parsing command line argumens, loading configuration yaml file
    parseArguments(sys.argv)
    configDict = getConfiguration(sys.argv[1])
    taskName   = configDict['task']

    # Creating functions for initialization, computing and drawing
    wrk = WORKER_REGISTRY[taskName]
    initFunc    = wrk.registry['init']
    worker      = wrk.registry['worker']
    postProcess = wrk.registry['post']

    engine = ENGINE_REGISTRY[taskName]

    def gridMaker(configDict): pass
    workflow(configDict, initFunc, gridMaker, worker, engine, postProcess)

