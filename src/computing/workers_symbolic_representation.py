import operator

from src.systems import FitzHughNagumo, ThreeConnectedNeurons

from src.computing.util import register
from src.cuda_sweep.sweep_symbolic import sweep

registry = {
    "worker": {},
    "init": {},
    "post": {}
}

class ConfigDataSymbolicRepresenatition:
    dt    : float
    n     : int
    initPt: list[float]
    events: list

    def __init__(self):
        self.available_systems = {
            "fitz_hugh_nagumo"       : FitzHughNagumo,
            "three_connected_neurons": ThreeConnectedNeurons
        }

        self.comparison_operators = {
            "<" : operator.lt,  "<="    : operator.le,
            "==": operator.eq,  "!="    : operator.ne,
            ">" : operator.gt,  ">="    : operator.ge,
            "is": operator.is_, "is not": operator.is_not
        }

        self.components_conversion = {
            'x1' : 0, 'y1' : 1, 'z1' : 2,
            'x2' : 3, 'y2' : 4, 'z2' : 5,
            'x3' : 6, 'y3' : 7, 'z3' : 8
        }


    def initialize(self, config):
        # Initialize system
        system_name = config['system']
        if system_name not in self.available_systems:
            raise RuntimeError(f"Unknown system: {system_name}")
        system_class = self.available_systems[system_name]
        self.system  = system_class(**config['system_params'])

        # Add grid to the system
        self.system.addGrid(config['grid'])

        print(self.system.__dict__)

        # Required system params
        evaluation_params = config['evaluation_params']
        self.dt       = evaluation_params['dt']
        self.n        = evaluation_params['n']
        self.initPt   = evaluation_params['initPt']

        # Optional system params
        if 'tSkip' not in evaluation_params:
          self.tSkip = None
        else:
          self.tSkip = evaluation_params['tSkip']

        if 'rtol' not in evaluation_params:
            self.rtol = 1e-8
        else:
            self.rtol = evaluation_params['rtol']

        if 'atol' not in evaluation_params:
            self.atol = 1e-8
        else:
            self.atol = evaluation_params['atol']

        # Initialize events
        events = config['events']
        self.events = list()
        for event in events:
            event_dict = events[event]

            compare = event_dict['compare']
            if compare not in self.comparison_operators:
                raise RuntimeError(f"Comparator must be one of these: {self.comparison_operators.keys()}")
            compare = self.comparison_operators[compare]
            component = event_dict['component']
            value = event_dict['value']
            color = event_dict['color']

            self.events.append([self.components_conversion[component], compare, value, color])
        if len(self.events) == 0:
           self.events = None


data = ConfigDataSymbolicRepresenatition()


@register(registry, 'init')
def init_symbolic_representation(config, timeStamp):
    data.initialize(config)
    return{}


@register(registry, 'worker')
def worker_symbolic_representation(config, initResult, timeStamp):
    sweep(
        data.system,
        data.dt,
        data.n,
        data.initPt,
        data.tSkip,
        data.rtol,
        data.atol,
        data.events
    )
    return{}


@register(registry, 'post')
def post_symbolic_representation(config, initResult, workerResult, grid, startTime):
    return{}
