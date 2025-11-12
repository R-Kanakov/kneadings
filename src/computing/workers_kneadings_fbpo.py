import numpy as np
import pprint
import matplotlib.pyplot as plt

from lib.computation_template.workers_utils import makeFinalOutname
from src.computing.util import register

import lib.eq_finder.SystOsscills as so
from src.system_analysis.get_inits import continue_equilibrium, get_saddle_foci_grid, find_inits_for_equilibrium_grid, generate_parameters
from src.mapping.convert import decimal_to_number_system
from src.cuda_sweep.sweep_fbpo import sweep
from src.mapping.plot_kneadings import plot_mode_map, set_random_color_map

registry = {
    "worker": {},
    "init": {},
    "post": {}
}

class ConfigDataKneadingsFbpo:
    """defaultSystem: system parameters"""
    w: int
    a: float
    b: float
    r: int

    """grid: grid parameters"""
    left_n    : int
    right_n   : int
    left_step : float
    right_step: float
    x_name    : str
    x_caption : str

    down_n   : int
    up_n     : int
    down_step: float
    up_step  : float
    y_name   : str
    y_caption: str

    """kneadings_fbpo: system evaluation parameters"""
    dt             : float
    n              : int
    stride         : int
    kneadings_start: int
    kneadings_end  : int

    """Output parameters"""
    img_extension: str

    """misc: additional keyword-arguments"""
    font_size     : int
    init_res_name : str
    param_to_index: dict


    def __init__(self):
        pass

    def initialize(self, config):
        def_sys_dict = config['defaultSystem']
        self.w       = def_sys_dict['w']
        self.a       = def_sys_dict['a']
        self.b       = def_sys_dict['b']
        self.r       = def_sys_dict['r']

        grid_dict = config['grid']

        grid_dict_first = grid_dict['first']
        self.left_n     = grid_dict_first['left_n']
        self.left_step  = grid_dict_first['left_step']
        self.right_n    = grid_dict_first['right_n']
        self.right_step = grid_dict_first['right_step']
        self.x_caption  = f"{grid_dict_first['caption']}"
        self.x_name     = grid_dict_first['name']

        grid_dict_second = grid_dict['second']
        self.up_n        = grid_dict_second['up_n']
        self.up_step     = grid_dict_second['up_step']
        self.down_n      = grid_dict_second['down_n']
        self.down_step   = grid_dict_second['down_step']
        self.y_caption   = f"{grid_dict_second['caption']}"
        self.y_name      = grid_dict_second['name']

        kneadings_dict       = config['kneadings_fbpo']
        self.dt              = kneadings_dict['dt']
        self.n               = kneadings_dict['n']
        self.stride          = kneadings_dict['stride']
        self.kneadings_start = kneadings_dict['kneadings_start']
        self.kneadings_end   = kneadings_dict['kneadings_end']

        output_dict        = config['output']
        self.img_extension = output_dict['imageExtension']
        self.directory     = {'targetDir': output_dict['directory']}

        misc_dict           = config['misc']
        self.font_size      = misc_dict['plot_params']['font_size']
        self.init_res_name  = misc_dict['init_res']
        self.start_eq       = misc_dict['start_eq']
        self.param_to_index = misc_dict['param_to_index']

data = ConfigDataKneadingsFbpo()
import time
@register(registry, 'init')
def init_kneadings_fbpo(config, timeStamp):
    data.initialize(config)

    if data.init_res_name != "ignore":  # init_flag
        init_res = np.load("./input/" + data.init_res_name + ".npz")
        inits    = init_res['inits']
        nones    = init_res['nones']
        params_x = init_res['params_x']
        params_y = init_res['params_y']

        if len(inits) != (3 * (data.left_n + data.right_n + 1) * (data.up_n + data.down_n + 1)):
            raise Exception("Saved initial conditions array differs from entered grid params!")

        if not np.array_equal(data.start_eq, init_res['start_eq']):
            raise Exception("Start equilibrium of saved initial conditions array ")
    else:
        start_sys = so.FourBiharmonicPhaseOscillators(data.w, data.a, data.b, data.r)
        reduced_rhs_wrapper = start_sys.getReducedSystem
        reduced_jac_wrapper = start_sys.getReducedSystemJac
        get_params = start_sys.getParams
        set_params = start_sys.setParams

        if data.start_eq is not None:
            eq_grid = continue_equilibrium(reduced_rhs_wrapper, reduced_jac_wrapper, get_params, set_params,
                                           data.param_to_index, 'a', 'b',
                                           data.start_eq, data.up_n, data.down_n, data.left_n, data.right_n,
                                           data.up_step, data.down_step, data.left_step, data.right_step)
            sf_grid = get_saddle_foci_grid(eq_grid, data.up_n, data.down_n, data.left_n, data.right_n)
            inits, nones = find_inits_for_equilibrium_grid(sf_grid, 3, data.up_n, data.down_n, data.left_n, data.right_n)
            params_x, params_y = generate_parameters((data.a, data.b), data.up_n, data.down_n, data.left_n, data.right_n,
                                                data.up_step, data.down_step, data.left_step, data.right_step)
        else:
            raise Exception("Start saddle-focus was not found!")

    return {'inits': inits, 'nones': nones, 'params_x': params_x, 'params_y': params_y}


@register(registry, 'worker')
def worker_kneadings_fbpo(config, initResult, timeStamp):
    inits    = initResult['inits']
    nones    = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    kneadings_records = pprint.pformat(config) + "\n\n"

    kneadings_weighted_sum_set = sweep(
        inits,
        nones,
        params_x,
        params_y,
        [data.w, data.a, data.b, data.r],
        data.param_to_index,
        data.x_name,
        data.y_name,
        data.up_n,
        data.down_n,
        data.left_n,
        data.right_n,
        data.dt,
        data.n,
        data.stride,
        data.kneadings_start,
        data.kneadings_end
    )

    # print("Results:")
    for idx in range((data.left_n + data.right_n + 1) * (data.up_n + data.down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_number_system(kneading_weighted_sum, 4)

        # print(f"a: {params_x[idx]:.9f}, "
        #       f"b: {params_y[idx]:.9f} => "
        #       f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")
        kneadings_records = (kneadings_records + f"a: {params_x[idx]:.9f}, "
                                                 f"b: {params_y[idx]:.9f} => "
                                                 f"{kneading_symbolic} (Raw: {kneading_weighted_sum})\n")

    return {'kneadings_weighted_sum_set': kneadings_weighted_sum_set, 'kneadings_records': kneadings_records}


@register(registry, 'post')
def post_kneadings_fbpo(config, initResult, workerResult, grid, startTime):
    inits    = initResult['inits']
    nones    = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    kneadings_weighted_sum_set = workerResult['kneadings_weighted_sum_set']
    kneadings_records          = workerResult['kneadings_records']

    param_x_caption = data.x_caption
    param_x_count   = data.left_n + data.right_n + 1
    param_x_start   = data.a - data.left_n * data.left_step
    param_x_end     = data.a + data.right_n * data.right_step
    param_y_caption = data.y_caption
    param_y_count   = data.up_n + data.down_n + 1
    param_y_start   = data.b - data.down_n * data.down_step
    param_y_end     = data.b + data.up_n * data.up_step

    plot_mode_map(kneadings_weighted_sum_set, set_random_color_map,
                  param_x_caption, param_y_caption,
                  param_x_start, param_x_end, param_x_count,
                  param_y_start, param_y_end, param_y_count,
                  data.font_size)
    plt.title(r'$\omega = 0$, $r = 1$', fontsize=data.font_size)

    if data.init_res_name != "ignore":
        npz_outname = makeFinalOutname(config, data.directory, "npz", startTime)
        np.savez(
            npz_outname,
            inits=inits,
            nones=nones,
            params_x=params_x,
            params_y=params_y,
            start_eq=data.start_eq
        )
        print("Init stage results successfully saved")

    npy_outname = makeFinalOutname(config, data.directory, "npy", startTime)
    np.save(npy_outname, kneadings_weighted_sum_set)
    print("Kneadings set successfully saved")

    txt_outname = makeFinalOutname(config, data.directory, "txt", startTime)
    with open(f'{txt_outname}', 'w') as txt_output:
        txt_output.write(kneadings_records)
    print("Kneadings records successfully saved")

    plot_outname = makeFinalOutname(config, data.directory, data.img_extension, startTime)
    plt.savefig(plot_outname, dpi=300, bbox_inches='tight')
    plt.close()
    print("Mode map successfully saved")