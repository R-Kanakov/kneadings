from math import prod
import numpy as np
from numba import cuda
import importlib

# This is the only way i found to create `cuda.local.array` with "dynamic" size
# (actually it's not really dynamic, it's just defined before @cuda.jit is run)
system_dim = None
module     = None

@cuda.jit
def stepper_rk4(params, y_curr, dt, k1):
    """Makes RK-4 step and saves the value in y_curr"""
    k2     = cuda.local.array(system_dim, dtype=np.float64)
    k3     = cuda.local.array(system_dim, dtype=np.float64)
    k4     = cuda.local.array(system_dim, dtype=np.float64)
    y_temp = cuda.local.array(system_dim, dtype=np.float64)

    module.rhs_jit(y_curr, params, k1)

    for i in range(system_dim):
        y_temp[i] = y_curr[i] + k1[i] * dt / 2.0
    module.rhs_jit(y_temp, params, k2)

    for i in range(system_dim):
        y_temp[i] = y_curr[i] + k2[i] * dt / 2.0
    module.rhs_jit(y_temp, params, k3)

    for i in range(system_dim):
        y_temp[i] = y_curr[i] + k3[i] * dt
    module.rhs_jit(y_temp, params, k4)

    for i in range(system_dim):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0


events_len    = None
events_in_int = None

@cuda.jit
def integrator_rk4(y_curr, params, dt, n, tSkip, events, events_flags):
    """Computes events happend for trajectory"""
    y_tmp = cuda.local.array(system_dim, dtype=np.float64)
    if tSkip is not None:
        for i in range(system_dim):
            y_tmp[i] = y_curr[i]

        for i in range(tSkip):
            _ = cuda.local.array(system_dim, dtype=np.float64)
            stepper_rk4(params, y_tmp, dt, _)

        for i in range(system_dim):
            y_curr[i] = y_tmp[i]

    dX_prev = cuda.local.array(system_dim, dtype=np.float64)
    for i in range(system_dim):
        dX_prev[i] = -float('inf')

    for i in range(1, n):
        dX_curr = cuda.local.array(system_dim, dtype=np.float64)
        stepper_rk4(params, y_curr, dt, dX_curr)

        for j in range(events_len):
            component = events[j]

            if dX_prev[component] > 0.0 and dX_curr[component] <= 0.0:
                curr = i * events_len + j
                ind  = curr // events_in_int
                mod  = curr % events_in_int
                events_flags[ind] |= 1 << mod

        for j in range(system_dim):
            dX_prev[j] = dX_curr[j]


dots_with_events_per_curve = None
len_change_params          = None
len_system_params          = None
total_events_len           = None

@cuda.jit
def sweep_threads(
    symbolic_hash_set_gpu,
    total_parameter_space_size,
    init,
    system_parameters,
    parameters_places,
    parameters_to_change,
    grid_params,
    grid_lengths,
    dt,
    n,
    tSkip,
    events
):
    """CUDA kernel"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    cuda.syncthreads()

    if idx < total_events_len:
        multi_idx = cuda.local.array(len_change_params, dtype=np.int64)
        idx_tmp = idx
        for i in range(len(grid_lengths)):
            multi_idx[i] = idx_tmp % grid_lengths[i]
            idx_tmp //= grid_lengths[i]

        system_parameters_tmp = cuda.local.array(len_system_params, dtype=np.float64)
        for i in range(len(system_parameters)):
            system_parameters_tmp[i] = system_parameters[i]

        for i in range(len(parameters_to_change)):
            # TODO: make this work not for only parameters with len == 1
            system_parameters_tmp[parameters_places[parameters_to_change[i]][0]] = grid_params[i, multi_idx[i]]

        events_flags = cuda.local.array(dots_with_events_per_curve, dtype=np.uint64)
        for i in range(dots_with_events_per_curve):
            events_flags[i] = 0

        integrator_rk4(init, system_parameters_tmp, dt, n, tSkip, events, events_flags)

        current_array_idx = idx * dots_with_events_per_curve
        for i in range(dots_with_events_per_curve):
            symbolic_hash_set_gpu[current_array_idx + i] = events_flags[i]


THREADS_PER_BLOCK = 20

def sweep(
    system,
    dt,
    n,
    initPt,
    tSkip,
    rtol,
    atol,
    events
):
    """Calls CUDA kernel and gets kneadings set back from GPU"""
    grid_lengths = [len(param_grid) for _, param_grid in system.grid.items()]
    total_parameter_space_size = prod(grid_lengths)

    global module, events_len, events_in_int, dots_with_events_per_curve, total_events_len
    module                     = importlib.import_module(system.module_name)
    events_len                 = len(events)
    events_in_int              = 64 // events_len
    dots_with_events_per_curve = (n // events_in_int + (0 if n % events_in_int == 0 else 1))

    total_events_len      = dots_with_events_per_curve * total_parameter_space_size
    symbolic_hash_set_gpu = cuda.device_array(total_events_len)

    initPt_gpu               = cuda.to_device(initPt)
    system_parameters_gpu    = cuda.to_device(system.getParameters())
    parameters_places_gpu    = cuda.to_device(system.getParametersPlaces())
    parameters_to_change_gpu = cuda.to_device(system.getParametersToChange())
    grid_lengths_gpu         = cuda.to_device(grid_lengths)
    events_gpu               = cuda.to_device([event[0] for event in events])

    global system_dim, len_change_params, len_system_params
    system_dim        = len(initPt_gpu)
    len_change_params = len(parameters_to_change_gpu)
    len_system_params = len(system_parameters_gpu)

    max_len       = max(len(v) for v in system.grid.values())
    padded_arrays = [np.pad(np.array(v, dtype=np.float64), (0, max_len - len(v)), mode='constant') for v in system.grid.values()]
    grid_data     = np.stack(padded_arrays)
    grid_data_gpu = cuda.to_device(grid_data)

    dim_grid  = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    dim_block = THREADS_PER_BLOCK

    str1 = "Num of blocks per grid:"
    str2 = "Num of threads per block:"
    str3 = "Total Num of threads running:"
    str4 = [f"Count of '{param_name}' parameters:" for param_name, _ in system.grid.items()]
    str5 = "Number of all parameters combinations:"

    all_strings = [str1, str2, str3, *str4, str5]
    max_str_len = max(len(s) for s in all_strings)

    print(f"{str1:<{max_str_len}} {dim_grid}")
    print(f"{str2:<{max_str_len}} {dim_block}")
    print(f"{str3:<{max_str_len}} {dim_grid * dim_block}")
    for (_, param_grid), str4_elem in zip(system.grid.items(), str4):
       print(f"{str4_elem:<{max_str_len}} {len(param_grid)}")
    print(f"{str5:<{max_str_len}} {total_parameter_space_size}")

    # Call CUDA kernel
    sweep_threads[dim_grid, dim_block](
        symbolic_hash_set_gpu,
        total_parameter_space_size,
        initPt_gpu,
        system_parameters_gpu,
        parameters_places_gpu,
        parameters_to_change_gpu,
        grid_data_gpu,
        grid_lengths_gpu,
        dt,
        n,
        tSkip,
        events_gpu
    )

    symbolic_representation_set = np.zeros(total_events_len)
    symbolic_hash_set_gpu.copy_to_host(symbolic_representation_set)

    return symbolic_representation_set, total_parameter_space_size, grid_lengths, dots_with_events_per_curve, events_in_int, events_len
