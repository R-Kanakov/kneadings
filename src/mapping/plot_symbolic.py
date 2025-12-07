import sys
import colorsys
import time
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def bits_to_fraction(arr):
    numerator         = 0
    found_first_one   = False
    first_event_found = -1

    s = 0
    for elt in arr:
        s += elt
    if s == 0:
        return 0, -1

    for i in range(dots_with_events_per_curve): # uint64 number
        uint64_val = int(arr[i])

        for j in range(events_in_int): # dot
            mask = (1 << events_len) - 1
            elem = (uint64_val >> (j * events_len)) & mask

            for k in range(events_len): # events in dot
                bit = (elem >> k) & 1
                if not found_first_one and bit == 1:
                        found_first_one = True
                        numerator = 1
                        first_event_found = k
                else:
                    numerator = (numerator << 1) | bit

    return numerator, first_event_found


def remove_leading_zeros(num):
    if num == 0:
        return 0
    while (num & 1) == 0:
        num >>= 1

    return num


def value_to_color(value):
    if value == 0.0:
        r, g, b = [1.0, 1.0, 1.0]
    else:
        r, g, b = colorsys.hsv_to_rgb(value, 1.0, 1.0)
    return [r, g, b]


def list_to_array(colors_list, grid_lengths):
    rgb_array = [np.array(color_list, dtype=np.uint8) for color_list in colors_list]
    rgb_array = np.array(rgb_array)
    rgb_array = rgb_array.reshape(*grid_lengths, 3)
    return rgb_array


def plot_symbolic_representation(workerResult, dir, extension, grid):
    sys.set_int_max_str_digits(0)

    global dots_with_events_per_curve, events_in_int, events_len
    symbolic_representation_set, total_parameter_space_size, grid_lengths, dots_with_events_per_curve, events_in_int, events_len = workerResult

    start = time.time()

    result = []
    for arr in np.array_split(symbolic_representation_set, total_parameter_space_size):
        res, fst = bits_to_fraction(arr)
        color = []
        if fst != -1:
            res_without_zeros = res #remove_leading_zeros
            digits = len(str(res_without_zeros))
            frac_result = res_without_zeros / (10 ** digits)
            color = value_to_color(frac_result)
        else:
            color = value_to_color(0.0)

        result.append([int(c * 255) for c in color])
    result = list_to_array(result, grid_lengths)

    grid_params = [name for name, _ in grid.items()]
    grid_vals   = [[min(param_grid), max(param_grid)] for _, param_grid in grid.items()]

    img_rgb = result.astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.imshow(img_rgb, extent=[*grid_vals[1], *grid_vals[0]])
    ax.set_xlabel(f"{grid_params[1]}", fontsize = 12)
    ax.set_xlim(grid_vals[1])
    ax.set_ylabel(f"{grid_params[0]}", fontsize = 12)
    ax.set_ylim(grid_vals[0])
    plt.title('Symbolic Representation Hash Heatmap', fontsize = 14, pad = 20)
    plt.colorbar(label = 'Value Intensity')

    plt.savefig(f"{dir}/symbolic_out.{extension}", dpi = 300, bbox_inches = 'tight')
    plt.close()

    end = time.time()
    print(f"Took {end - start}s ({datetime.timedelta(seconds=end - start)})")
