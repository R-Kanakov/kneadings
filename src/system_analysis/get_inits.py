import numpy as np
import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so
import scipy

from src.cuda_sweep.sweep_fbpo import PARAM_TO_INDEX


def find_equilibrium_by_guess(rhs, jac, initial_guess=np.zeros(3), tol=1e-10):
    """Находит состояние равновесия системы для заданных параметров."""
    initial_guess = np.asarray(initial_guess)

    # result = scipy.optimize.root(
    #     rhs,
    #     initial_guess,
    #     method='krylov',
    #     options={'xtol': tol}  # 'jac_options': method
    # )
    result = scipy.optimize.root(
        rhs,
        initial_guess,
        jac=jac,
        method='lm',
        options={'xtol': tol}
    )
    if not result.success:
        return None

    eq_coords = result.x
    eq_obj = sf.getEquilibriumInfo(eq_coords, jac)

    return eq_obj


def continue_equilibrium(rhs, jac, get_params, set_params, param_to_index, param_x_name, param_y_name, start_eq_coords,
                         up_n, down_n, left_n, right_n, up_step, down_step, left_step, right_step):
    """Продолжает состояние равновесия по сетке параметров"""
    rows = up_n + down_n + 1
    cols = left_n + right_n + 1
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    # начало координат слева снизу
    # обработка стартовой точки
    start_row, start_col = down_n, left_n
    start_eq_obj = sf.getEquilibriumInfo(start_eq_coords, jac)
    grid[down_n][left_n] = start_eq_obj

    # список всех возможных смещений
    deltas = []
    for dj in range(-down_n, up_n + 1):
        for di in range(-left_n, right_n + 1):
            if (di, dj) == (0, 0):
                continue
            deltas.append((di, dj))

    # Так ли важно использовать новую полученную точку как приближение?
    # На моём ПК эта функция занимает 1.5 минуты.
    # Не будет ли быстрее сюда же навесить куду и для каждой точки на сетке
    # найти (или не найти) состояние равновесия?
    # Или здесь пострадает точность?

    # сортируем по удалению от центра (сначала ближайшие точки)
    deltas.sort(key=lambda delta: abs(delta[0]) + abs(delta[1]))

    params = get_params()
    param_x = params[param_to_index[param_x_name]]
    param_y = params[param_to_index[param_y_name]]

    # находим с.р. в каждой точке
    for di, dj in deltas:
        curr_i = start_col + di
        curr_j = start_row + dj

        dx = di * (up_step if di > 0 else down_step)
        dy = dj * (right_step if dj > 0 else left_step)

        curr_param_x = param_x + dx
        curr_param_y = param_y + dy

        set_params({param_x_name: curr_param_x, param_y_name: curr_param_y})

        neighbors = []
        for ni, nj in [(curr_i - 1, curr_j), (curr_i + 1, curr_j),
                       (curr_i, curr_j - 1), (curr_i, curr_j + 1)]:
            if 0 <= ni < cols and 0 <= nj < rows and grid[nj][ni] is not None:
                neighbors.append(grid[nj][ni].coordinates)

        eq_obj = None
        for guess in neighbors:
            eq_obj = find_equilibrium_by_guess(rhs, jac, initial_guess=guess)
            if eq_obj is not None:
                break

        if eq_obj is not None:
            grid[curr_j][curr_i] = eq_obj
            #print(
            #    f"Node ({curr_i}, {curr_j}) | Equilibrium {eq_obj.coordinates} was found "
            #    f"with parameters ({curr_param_x:.3f}, {curr_param_y:.3f})"
            #    )
        else:
            #print(
            #    f"Node ({curr_i}, {curr_j}) | No equilibrium was found "
            #    f"with parameters ({curr_param_x:.3f}, {curr_param_y:.3f})"
            #    )
            _ = _

    return grid


def get_saddle_foci_grid(grid, up_n, down_n, left_n, right_n):
    """Составляет сетку седло-фокусов по сетке состояний равновесия"""
    print("Filling up saddle-foci grid...")
    sf_grid = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            if grid[j][i] is not None and sf.is3DSaddleFocusWith1dU(grid[j][i], sf.STD_PRECISION):
                sf_grid[j][i] = grid[j][i]
            #     print(f"{i + j * (left_n + right_n + 1)} {i, j} -- saddle-focus")
            # else:
            #     print(f"{i + j * (left_n + right_n + 1)} {i, j} -- none")

    return sf_grid


def find_inits_for_equilibrium_grid(sf_grid, dim, up_n, down_n, left_n, right_n):
    """Находит начальные условия для сетки седло-фокусов"""
    # допустим, что grid на самом деле это просто двумерный массив с eq_objs
    # тогда мы можем записывать inits сразу в одномерный массив
    # опираясь только на i j в двумерном массиве grid
    print("Finding initial conditions...")

    inits = np.empty(dim * (left_n + right_n + 1) * (up_n + down_n + 1))
    nones = []  # массив индексов там, где None. Нужен при обходе нидингов

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            eq_obj = sf_grid[j][i]
            if eq_obj is not None:
                if sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, sf.pickCirSeparatrix, sf.STD_PRECISION):
                    init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, sf.pickCirSeparatrix, sf.STD_PRECISION)[0]
                    for k in range(dim):
                        inits[index * dim + k] = init_pt[k]
                    # print(f"{index} {init_pt}")
                else:
                    nones.append(index)
                    # print(f"{index} {i, j} None: no initial condition was found for the saddle-focus")
            else:
                nones.append(index)
                # print(f"{index} {i, j} None: no equilibrium object")

    # print(f"\nNones: {nones}")

    return inits, nones


def generate_parameters(start_params, up_n, down_n, left_n, right_n,
                        up_step, down_step, left_step, right_step):
    """Генерирует массивы параметров для последующего подсчёта нидингов"""
    print("Generating parameters...")
    start_params_x = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    start_params_y = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            da = (i - left_n) * (right_step if i > left_n else left_step)
            db = (j - down_n) * (up_step if j > down_n else down_step)

            start_params_x[index] = start_params[0] + da
            start_params_y[index] = start_params[1] + db
            # print(f"param1_{index} {i, j} {start_params_x[index]}")
            # print(f"param2_{index} {i, j} {start_params_y[index]}")

    return start_params_x, start_params_y


if __name__ == '__main__':
    w = 0
    a = -2.911209192326542
    b = -1.612684228842761
    r = 1.0

    # start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)

    # # поиск стартового седло-фокуса
    # bounds = [(-0.1, 2 * np.pi + 0.1)] * 2
    # borders = [(-1e-15, 2 * np.pi + 1e-15)] * 2
    #
    # # первые две функции -- общая система, вторые две -- в которой ищем с.р., дальше функция приведения
    # equilibria = sf.findEquilibria(lambda psis: start_sys.getReducedSystem(psis), lambda psis: start_sys.getReducedSystemJac(psis),
    #                                lambda psis: start_sys.getRestriction(psis), lambda psis: start_sys.getRestrictionJac(psis),
    #                                lambda phi: np.concatenate([[0.], phi]), bounds, borders,
    #                                sf.ShgoEqFinder(1000, 1, 1e-10),
    #                                sf.STD_PRECISION)
    #
    # start_eq = None
    # for eq in equilibria:  # перебираем все с.р., которые были найдены
    #     print(f"{sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION)} at {eq.coordinates}")
    #     if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
    #         start_eq = np.array(eq.coordinates)
    #         print(f"Starting with saddle-focus {start_eq.round(4)} with parameters ({w:.3f}, {a:.3f}, {b:.3f}, {r:.3f})")
    #         break

    start_eq = [0.0, 2.30956058, 4.75652024]

    up_n = 10
    down_n = 10
    left_n = 10
    right_n = 10

    up_step = 0.001
    down_step = 0.001
    left_step = 0.001
    right_step = 0.001

    start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)
    reduced_rhs_wrapper = start_sys.getReducedSystem
    reduced_jac_wrapper = start_sys.getReducedSystemJac
    get_params = start_sys.getParams
    set_params = start_sys.setParams

    if start_eq is not None:
        eq_grid = continue_equilibrium(reduced_rhs_wrapper, reduced_jac_wrapper, get_params, set_params,
                                       PARAM_TO_INDEX, 'a', 'b',
                                       start_eq, up_n, down_n, left_n, right_n,
                                       up_step, down_step, left_step, right_step)
        sf_grid = get_saddle_foci_grid(eq_grid, up_n, down_n, left_n, right_n)
        inits, nones = find_inits_for_equilibrium_grid(sf_grid, 3, up_n, down_n, left_n, right_n,)
        params_x, params_y = generate_parameters((a, b), up_n, down_n, left_n, right_n,
                                            up_step, down_step, left_step, right_step)
    else:
        print("Start saddle-focus was not found")

    # for j in range(up_n + down_n + 1):
    #     for i in range(left_n + right_n + 1):
    #         index = i + j * (left_n + right_n + 1)
    #         if index in nones:
    #             print(f"Node ({i}, {j}) | {index} | IS IN NONES")
    #         else:
    #             print(f"Node ({i}, {j}) | {index} | Init {inits[index * 3 + 0], inits[index * 3 + 1], inits[index * 3 + 2]} "
    #                   f"for equilibrium {sf_grid[j][i].coordinates} with parameters {params_x[index], params_y[index]}")

    # np.savez(
    #     'inits.npz',
    #     inits=inits,
    #     nones=nones,
    #     alphas=params_x,
    #     betas=params_y,
    #     up_n=up_n,
    #     down_n=down_n,
    #     left_n=left_n,
    #     right_n=right_n
    # )