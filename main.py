import math

import numpy as np
from typing import Callable, List, Dict
from random import random
from matplotlib import pyplot as plt

VERTEX_CNT = 1000
ITERATIONS = 50000
T_0 = 10
T_COEFF = 0.88
EPS = 10 ** (-(np.log10(T_0) + 1))

functions: Dict = {
    "Himmelblau": [lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2, [[-5, 5], [-5, 5]],
                   [
                       [3.0, 2.0, 0.0],
                       [-2.805118, 3.131312, 0.0],
                       [-3.779310, -3.283186, 0.0],
                       [3.584428, -1.848126, 0.0]
                   ]],
    "Booth": [lambda x, y: (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2, [[-10, 10], [-10, 10]], [[1, 3, 0]]],
    "Easom": [lambda x, y: -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2)), [[-100, 100],
                                                                                                     [-100, 100]],
              [[np.pi, np.pi, -1]]],
    "Three_hump_camel": [lambda x, y: 2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2, [[-5, 5], [-5, 5]],
                         [[0, 0, 0]]],
    "Rozenbrok": [lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2, [[-10, 10], [-10, 10]], [[1, 1, 0]]]
}

annealings: Dict = {
    "Boltzmann": [
        lambda t, k: T_0 / np.log(1 + k),
        lambda state, temperature: state + np.random.normal(0, 1, len(state)) * np.sqrt(temperature)
    ],
    "Cauchy": [
        lambda t, k: T_0 / k,
        lambda state, temperature:
        state + (np.random.normal(0, 1, len(state)) / np.random.standard_cauchy()) * np.sqrt(temperature)
    ],
    "Tushenie": [
        lambda t, k: T_COEFF * t,
        lambda state, temperature: state + np.random.normal(0, 1, len(state)) * np.sqrt(temperature)
    ]
}


def generate_3d(boundaries: List[List[float]]) -> [List[float], List[float]]:
    axes = [np.linspace(start, end, VERTEX_CNT) for start, end in boundaries]
    return np.meshgrid(axes[0], axes[1])


def annealing_function(boundaries: List[List[float]], function: Callable[[float, float], float], iterations: int,
                       temperature_law: Callable[[float, int], float],
                       g: Callable[[List[float], float], List[float]],
                       start_vertex: List[float]
                       ) -> list[list[float]]:
    # function value in vertex (energy)
    energy = function(start_vertex[0], start_vertex[1])
    minimum_ = energy
    minimum_state = start_vertex
    state = start_vertex
    temperature = T_0
    min_temp = temperature

    results = [[state[0], state[1], energy]]
    k = 0

    for k in range(1, iterations):
        if energy < minimum_:
            minimum_ = energy
            minimum_state = state
            min_temp = temperature
            results.append([state[0], state[1], energy])
        for i in range(1, iterations):
            state_ = g(state, temperature)
            if boundaries[0][0] < state_[0] < boundaries[0][1] and boundaries[1][0] < state_[1] < boundaries[1][1]:
                energy_ = function(state_[0], state_[1])
                alpha = random()
                if alpha < h2(energy - energy_, temperature):
                    state = state_
                    energy = energy_
                    break
        temperature = temperature_law(temperature, k)
    # print("Minimum value {} was detected in ({}, {}). Temperature: {}".format(minimum_, minimum_state[0],
    #                                                                           minimum_state[1], min_temp))
    results.append([minimum_state[0], minimum_state[1], minimum_])
    return results


def h(delta_e: float, t: float) -> float:
    return 1 / (1 + np.exp(delta_e / t))


def h2(delta_e: float, t: float) -> float:
    return np.exp(delta_e / t)


for function_name in functions:
    print("Testing function: ", function_name)
    curr_function = functions[function_name]
    x, y = generate_3d(boundaries=curr_function[1])
    z = np.vectorize(curr_function[0])(x, y)
    random_vertex = [
        curr_function[1][0][0] + (random() * (curr_function[1][0][1] - curr_function[1][0][0])),
        curr_function[1][1][0] + (random() * (curr_function[1][1][1] - curr_function[1][1][0])),
    ]
    print("Starting from point: [{}; {}]\n".format(random_vertex[0], random_vertex[1]))
    minimums = curr_function[2]
    for name in annealings:
        print("Annealing method: ", name)
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')

        x_, y_, z_ = zip(*annealing_function(curr_function[1], curr_function[0], ITERATIONS,
                                             annealings[name][0],
                                             annealings[name][1], random_vertex))

        dif = 10000
        for minimum in minimums:
            if math.fabs(minimum[2] - z_[-1]) < dif:
                dif = math.fabs(minimum[2] - z_[-1])

        print("Difference from closest actual minimum ", dif)
        print("==========================================================")
        ax.scatter(x_, y_, z_, c='green')
        plt.grid(True)
        plt.title(name)
    plt.tight_layout()
    plt.show()

# if __name__ == '__main__':
#     TEST_ITER_CNT = 1000
#
#     for function_name in functions:
#         print(function_name)
#         curr_function = functions[function_name]
#         average_dif = dict()
#         minimums = curr_function[2]
#         for i in range(TEST_ITER_CNT):
#             random_vertex = [
#                 curr_function[1][0][0] + (random() * (curr_function[1][0][1] - curr_function[1][0][0])),
#                 curr_function[1][1][0] + (random() * (curr_function[1][1][1] - curr_function[1][1][0])),
#                 ]
#             for name in annealings:
#                 results, iter_cnt = annealing_function(curr_function[1], curr_function[0], ITERATIONS,
#                                                        annealings[name][0],
#                                                        annealings[name][1], random_vertex)
#                 x_, y_, z_ = zip(*results)
#
#                 res = [x_[-1], y_[-1], z_[-1]]
# dif = 10000
# for minimum in minimums:
#     if math.fabs(minimum[2] - z_[-1]) < dif:
#         dif = math.fabs(minimum[2] - z_[-1])
# if name not in average_dif:
#     average_dif[name] = dif
# else:
#     average_dif[name] = (average_dif[name] + dif) / 2
# print("=======================================")
# for annealing_name in average_dif:
#     print("{}. Average difference from actual minimum: {}".format(annealing_name, average_dif[annealing_name]))
# print("=======================================")

# for function_name in functions:
#     print(function_name)
#     curr_function = functions[function_name]
#     lab2_func_format = lambda point: curr_function[0](point[0], point[1])
#     lab2_functions = dict()
#     lab4_functions = dict()
#     for i in range(TEST_ITER_CNT):
#         random_vertex = [
#             curr_function[1][0][0] + (random() * (curr_function[1][0][1] - curr_function[1][0][0])),
#             curr_function[1][1][0] + (random() * (curr_function[1][1][1] - curr_function[1][1][0])),
#         ]
#
#         simple = gradient_descent_simple(lab1_func_format, random_vertex, 0.1)
#         dichotomy = gradient_descent_dichotomy(lab1_func_format, random_vertex, 1)
#         nelder_ans = nelder_mead_method(lab1_func_format, random_vertex)
#         nelder = nelder_ans.x, nelder_ans.nit, lab1_func_format(nelder_ans.x)
#
#         if i == 0:
#             lab1_functions["Gradient with constant LR"] = [simple[0][0], simple[0][1], simple[2]]
#             lab1_functions["Gradient with dichotomy"] = [dichotomy[0][0], dichotomy[0][1], dichotomy[2]]
#             lab1_functions["Nelder-Mead"] = [nelder[0][0], nelder[0][1], nelder[2]]
#         else:
#             avg_simple = lab1_functions["Gradient with constant LR"]
#             lab1_functions["Gradient with constant LR"] = [
#                 (avg_simple[0] + simple[0][0]) / 2,
#                 (avg_simple[1] + simple[0][1]) / 2,
#                 (avg_simple[2] + simple[2]) / 2,
#             ]
#             avg_dichotomy = lab1_functions["Gradient with dichotomy"]
#             lab1_functions["Gradient with dichotomy"] = [
#                 (avg_dichotomy[0] + dichotomy[0][0]) / 2,
#                 (avg_dichotomy[1] + dichotomy[0][1]) / 2,
#                 (avg_dichotomy[2] + dichotomy[2]) / 2,
#             ]
#             avg_nelder = lab1_functions["Nelder-Mead"]
#             lab1_functions["Nelder-Mead"] = [
#                 (avg_nelder[0] + nelder[0][0]) / 2,
#                 (avg_nelder[1] + nelder[0][1]) / 2,
#                 (avg_nelder[2] + nelder[2]) / 2,
#             ]
#         iter_cnt = max(max(nelder[1], dichotomy[1]), simple[1])
#
#         for name in annealings:
#             results, iter_cnt = annealing_function(curr_function[1], curr_function[0], iter_cnt,
#                                                    annealings[name][0],
#                                                    annealings[name][1], random_vertex)
#             if name not in lab4_functions:
#                 lab4_functions[name] = results[-1]
#             else:
#                 avg_values = lab4_functions[name]
#                 lab4_functions[name] = [(avg_values[0] + results[-1][0]) / 2, (avg_values[1] + results[-1][1]) / 2,
#                                         (avg_values[2] + results[-1][2]) / 2]
#     print("===============================================================")
#     print("Results for {} iterations".format(TEST_ITER_CNT))
#     print("Lab 1 methods results: ")
#
#     for name in lab1_functions:
#         print(name)
#         lab1_func = lab1_functions[name]
#         print("Average minimum value: {}. Average minimum point: ({}; {})".format(lab1_func[2], lab1_func[0],
#                                                                                   lab1_func[1]))
#     print("----------------------------------------------------------------")
#     print("Lab 4 methods results: ")
#
#     for name in lab4_functions:
#         print(name)
#         lab4_func = lab4_functions[name]
#         print("Average minimum value: {}. Average minimum point: ({}; {})".format(lab4_func[2], lab4_func[0],
#                                                                                   lab4_func[1]))
#
#     print("===============================================================")
# options = {"tolerance": 1e-5, "max_epoch": 2000, "learning_rate": 1, "eps": 1e-4, "contour_draw": True}
# for function_name in functions:
#     print(function_name)
#     curr_function = functions[function_name]
#     func = lambda point: curr_function[0](point[0], point[1])
#     lab2_functions = dict()
#     lab4_functions = dict()
#
#     for i in range(TEST_ITER_CNT):
#         random_vertex = [
#             curr_function[1][0][0] + (random() * (curr_function[1][0][1] - curr_function[1][0][0])),
#             curr_function[1][1][0] + (random() * (curr_function[1][1][1] - curr_function[1][1][0])),
#         ]
#
#         newton = newton_method(random_vertex, func, function_name, options)
#         dich_newton = dichotomy_newton_method(random_vertex, func, function_name, options)
#         scipy_ans = newton_method_scipy(func, random_vertex, function_name, function_name)
#         scipy_newton = scipy_ans.x, scipy_ans.nit, func(scipy_ans.x)
#         quasi_ans = bfgs_method_scipy(func, random_vertex)
#         quasi_newton = quasi_ans.x, quasi_ans.nit, func(quasi_ans.x)
#
#         if i == 0:
#             lab2_functions["Newton method"] = [newton[0][0], newton[0][1], newton[2]]
#             lab2_functions["Newton method with dichotomy"] = [dich_newton[0][0], dich_newton[0][1], dich_newton[2]]
#             lab2_functions["Newton method from scipy"] = [scipy_newton[0][0], scipy_newton[0][1], scipy_newton[2]]
#             lab2_functions["Quasi Newton method"] = [quasi_newton[0][0], quasi_newton[0][1], quasi_newton[2]]
#         else:
#             avg_newton = lab2_functions["Newton method"]
#             lab2_functions["Newton method"] = [
#                 (avg_newton[0] + newton[0][0]) / 2,
#                 (avg_newton[1] + newton[0][1]) / 2,
#                 (avg_newton[2] + newton[2]) / 2,
#             ]
#             avg_dich_newton = lab2_functions["Newton method with dichotomy"]
#             lab2_functions["Newton method with dichotomy"] = [
#                 (avg_dich_newton[0] + dich_newton[0][0]) / 2,
#                 (avg_dich_newton[1] + dich_newton[0][1]) / 2,
#                 (avg_dich_newton[2] + dich_newton[2]) / 2,
#             ]
#             avg_scipy_newton = lab2_functions["Newton method from scipy"]
#             lab2_functions["Newton method from scipy"] = [
#                 (avg_scipy_newton[0] + scipy_newton[0][0]) / 2,
#                 (avg_scipy_newton[1] + scipy_newton[0][1]) / 2,
#                 (avg_scipy_newton[2] + scipy_newton[2]) / 2,
#             ]
#             avg_quasi_newton = lab2_functions["Quasi Newton method"]
#             lab2_functions["Quasi Newton method"] = [
#                 (avg_quasi_newton[0] + quasi_newton[0][0]) / 2,
#                 (avg_quasi_newton[1] + quasi_newton[0][1]) / 2,
#                 (avg_quasi_newton[2] + quasi_newton[2]) / 2,
#             ]
#         iter_cnt = max(max(newton[1], dich_newton[1]), max(scipy_newton[1], quasi_newton[1]))
#
#         for name in annealings:
#             results = annealing_function(curr_function[1], curr_function[0], iter_cnt,
#                                                    annealings[name][0],
#                                                    annealings[name][1], random_vertex)
#             if name not in lab4_functions:
#                 lab4_functions[name] = results[-1]
#             else:
#                 avg_values = lab4_functions[name]
#                 lab4_functions[name] = [(avg_values[0] + results[-1][0]) / 2, (avg_values[1] + results[-1][1]) / 2,
#                                         (avg_values[2] + results[-1][2]) / 2]
#     print("===============================================================")
#     print("Results for {} iterations".format(TEST_ITER_CNT))
#     print("Lab 1 methods results: ")
#
#     for name in lab2_functions:
#         print(name)
#         lab2_func = lab2_functions[name]
#         print("Average minimum value: {}. Average minimum point: ({}; {})".format(lab2_func[2], lab2_func[0],
#                                                                                   lab2_func[1]))
#     print("----------------------------------------------------------------")
#     print("Lab 4 methods results: ")
#
#     for name in lab4_functions:
#         print(name)
#         lab4_func = lab4_functions[name]
#         print("Average minimum value: {}. Average minimum point: ({}; {})".format(lab4_func[2], lab4_func[0],
#                                                                                   lab4_func[1]))
#
#     print("===============================================================")
