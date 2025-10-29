from math import fabs, e
from numpy import arange
import numpy as np
import pandas as pd


def f_x (t, x, y):

    return 998 * x + 1998 * y

def f_y (t, x, y):

    return 999 * x - 1999 * y

def check_x (t):

    return 4 * e ** (-t) - 3 * e ** (-1000 * t)

def check_y (t):

    return -2 * e ** (-t) + 3 * e ** (-1000 * t)

def runge1 (x0, y0, h0, t0, eps, iterations):

    y_nabl = y0
    x_nabl = x0

    h = h0
    i = 1

    while i < iterations + 1:

        copy_x = x_nabl
        copy_y = y_nabl

        x_nabl = x0 + h * f_x (t0 + h,copy_x, copy_y)
        y_nabl = y0 + h * f_y (t0 + h, copy_x, copy_y)

        if (fabs(x_nabl - copy_x) + fabs(y_nabl - copy_y) < eps):
            return (x_nabl, y_nabl, h)
        
        elif (i == iterations):
            i = 0
            h /= 2

            y_nabl = y0
            x_nabl = x0

        i += 1

def javna_3_8 (values_list, t_list, h_opt, n):

    for i in range(4, n):

        x_new = values_list[i - 3][0] + 3 / 8 * h_opt * (7 * f_x(t_list[i - 1],values_list[i - 1][0], values_list[i - 1][1]) - 3 * 
                                                         f_x (t_list[i - 2],values_list[i - 2][0], values_list[i - 2][1]) + 5 * f_x (t_list[i - 3],values_list[i - 3][0], values_list[i - 3][1]) - 
                                                         f_x (t_list[i - 4], values_list[i - 4][0], values_list[i - 4][1]))
        
        y_new = values_list[i - 3][1] + 3 / 8 * h_opt * (7 * f_y(t_list[i - 1], values_list[i - 1][0], values_list[i - 1][1]) - 3 * 
                                                         f_y (t_list[i - 2], values_list[i - 2][0], values_list[i - 2][1]) + 5 * f_y (t_list[i - 3], values_list[i - 3][0], values_list[i - 3][1]) - 
                                                         f_y (t_list[i - 4], values_list[i - 4][0], values_list[i - 4][1]))
        
        values_list.append ((x_new, y_new))

def real_values (t_list):
    
    real_list = []

    for i in t_list:

        x_real = check_x(i)
        y_real = check_y(i)

        real_list.append((x_real, y_real))

    return real_list

def main (h, eps, k):

    if h == 1 or h % 2 == 0:

        print ('bad step')
        return
    
    values_list = [(1,1)]
    h_list = []
    t_list = [0]

    h_new = h

    # variant 1

    # for i in range (1, 4):

    #     x, y, h_new = runge1(values_list[i - 1][0], values_list[i - 1][1], h_new, t_list[i - 1], eps, k)

    #     values_list.append ((x, y))
    #     h_list.append (h_new)
    #     t_list.append(t_list[i-1] + h_new)

    # h_opt = min (h_list)

    # variant 2

    x, y, h_new = runge1(values_list[0][0], values_list[0][1], h, 0, eps, k)
    x_pos, y_pos, h_new_pos = runge1(values_list[0][0], values_list[0][1], h_new / 2, 0, eps, k)
    x_toCheck, y_toCheck, h_new_pos = runge1(x_pos, y_pos, h_new / 2 + h_new / 2, h_new/2, eps, k)

    # if (fabs(x_toCheck - x) + fabs(y_toCheck - y) < eps): 

    #     print("good h")

    while (fabs(x_toCheck - x) + fabs(y_toCheck - y) > eps):

        h_new /= 2

        x, y = x_pos, y_pos
        x_pos, y_pos, h_new_pos = runge1(values_list[0][0], values_list[0][1], h_new / 2, 0, eps, k)
        x_toCheck, y_toCheck, h_new_pos = runge1(x_pos, y_pos, h_new / 2, h_new/2, eps, k)

    h_opt = h_new
    #==============================================================================================
    values_list = values_list[0:1]
    t_list = t_list[0:1]
    h_list = []

    for i in range (1, 4):

        x, y, h_new = runge1(values_list[i - 1][0], values_list[i - 1][1], h_opt, t_list[i - 1], eps, k)

        values_list.append ((x, y))
        h_list.append (h_new)
        t_list.append(t_list[i-1] + h_new)

    n = 100

    t_list = [i * h_opt for i in range(n)]

    javna_3_8 (values_list, t_list, h_opt, n)

    real_list = real_values (t_list)

    real_list = np.array (real_list)
    values_list = np.array (values_list)

    # ======================================

    delta_x = np.fabs(values_list[:, 0] - real_list[:, 0]).reshape(-1, 1)
    delta_y = np.fabs(values_list[:, 1] - real_list[:, 1]).reshape(-1, 1)

    x_val = values_list[:, 0]
    y_val = values_list[:, 1]

    x_real = real_list[:, 0]
    y_real = real_list[:, 1]

    t_list = np.array (t_list).reshape (-1, 1)

    result = np.column_stack ((t_list, x_val, y_val, x_real, y_real, delta_x, delta_y))
    columns = ['t', 'x', 'y', 'x_real', 'y_real', 'delta_x', 'delta_y']

    result = pd.DataFrame(result, columns=columns)

    return result

result = main(0.2, 0.00005, 2000)

print(result)