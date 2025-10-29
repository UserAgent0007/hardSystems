from math import fabs, e
from numpy import arange
import numpy as np
import pandas as pd
# import streamlit as st

def f_x (t, x, y):

    return 998 * x + 1998 * y

def f_y (t, x, y):

    return -999 * x - 1999 * y

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
            return (x_nabl, y_nabl)

        i += 1
    print ("sth went wrong")
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

def main (eps, k):
    
    values_list = [(1,1)]
    t_list = [0]

    t1 = 5 / 1000
    h_min = t1 / 100
    h_max = t1 / 3 
    n_min = 101
    n_max = int((1 - t1) // h_max)


    #==============================================================================================
    

    for i in range (1, 4):

        x, y = runge1(values_list[i - 1][0], values_list[i - 1][1], h_min, t_list[i - 1], eps, k)

        values_list.append ((x, y))

        t_list.append(t_list[i-1] + h_min)



    t_list = [i * h_min for i in range(n_min)]

    javna_3_8 (values_list, t_list, h_min, n_min)

    real_list_min = real_values (t_list)

    real_list_min = np.array (real_list_min)
    values_list_min = np.array (values_list)

    # ======================================

    delta_x_min = np.fabs(values_list_min[:, 0] - real_list_min[:, 0]).reshape(-1, 1)
    delta_y_min = np.fabs(values_list_min[:, 1] - real_list_min[:, 1]).reshape(-1, 1)

    x_val_min = values_list_min[:, 0]
    y_val_min = values_list_min[:, 1]

    x_real_min = real_list_min[:, 0]
    y_real_min = real_list_min[:, 1]

    t_list_copy = np.array (t_list).reshape (-1, 1)

    result = np.column_stack ((t_list_copy, x_val_min, y_val_min, x_real_min, y_real_min, delta_x_min, delta_y_min))
    columns = ['t', 'x', 'y', 'x_real', 'y_real', 'delta_x', 'delta_y']

    result = pd.DataFrame(result, columns=columns)

    # ===================================

    t_list_max = t_list[len(t_list) - 4::] + [t_list[-1] + i * h_max for i in range(1, n_max+1)]
    values_list_max = values_list[len(values_list_min)-4::]

    javna_3_8 (values_list_max, t_list_max, h_max, n_max + 4)

    real_list_max = real_values (t_list_max)
    real_list_max = np.array (real_list_max)

    values_list_max = np.array(values_list_max)

    delta_x_max = np.fabs(values_list_max[4:, 0] - real_list_max[4:, 0]).reshape(-1, 1)
    delta_y_max = np.fabs(values_list_max[4:, 1] - real_list_max[4:, 1]).reshape(-1, 1)

    x_val_max = values_list_max[4:, 0]
    y_val_max = values_list_max[4:, 1]

    x_real_max = real_list_max[4:, 0]
    y_real_max = real_list_max[4:, 1]

    t_list_max = np.array (t_list_max[4:]).reshape (-1, 1)

    result_max = np.column_stack ((t_list_max, x_val_max, y_val_max, x_real_max, y_real_max, delta_x_max, delta_y_max))
    columns = ['t', 'x', 'y', 'x_real', 'y_real', 'delta_x', 'delta_y']

    result_max = pd.DataFrame(result_max, columns=columns)


    return (result, result_max)
    
result = main(0.00005, 2000)

print(result[0])
print("#=============================")
print(result[1])
