from math import fabs, e, cos, sin, pi
from numpy import arange
import numpy as np
import pandas as pd
# import streamlit as st
import matplotlib.pyplot as plt

def f_x (t, y):

    return t / y

def f_y (t, x):

    return -t / x

def check_x (t):

    return e ** (t**2 / 2)

def check_y (t):

    return e ** (-t**2 / 2)

def fi1 (teta):

    return 8 * np.cos(teta) - 8 * np.cos(4 * teta)

def fi2 (teta):

    return 8 * np.sin(teta) - 8 * np.sin(4 * teta)

def fi3 (teta):

    return 21 * np.cos(3 * teta) - 9 * np.cos(2 * teta) + 15 * np.cos(teta) - 3

def fi4 (teta):

    return 21 * np.sin(3* teta) - 9 * np.sin(2*teta) + 15 * np.sin(teta)

def U(teta):

    return (fi1(teta) * fi3(teta) + fi2(teta) * fi4(teta)) / (fi3(teta)**2 + fi4(teta)**2)

def V(teta):

    return (fi2(teta)*fi3(teta) - fi1(teta)*fi4(teta))/(fi3(teta)**2 + fi4(teta)**2)

def runge1 (x0, y0, h0, t0, eps, iterations):

    y_nabl = y0
    x_nabl = x0

    h = h0
    i = 1

    while i < iterations + 1:

        copy_x = x_nabl
        copy_y = y_nabl

        x_nabl = x0 + h * f_x (t0 + h, copy_y)
        y_nabl = y0 + h * f_y (t0 + h, copy_x)

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

        x_new = values_list[i - 3][0] + 3 / 8 * h_opt * (7 * f_x(t_list[i - 1], values_list[i - 1][1]) - 3 * 
                                                         f_x (t_list[i - 2], values_list[i - 2][1]) + 5 * f_x (t_list[i - 3], values_list[i - 3][1]) - 
                                                         f_x (t_list[i - 4], values_list[i - 4][1]))
        
        y_new = values_list[i - 3][1] + 3 / 8 * h_opt * (7 * f_y(t_list[i - 1], values_list[i - 1][0]) - 3 * 
                                                         f_y (t_list[i - 2], values_list[i - 2][0]) + 5 * f_y (t_list[i - 3], values_list[i - 3][0]) - 
                                                         f_y (t_list[i - 4], values_list[i - 4][0]))
        
        values_list.append ((x_new, y_new))

def real_values (t_list):
    
    real_list = []

    for i in t_list:

        x_real = check_x(i)
        y_real = check_y(i)

        real_list.append((x_real, y_real))

    return real_list

def ro_check (ro):

    return (8 * (ro - ro**4)) / (3 * (7*ro**3 - 3*ro**2 + 5*ro - 1))

def main ():

    teta_coords = np.arange(0, 2001, 1)
    teta_coords = teta_coords * (pi / 1000)
    
    u = U(teta_coords)
    v = V(teta_coords)

    fig, ax = plt.subplots()

    ax.fill(u, v, alpha=0.5, color='grey')
    ax.scatter(u, v, s=20)

    checking = ro_check(-0.2)
    ax.scatter(-0.2, checking, color='r', s=20)

    plt.show()
# print(main (11, 0.00000002, 10))
main()