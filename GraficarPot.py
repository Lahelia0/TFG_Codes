
import numpy 
import matplotlib.pyplot as plt
from random import uniform
from math import *
from numpy import *
from scipy.optimize import least_squares
from scipy.optimize import curve_fit


V_lsq = [ 0.048744, 0.37368474, 0.5998536, 0.7695813,0.90403349, 1.01496849, 1.1094415 ,1.19194217, 1.26546358, 1.33207786, 1.39326511, 1.45011074 ,1.50342881 , 1.60183395 , 1.64778744    , 1.73474346  , 1.81653325  , 1.8943967    , 1.96920634  , 2.04159398 ,  2.1120284   ,  2.24837694 , 2.28170546 , 2.31477974 ,  2.38024381  , 2.44490652 , 2.5088797, 2.57225527  , 2.63510954,  2.66636155  , 2.72855033,  2.79035922 , 2.85183043 ,  2.91300014   , 2.97389954 ,  3.06479996 , 3.09499204 , 3.12513436 ,  3.18527897,   3.24525077 ,  3.3050648 ,   3.36473439  , 3.4242714  , 3.45399356,  3.51335125 ,  3.57260088 ,  3.63175052 , 3.66129017 , 3.69080752 , 3.72030336 , 3.74977847 , 3.77923358 ,  3.83808656 ,  3.8968675, 3.95558116 , 3.98491413 ,  4.04353488 , 4.07282362 ,  4.13136004  , 4.18984449]
rad_test = linspace(0.5, 4.5,60)

rad_test2 = linspace(0.4, 4.5,37)
rad_test3 = linspace(0.5, 4.5,35)
Potencialteo2 = []

#Salen de simular una vez el programa para ajustar la curva
#se realiza una segunda simulacion añadiendo estos valores.
sigma = 0.7077707130967446
b=0.7110581836287054
c=1.2077608955279044
for i in range(37):
    Potencialteo = sigma*rad_test2[i] - b/rad_test2[i] + c
    Potencialteo2.append(Potencialteo)

def func(rad_test,sigma,b,c): 
    return sigma*rad_test - b/rad_test + c 

popt, pcov, = curve_fit(func, rad_test, V_lsq)

# Valores ajustados de sigma, b, y c y sus errores
sigma_opt, b_opt, c_opt = popt
perr = numpy.sqrt(numpy.diag(pcov))
sigma_err, b_err, c_err = perr

print("sigma_opt:", sigma_opt, "±", sigma_err)
print("b_opt:", b_opt, "±", b_err)
print("c_opt:", c_opt, "±", c_err)



fig, ax = plt.subplots()
ax.scatter(rad_test,V_lsq, marker = '+',c='blue',label='Potencial obtenido')
plt.plot(rad_test2, Potencialteo2,'grey', label="Ajuste Scipy")
plt.axis([0,3.3,0,5])
plt.legend(loc='upper right')
plt.title('Potencial Quark-Antiquark estático')
plt.xlabel('$r/a$')
plt.ylabel('$V(r)a$')
plt.show()
