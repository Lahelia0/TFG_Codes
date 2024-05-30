import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import random
from numpy import *

#Function, coming from the functools packet,that given a function gives back the function
#computed in one of the variables of the function
#Funcion que dada una funcion nos devuelve dicha funcion computada
#en una de las variables de la funcion.

def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

#Solucion exacta (1dimension)
#input:-x:posicion
#output:-Resultado del propagador
#inner parameters:-E0: Energia del estado fundamental
#                 -T:intervalo de tiempo
def exact(x):
    E0=0.5
    T=4
    return math.exp(-E0*T)*(math.exp(-(x**2)/2)*(math.pi**(-1/4)))**2

#Funcion que computa la path integral de 7 dimensiones de un oscilador armonico
#input:-y:variable de integracion
#      -x:Condicion ciclica
#output:-funcion
#inner parameters:-N: Discretizacion del tiempo
#                 -a=T/N: tiempo discretizado
#                 -m:masa
def f(x,y):
    T=4.
    m=1.
    N=8
    a=T/N
    Slat = ((m/(2*a))*(y[0]-x)**2+a*((x)**2)/(2))
    for d in range(N-1):
        if d==N-2:
            Slat += ((m/(2*a))*(x-y[d])**2+a*((y[d])**2)/(2))
        else:
            Slat += ((m/(2*a))*(y[d+1]-y[d])**2+a*((y[d])**2)/(2))
    return math.exp(-(Slat))*((m/(a*2*math.pi))**(N/2))

#asignacion de arrays
exa=ones(100, 'double')
y=ones(100, 'double')
result=ones(10, 'double')
z=ones(10, 'double')
err=ones(10, 'double')

#Computo para 10 valores de x, usando la aproximacion de Monte Carlo
for s in range(10):
    z[s]=s*2/9
    integ = vegas.Integrator(7*[[-5, 5]])
    integration = integ(partial(f,z[s]), nitn=10, neval=100000)
    result[s]=integration.mean
    err[s]=integration.sdev
    print(z[s],result[s],err[s])

#25 valores del valor exacto
for s in range(100):
    y[s]=s*2/100
    exa[s] = exact(y[s])

#Plot
fig, ax = plt.subplots()
plt.plot(y,exa,'grey',label='Exacto')
ax.scatter(z, result,marker='+', color='blue', label='Vegas');
plt.legend(loc='upper right')
plt.axis([0,2.0,0.0,0.1])
plt.xlabel('x')
plt.ylabel(r'$\langle x|e^{-HT}|x\rangle$')
plt.title('Oscilador Armónico')
plt.show()