import matplotlib
from matplotlib import pyplot as plt
import vegas
import math
import random
from numpy import *

#Actualiza la posicion en funcion de una accion dada
#input:-x:vector posicion
#inner parameter:-eps:intervalo
#                -N:Numero de puntos en la lattice
def update(x):
    N=20
    eps=1.4
    for j in range(0,N):
        old_x = x[j] # save original value
        old_Sj = S(j,x)
        x[j] = x[j] + random.uniform(-eps,eps) # update x[j]
        dS = S(j,x) - old_Sj # change in action
        if dS>0 and exp(-dS)<random.uniform(0,1):
                x[j] = old_x # restore old value

#accion de un oscilador armonico
#input:-j:posicion donde se calcula la accion
#      -x:vector posicion
#inner parameter:-a:lattice spacing
#                -N:numero de puntos en la lattice
def S(j,x): 
    N=20
    a=1/2
    jp = (j+1)%N 
    jm = (j-1)%N 
    return a*x[j]**2/2 +x[j]*(x[j]-x[jp]-x[jm])/a

#Computar la funcion como un valor medio de la integral de camino
#input:-x:vector posicion
#      -n:Posicion
#inner parameters:-N:numero de puntos en la lattice
def compute_G(x,n):
    g = 0
    N=20
    for j in range(0,N):
        g = g + (x[j])*(x[(j+n)%N])
    return g/N
    
#Asignacion de arrays y parametros
a=1/2
N=20
N_cf=10000
N_cor=20
x=ones(N, 'double')
G=ones((N_cf,N), 'double')
avg_G=ones(N, 'double')
avg_Gsquare=ones(N, 'double')
errorG=ones(N, 'double')

#Computar el MC value del propagador para todos los puntos.
for j in range(0,N): #inicializar x
    x[j] = 0
for j in range(0,5*N_cor): # termalizar x
    update(x)
for alpha in range(0,N_cf): # Bucle para caminos aleatorios
    for j in range(0,N_cor):
        update(x)
    for n in range(0,N):
        G[alpha][n] = compute_G(x,n)
for n in range(0,N): 
    avg_G[n] = 0
    avg_Gsquare[n] = 0
    for alpha in range(0,N_cf):
        avg_G[n] = avg_G[n] + G[alpha][n]
        avg_Gsquare[n]=avg_Gsquare[n] + (G[alpha][n])**2
    avg_G[n] = avg_G[n]/N_cf  
    avg_Gsquare[n]=avg_Gsquare[n]/N_cf
    errorG[n]=((avg_Gsquare[n]-(avg_G[n])**2)/N_cf)**(1/2)
    print(n*a,avg_G[n],errorG[n]) #print

#Asignar parametros y arrays
errorE=ones(N-1, 'double')
deltaE=ones(N-1, 'double')
t=ones(N-1, 'double')
exact=ones(N-1, 'double')

#Computar la variacion de energia para cada par de puntos.
for n in range(0,N-1):
    deltaE[n]=log(abs(avg_G[n]/avg_G[n+1]))/a
    errorE[n]=(((errorG[n]*avg_G[n+1])/(a*avg_G[n]))**2+
              ((errorG[n+1]*avg_G[n])/(a*avg_G[n+1]))**2)**(1/2)
    t[n]=n*a
    exact[n]=1
    print(n*a,deltaE[n],errorE[n])

#Plot
fig, ax = plt.subplots()
plt.plot(t,exact,c='gray',label='deltaE(inf) = 1')
ax.scatter(t, deltaE, marker ='+', c='blue', label='Energia');
plt.legend(loc='upper right')
plt.xlim(-0.02, 3.2)  
plt.ylim(0, 2)  
plt.xlabel('t')
plt.ylabel('deltaE(t)')
plt.title('Monte Carlo N_cf=10000')

plt.show()