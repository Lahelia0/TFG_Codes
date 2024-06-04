import itertools
from matplotlib import pyplot as plt
import math
import cmath
import random
from numpy import *
from scipy.optimize import least_squares

#matriz compuesta conjugada
def dagger(M):
    N=len(M)
    H=zeros((N,N),dtype=complex)
    R=matrix(M)
    H=R.getH()
    return H.copy()

#Calculo regresion
#input:-x:parametro
#      -r:variable radial
#      -y:ordenada
def exacto(x,r,y):
    return x[0]*r-x[1]/r+x[2]-y

#potencial estatico
#input:-r:radio
#      -a:string tension
#      -b:parametro coulombiano
#      -c:constante
def data(r,a,b,c):
    return a*r-b/r+c

#Generador de Matrices de SU(3)
#input:-M:Matrix array
#output:-M:Matrix array con matrices SU(3)
#innner parameters: -Nmatrix: matrices que vamos a generar
#                   -eps: parametro
def randommatrixSU3(M):
    identity=eye(3)
    Nmatrix=100
    M=zeros((200,3,3),dtype=complex)
    H=zeros((3,3),dtype=complex)
    eps=0.24
    w=cmath.sqrt(-1)
    for s in range(0,Nmatrix):
        for j in range(0,3):
            for i in range(0,3):
                H[j,i]=complex(random.uniform(-1,1),
                random.uniform(-1,1))
        H=(H.copy()+dagger(H.copy()))/2. #matriz hermitica
        for n in range(30): #Taylor
            M[s]=M[s]+(w*eps)**n/math.factorial(n)*linalg.matrix_power(H,n)
        M[s]=M[s]/linalg.det(M[s])**(1/3) 

        M[s+Nmatrix]=dagger(M[s]) #inversa
    return M.copy()


#Actualizar la posicion en funcion de la accion de wilson para QCD, 
#usando un algoritmo de tipo metropolis.
#input:-U:array de las variables de enlace
#      -M:matrices de SU(3)
#inner parameters: -N:numero de puntos en la lattice
def update(U,M):
    Nmatrix=100
    N=8
    beta=5.5
    beta_imp=1.719
    u0=0.797
    improved=True #true = usar gamma improved, false = usar gamma
    for x, y, z, t in itertools.product(range(N), repeat=4):
        for mi in range(0,4):
            gamma=Gamma(U,mi,x,y,z,t) 
            if improved: 
                    gamma_imp=Gamma_improved(U,mi,x,y,z,t) 
            for p in range(10): #numero de iteraciones por punto
                s=random.randint(2,2*Nmatrix) #elegir matriz
                if improved: 
                    dS = -beta_imp/(3)*(5/(3*u0**4)*
                    real(trace(dot((dot(M[s],U[x,y,z,t,mi])
                    -U[x,y,z,t,mi]),gamma)))-1/(12*u0**6)*
                    real(trace(dot((dot(M[s],U[x,y,z,t,mi])
                    -U[x,y,z,t,mi]),gamma_imp)))) 
                else:
                    dS = -beta/(3)*real(trace(dot((dot(M[s].copy(),
                    U[x,y,z,t,mi].copy())
                    -U[x,y,z,t,mi].copy()),gamma.copy()))) 
                    if dS<0 or exp(-dS)>random.uniform(0,1):
                        U[x,y,z,t,mi] = dot(M[s].copy(),
                        U[x,y,z,t,mi].copy())  # update U

#Computar Gamma para una acción de Wilson
#input:-x,y,z,t: posiciones
#      -U:array de las variables de enlace
#      -ro:direccion de computo
#inner parameter:-N:numero de puntos en la lattice
def Gamma(U,ro,x,y,z,t):
    N=8
    gamma=0. #inicializar gamma
    inc_ro=zeros((4),'int')
    inc_ni=zeros((4),'int')
    inc_ro[ro]=1 
    #siguiente punto en mi
    xp_ro=(x+inc_ro[0].copy())%N
    yp_ro=(y+inc_ro[1].copy())%N
    zp_ro=(z+inc_ro[2].copy())%N
    tp_ro=(t+inc_ro[3].copy())%N
    for ni in range(0,4):
        if ni!=ro :
            inc_ni[ni]=1 
            #siguiente punto de ni
            xp_ni=(x+inc_ni[0].copy())%N
            yp_ni=(y+inc_ni[1].copy())%N
            zp_ni=(z+inc_ni[2].copy())%N
            tp_ni=(t+inc_ni[3].copy())%N
            #anterior punto de ni
            xm_ni=(x-inc_ni[0].copy())%N
            ym_ni=(y-inc_ni[1].copy())%N
            zm_ni=(z-inc_ni[2].copy())%N
            tm_ni=(t-inc_ni[3].copy())%N
            #siguiente punto de ni y mi
            xp_ro_p_ni=(x+inc_ro[0].copy()+inc_ni[0].copy())%N
            yp_ro_p_ni=(y+inc_ro[1].copy()+inc_ni[1].copy())%N
            zp_ro_p_ni=(z+inc_ro[2].copy()+inc_ni[2].copy())%N
            tp_ro_p_ni=(t+inc_ro[3].copy()+inc_ni[3].copy())%N
            #siguiente de mi y anterior de ni
            xp_ro_m_ni=(x+inc_ro[0].copy()-inc_ni[0].copy())%N
            yp_ro_m_ni=(y+inc_ro[1].copy()-inc_ni[1].copy())%N
            zp_ro_m_ni=(z+inc_ro[2].copy()-inc_ni[2].copy())%N
            tp_ro_m_ni=(t+inc_ro[3].copy()-inc_ni[3].copy())%N
            inc_ni[ni]=0
            gamma=gamma+dot(dot(U[xp_ro,yp_ro,zp_ro,tp_ro,ni],dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,ro]))
                        ,dagger(U[x,y,z,t,ni])) \
                        +dot(dot(dagger(U[xp_ro_m_ni,yp_ro_m_ni,zp_ro_m_ni,tp_ro_m_ni,ni]),
                         dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,ro])),U[xm_ni,ym_ni,zm_ni,tm_ni,ni])

    return gamma.copy()

#Computar Gamma para una acción de Wilson mejorada
#input:-x,y,z,t: posiciones
#      -U:array de las variables de enlace
#inner parameter:-N:numero de puntos en la lattice
def Gamma_improved(U,mi,x,y,z,t):
    N=8
    gamma_imp=0. #inicializar gamma
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    incmi[mi]=1 
    #siguiente punto en mi
    xp_mi=(x+incmi[0].copy())%N
    yp_mi=(y+incmi[1].copy())%N
    zp_mi=(z+incmi[2].copy())%N
    tp_mi=(t+incmi[3].copy())%N
    #anterior punto de mi
    xm_mi=(x-incmi[0].copy())%N
    ym_mi=(y-incmi[1].copy())%N
    zm_mi=(z-incmi[2].copy())%N
    tm_mi=(t-incmi[3].copy())%N
    #siguiente punto de 2mi
    xp_2mi=(x+2*incmi[0].copy())%N
    yp_2mi=(y+2*incmi[1].copy())%N
    zp_2mi=(z+2*incmi[2].copy())%N
    tp_2mi=(t+2*incmi[3].copy())%N

    for ni in range(0,4):
        if ni!=mi :
            incni[ni]=1
            #siguiente punto de ni
            xp_ni=(x+incni[0].copy())%N
            yp_ni=(y+incni[1].copy())%N
            zp_ni=(z+incni[2].copy())%N
            tp_ni=(t+incni[3].copy())%N
            #anterior punto de ni
            xm_ni=(x-incni[0].copy())%N
            ym_ni=(y-incni[1].copy())%N
            zm_ni=(z-incni[2].copy())%N
            tm_ni=(t-incni[3].copy())%N
            #siguiente punto de ni and mi
            xp_mi_p_ni=(x+incmi[0].copy()+incni[0].copy())%N
            yp_mi_p_ni=(y+incmi[1].copy()+incni[1].copy())%N
            zp_mi_p_ni=(z+incmi[2].copy()+incni[2].copy())%N
            tp_mi_p_ni=(t+incmi[3].copy()+incni[3].copy())%N
            #siguiente punto de mi y anterior punto de ni
            xp_mi_m_ni=(x+incmi[0].copy()-incni[0].copy())%N
            yp_mi_m_ni=(y+incmi[1].copy()-incni[1].copy())%N
            zp_mi_m_ni=(z+incmi[2].copy()-incni[2].copy())%N
            tp_mi_m_ni=(t+incmi[3].copy()-incni[3].copy())%N
            #siguiente punto de 2ni
            xp_2ni=(x+2*incni[0].copy())%N
            yp_2ni=(y+2*incni[1].copy())%N
            zp_2ni=(z+2*incni[2].copy())%N
            tp_2ni=(t+2*incni[3].copy())%N
            #anterior punto de 2ni
            xm_2ni=(x-2*incni[0].copy())%N
            ym_2ni=(y-2*incni[1].copy())%N
            zm_2ni=(z-2*incni[2].copy())%N
            tm_2ni=(t-2*incni[3].copy())%N
            #siguiente punto de 2mi y anterior punto de ni
            xp_2mi_m_ni=(x+2*incmi[0].copy()-incni[0].copy())%N
            yp_2mi_m_ni=(y+2*incmi[1].copy()-incni[1].copy())%N
            zp_2mi_m_ni=(z+2*incmi[2].copy()-incni[2].copy())%N
            tp_2mi_m_ni=(t+2*incmi[3].copy()-incni[3].copy())%N
            #siguiente punto de mi y anterior punto de 2ni
            xp_mi_m_2ni=(x+incmi[0].copy()-2*incni[0].copy())%N
            yp_mi_m_2ni=(y+incmi[1].copy()-2*incni[1].copy())%N
            zp_mi_m_2ni=(z+incmi[2].copy()-2*incni[2].copy())%N
            tp_mi_m_2ni=(t+incmi[3].copy()-2*incni[3].copy())%N
            #anterior punto de ni y mi
            xm_mi_m_ni=(x-incmi[0].copy()-incni[0].copy())%N
            ym_mi_m_ni=(y-incmi[1].copy()-incni[1].copy())%N
            zm_mi_m_ni=(z-incmi[2].copy()-incni[2].copy())%N
            tm_mi_m_ni=(t-incmi[3].copy()-incni[3].copy())%N
            #siguiente punto de ni y anterior punto de mi
            xm_mi_p_ni=(x-incmi[0].copy()+incni[0].copy())%N
            ym_mi_p_ni=(y-incmi[1].copy()+incni[1].copy())%N
            zm_mi_p_ni=(z-incmi[2].copy()+incni[2].copy())%N
            tm_mi_p_ni=(t-incmi[3].copy()+incni[3].copy())%N
            incni[ni]=0

            gamma_imp=gamma_imp+dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,mi],
                                    U[xp_2mi,yp_2mi,zp_2mi,tp_2mi,ni]),
                                    dot(dagger(U[xp_mi_p_ni,yp_mi_p_ni,zp_mi_p_ni,tp_mi_p_ni,mi]),
                                    dot(dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi]),dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,mi],
                                    dagger(U[xp_2mi_m_ni,yp_2mi_m_ni,zp_2mi_m_ni,tp_2mi_m_ni,ni])),
                                    dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,mi]),
                                    dot(dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,mi]),U[xm_ni,ym_ni,zm_ni,tm_ni,ni]))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],U[xp_mi_p_ni,yp_mi_p_ni,zp_mi_p_ni,tp_mi_p_ni,ni]),
                                    dot(dagger(U[xp_2ni,yp_2ni,zp_2ni,tp_2ni,mi]),
                                    dot(dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,ni]),
                                    dagger(U[x,y,z,t,ni])))) \
                                +dot(dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,ni]),
                                    dagger(U[xp_mi_m_2ni,yp_mi_m_2ni,zp_mi_m_2ni,tp_mi_m_2ni,ni])),
                                    dot(dagger(U[xm_2ni,ym_2ni,zm_2ni,tm_2ni,mi]),
                                    dot(U[xm_2ni,ym_2ni,zm_2ni,tm_2ni,ni],U[xm_ni,ym_ni,zm_ni,tm_ni,ni]))) \
                                +dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi])),
                                    dot(dagger(U[xm_mi_p_ni,ym_mi_p_ni,zm_mi_p_ni,tm_mi_p_ni,mi]),
                                    dot(dagger(U[xm_mi,ym_mi,zm_mi,tm_mi,ni]),U[xm_mi,ym_mi,zm_mi,tm_mi,mi]))) \
                                +dot(dot(dagger(U[xp_mi_m_ni,yp_mi_m_ni,zp_mi_m_ni,tp_mi_m_ni,ni]),
                                    dagger(U[xm_ni,ym_ni,zm_ni,tm_ni,mi])),
                                    dot(dagger(U[xm_mi_m_ni,ym_mi_m_ni,zm_mi_m_ni,tm_mi_m_ni,mi]),
                                    dot(U[xm_mi_m_ni,ym_mi_m_ni,zm_mi_m_ni,tm_mi_m_ni,ni],U[xm_mi,ym_mi,zm_mi,tm_mi,mi])))
    return gamma_imp.copy()

#Computar el lado del Wilson Loop
#input:-x,y,z,t: posiciones
#      -U:array de las variables de enlace
#      -f:direccion
#inner parameter:-N:numero de puntos de la lattice
def ProductU(U,x,y,z,t,n,f):
    N=8
    I=eye(3)
    productU=zeros((3,3),dtype=complex) #crear el array productU
    productU=I.copy()  #inicializar ProductU
    inc=zeros((4),'int')
    inc[f]=1 #incre. en la direccion f 
    for i in range(n):
        x_inc=(x+i*inc[0])%N #incre. en dicha direccion
        y_inc=(y+i*inc[1])%N #incre. en dicha direccion
        z_inc=(z+i*inc[2])%N #incre. en dicha direccion
        t_inc=(t+i*inc[3])%N #incre. en dicha direccion
        productU=dot(productU,U[x_inc,y_inc,z_inc,t_inc,f]) 
    return productU.copy()

#Computar la inversa del lado del Wilson Loop
#input:-x,y,z,t: posiciones
#      -U:array de las variables de enlace
#      -f:direccion
#inner parameter:-N:numero de puntos de la lattice
def ProductUdagger(U,x,y,z,t,n,f):
    N=8
    I=eye(3)
    productUdagger=zeros((3,3),dtype=complex) #crear el array 
    productUdagger=I.copy() #inicializar ProductUdagger
    inc=zeros((4),'int')
    inc[f]=1 #incre. en la direccion f 
    for i in range(n):
        x_inc=(x-(i+1)*inc[0])%N #decre. en dicha direccion
        y_inc=(y-(i+1)*inc[1])%N #decre. en dicha direccion
        z_inc=(z-(i+1)*inc[2])%N #decre. en dicha direccion
        t_inc=(t-(i+1)*inc[3])%N #decre. en dicha direccion
        productUdagger=dot(productUdagger,
            dagger(U[x_inc,y_inc,z_inc,t_inc,f])) 
    return productUdagger.copy()

#Computar el Wilson loop a x a para cada punto, haciendo uso de U.
#Usando algoritmo del tipo Metropolis
#input:-U: variables de enlace
#      -x,y,z,t:posiciones
#inner parameters:-N:numero de puntos en la lattice
def compute_WL(U,x,y,z,t):
    N=8
    WL = 0.
    incmi=zeros((4),'int')
    incni=zeros((4),'int')
    for mi in range(0,4):
        incmi[mi]=1 
        #siguiente punto de mi
        xp_mi=(x+incmi[0].copy())%N
        yp_mi=(y+incmi[1].copy())%N
        zp_mi=(z+incmi[2].copy())%N
        tp_mi=(t+incmi[3].copy())%N
        for ni in range(0,mi):
            incni[ni]=1 
            #siguiente punto de ni
            xp_ni=(x+incni[0].copy())%N
            yp_ni=(y+incni[1].copy())%N
            zp_ni=(z+incni[2].copy())%N
            tp_ni=(t+incni[3].copy())%N
            incni[ni]=0

            WL=WL+trace(dot(U[x,y,z,t,mi],
                        dot(dot(U[xp_mi,yp_mi,zp_mi,tp_mi,ni],
                        dagger(U[xp_ni,yp_ni,zp_ni,tp_ni,mi])),
                        dagger(U[x,y,z,t,ni]))))
        incmi[mi]=0
    return real(WL)/(3.*6.)

#Smearing
#input:-U: array de las variables de enlace
#inner parameters:-eps: termino de substitucion
#                 -a: lattice spacing
#output:-U: U "smeared"
def SubstitutionU(U):
    eps=1./12.
    a=0.25
    N=8
    sec_der=zeros((N,N,N,N,4,3,3),dtype=complex) 
    sec_der=Gauge_cov_deriv(U) #derivada de segundo orden

    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    for mi in range(0,3):
                        U[x,y,z,t,mi]=U[x,y,z,t,mi].copy()
                        +eps*a**2*sec_der[x,y,z,t,mi].copy()
    return U

#computar la derivada del gauge covariante para todos los puntos y direcciones.
#input:-U:variables de enlace
#inner parameters:-u0: valor medio de U (Tadpole improvement)
#                 -a: lattice spacing
#                 -N: numero de puntos de la lattice
#output:-Gauge_der: array 
def Gauge_cov_deriv(U):
    u0=0.797
    a=0.25
    N=8
    Gauge_der=zeros((N,N,N,N,4,3,3),dtype=complex)
    incmi=zeros((4),'int')
    incro=zeros((4),'int')

    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    for mi in range(0,4):
                        incmi[mi]=1 
                        #siguiente punto de mi
                        xp_mi=(x+incmi[0].copy())%N
                        yp_mi=(y+incmi[1].copy())%N
                        zp_mi=(z+incmi[2].copy())%N
                        tp_mi=(t+incmi[3].copy())%N
                        for ro in range(0,4):
                            incro[ro]=1 
                            #siguiente punto de ro
                            xp_ro=(x+incro[0].copy())%N
                            yp_ro=(y+incro[1].copy())%N
                            zp_ro=(z+incro[2].copy())%N
                            tp_ro=(t+incro[3].copy())%N
                            #anterior punto de ro
                            xm_ro=(x-incro[0].copy())%N
                            ym_ro=(y-incro[1].copy())%N
                            zm_ro=(z-incro[2].copy())%N
                            tm_ro=(t-incro[3].copy())%N
                            #siguiente punto de mi y anterior de ni
                            xp_mi_m_ro=(x+incmi[0].copy()-incro[0].copy())%N
                            yp_mi_m_ro=(y+incmi[1].copy()-incro[1].copy())%N
                            zp_mi_m_ro=(z+incmi[2].copy()-incro[2].copy())%N
                            tp_mi_m_ro=(t+incmi[3].copy()-incro[3].copy())%N
                            incro[ro]=0
                            Gauge_der[x,y,z,t,mi]=Gauge_der[x,y,z,t,mi]+1./(u0**2*a**2)*(dot(dot(U[x,y,z,t,ro],U[xp_ro,yp_ro,zp_ro,tp_ro,mi]),dagger(U[xp_mi,yp_mi,zp_mi,tp_mi,ro]))) \
                                        -2./(a**2)*(U[x,y,z,t,mi].copy()) \
                                        +1./(u0**2*a**2)*(dot(dot(dagger(U[xm_ro,ym_ro,zm_ro,tm_ro,ro]),U[xm_ro,ym_ro,zm_ro,tm_ro,mi]),U[xp_mi_m_ro,yp_mi_m_ro,zp_mi_m_ro,tp_mi_m_ro,ro]))
                        incmi[mi]=0
    return Gauge_der.copy()

#Programa principal
#asignacion de arrays y parametros
a=0.25 
N=8 
N_sim_r=4 
N_sim_t=4 
Nmatrix=200 #Numero de matrices de SU3
N_cf=10 
N_cor=10 
U=zeros((N,N,N,N,4,3,3),dtype=complex) 
F=zeros((N,N,N,N,4,3,3),dtype=complex)
WL=zeros((N_cf,N_sim_t,N_sim_r), 'double') 
M=zeros((Nmatrix,3,3),dtype=complex)
avg_WL=zeros((N_sim_t,N_sim_r),'double') 
avg_WLSQ=zeros((N_sim_t,N_sim_r),'double') 
err_avg_WL=zeros((N_sim_t,N_sim_r),'double') 
potential=zeros(N_sim_r-1) 
err_potential=zeros(N_sim_r-1) 
rad=zeros(N_sim_r-1) 

# Inicializar U con la identidad, esta pertenece a SU3
for x in range(0,N):
    for y in range(0,N):
        for z in range(0,N):
            for t in range(0,N):
                for mi in range(0,4):
                    for n in range(0,3):
                        U[x,y,z,t,mi,n,n]=1.

# Generar matrices SU3
M=randommatrixSU3(M)
#Computar el MC value para el propagador de toda t
for j in range(0,2*N_cor): # termalizar
    update(U,M)
print('Termalized')
for alpha in range(0,N_cf): # loop de caminos aleatorios
    for j in range(0,N_cor):
        update(U,M)
    F=U.copy()
    F=SubstitutionU(F) #Smearing
    for time in range(1,N_sim_t):
        for radius in range(1,N_sim_r):
            WL[alpha,time,radius]=compute_WL(F,time,radius) #compute WL

    print("End simulation number:",alpha+1)

for time in range(1,N_sim_t): #cComputar el wilson loop para todos los valores de r y t
    for radius in range(1,N_sim_r):
        avg_WL[time,radius]=0.
        avg_WLSQ[time,radius]=0.
        for alpha in range(0,N_cf): # hacer la media
            avg_WL[time,radius]= avg_WL[time,radius]+WL[alpha,time,radius]
            avg_WLSQ[time,radius]=avg_WLSQ[time,radius]+WL[alpha,time,radius]**2
        avg_WL[time,radius] = avg_WL[time,radius]/N_cf  #media de todos los valores
        avg_WLSQ[time,radius] = avg_WLSQ[time,radius]/N_cf #media del cuadrado
        err_avg_WL[time,radius]=((avg_WLSQ[time,radius]-avg_WL[time,radius]**2)/N_cf)**(1/2) #error estadistico

for i in range(0,N_sim_r-1): 
    radius=i+1
    potential[i]=abs(avg_WL[N_sim_t-2,radius]/avg_WL[N_sim_t-1,radius]) #computo del potencial
    err_potential[i]=((err_avg_WL[N_sim_t-1,radius]/avg_WL[N_sim_t-1,radius])**2+(err_avg_WL[N_sim_t-2,radius]/avg_WL[N_sim_t-2,radius])**2)**(1/2)*potential[i] 
    rad[i]=radius


x=ones(3) 
res_lsq = least_squares(exacto, x, args=(rad, potential)) 
rad_test = linspace(0, 4.5,100) 
V_lsq = data(rad_test, *res_lsq.x) 
xs = linspace(1, 4.5,3) 


print ('ptential', potential) 
print ('V_lsq', V_lsq) 

#Copiar los resultados obtenidos en Graficar.py