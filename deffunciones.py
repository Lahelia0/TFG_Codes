import matplotlib
import numpy 
from matplotlib import pyplot as plt
import math
import cmath
import random
from numpy import *
from scipy.optimize import least_squares
import itertools

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

