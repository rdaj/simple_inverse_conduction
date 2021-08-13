from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def extendvector(F):
    """ makes the data periodic"""
    N = F.shape[0]
    x = np.arange(N)/N
    width = min(int(N/2),6)
    

    ca = np.vstack((x[:width]**0, x[:width]**1))
    ca = np.linalg.lstsq(ca.T,F[:width], rcond= None)[0]

    
    cb = np.vstack((x[N-width:]**0, x[N-width:]**1))
    cb = np.linalg.lstsq(cb.T,F[N-width:], rcond= None)[0]
    
    xval = np.hstack([x[:width],x[N-width:]])
    
    fb = cb[0]+cb[1]*(1+x[:width])
    fa = ca[0]+ca[1]*(-x[width:0:-1])
    fval = np.hstack([fb,fa])
    
    F2=interp1d(xval,fval, kind='cubic')
    return np.hstack([F2(x),F])

def sheSolv(H,G,a_fun=1,b_fun=1,xi_c:int=85,tol=1e-4):
    n = G.shape[0]
    xi_c=int(min(n,np.floor(xi_c)))

    H = extendvector(H)
    G = extendvector(G)
    # temperature, dTdx
    V = np.concatenate((G,-a_fun *H))

    points = np.linspace(0,1,9)
    tspan = [0,1]

    VV = solve_ivp(shesyst_ivp, t_span = tspan, y0=V,method='RK45', rtol = tol,max_step = 1./4.,first_step =0.0001,dense_output= True , args=(a_fun,b_fun,xi_c))
    ##[temp0:2n, dTdx2n:4]
    #
    # Extract  the temperature T(x,t) and the heat-flux a(x,T)T(x,t)
    #3
    T=VV.y[n:2*n]

    aTx=-VV.y[:]

    return T, aTx


def shesyst_ivp(points, V, a_fun,b_fun, xi_c):
    n=int(0.5*V.shape[0])
    A=a_fun
    B=b_fun
    ## temp, dTdx
    T = V[:n]
    dTdx = V[n:]

    W=np.fft.fft(T);

    k =np.pi * np.fft.fftfreq(n, d = 1/n)
    k[abs(k) > xi_c] = 0


    W=np.real(np.fft.ifft(1j*k*W))
    # dTempx, dTempt
    dV = np.concatenate((dTdx/A,W*B))
    return dV

from scipy.linalg import lu_factor, lu_solve
def forwardSolve(Ta, Tb,N = 10, a = 1,ic = 0):
    M = Ta.shape[0]

    dt = 1/(M-1)
    dx = 1/(N-1)



    r = dt / dx**2
    avec = np.ones(N-2)*a


    # A = -r , 2 +2r, -r for dirichlett boudary
    # B =  r , 2 -2r,  r
    A = -r*avec*np.eye(N-2,k=-1) + (2+2*r*avec)*np.eye(N-2,k=0) - r * avec*np.eye(N-2,k=1)
    B = +r*avec*np.eye(N-2,k=-1) + (2-2*r*avec)*np.eye(N-2,k=0) + r * avec*np.eye(N-2,k=1)


    b = np.ones(N-2)*0
    U = np.ones([N-2,M])*ic

    da = np.zeros(M)
    da2 = np.zeros(M)
    da3 = np.zeros(M)
    db = np.zeros(M)
    db2 = np.zeros(M)
    db3 = np.zeros(M)
    Ta[0:2] = 0
    Tb[0:2] = 0
    w = [0,1,3,7]

    for j in np.arange(M-1)+1:
        UU = U[:,j-1]
        Utmp = B.dot(UU)

        b[0]  = Ta[j-1] * r * avec[1]
        b[-1] = Tb[j-1] * r * avec[-1]

        b[0]  = b[0]  + Ta[j]   * r * avec[1]
        b[-1] = b[-1] + Tb[j]   * r * avec[-1]


        Utmp = Utmp + b

        lu, piv = lu_factor(A)

        U[:,j] = lu_solve((lu, piv), Utmp)

        da[j] = (-147.*Ta[j]+360.*U[0,j]-450.* U[1,j]+400.* U[2,j]-225.* U[3,j]+72.*  U[4,j]-10.*  U[5,j])/(60.*1.0*dx**1)

        db[j] = (10.  *U[-6,j]-72.*U[-5,j]+225.*U[-4,j]-400.*U[-3,j]+450.*U[-2,j]-360.*U[-1,j]+147.*Tb[j])/(60.*1.0*dx**1)

    return U, da ,db
