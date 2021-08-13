import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from inverse_functions import *
# read the data
df = pd.read_csv('simulated/heat_top_hatdataPoints.csv')
n = df.shape[0]
samp_time = 0.001
T0=n*samp_time
N2 = 2048;

T_INITA = df.iloc[:2,:].mean().mean()
t=np.linspace(0,T0,n);
tt= np.linspace(0,T0,N2);

df2 = interp1d(t,df.values-T_INITA,kind = 'linear', axis = 0 )
df22 = pd.DataFrame(df2(tt), columns = df.columns)
T1 =df22['heat_top_hat/middle'].values
T1 =df22['heat_top_hat/measure'].values
T2 = df22['heat_top_hat/1mmAbove']

cool = df22['heat_top_hat/cooled'].values
L1 = 1*1e-3
L0 = L1 + L1*0.1

k = 44.5
rho = 7850.
cp = 475.


tx=T0/(L1-L0)**2

#thermal diffusion times tota time/ total length **2
kappa = k /(cp*rho) * tx
U, da ,db = forwardSolve(T1, T2,N = 32,a = kappa);
da2 = -da*L0 / (L1-L0)
db2 = -db*L0 / (L1-L0)

tol=1e-4
xi_c=85

a_fun=k*T0/(L0)**2
b_fun=cp*rho
#scalef = T0/(L0)**2
cT1,aTx1=sheSolv(da2,T2,a_fun,b_fun,xi_c,tol);
kTx1 = aTx1*(L0)**2/T0/L0

plt.figure(figsize = [8,8])
plt.plot(T1+T_INITA, label = 'T1')
p = int(aTx1.shape[0]/4)
plt.plot(T2+T_INITA, label = 'T2')
plt.plot(cool+T_INITA, label = 'cool from data', linewidth=1.0)
#plt.plot(cT1[:, -1]+T_INITA, label = 'calculated', linewidth=7.0, alpha = 0.5)
plt.plot(-aTx1[p:2*p,-1]+T_INITA,'--', label = 'cool calculated')
plt.legend()
plt.show()
