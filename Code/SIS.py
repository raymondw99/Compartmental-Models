import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import*
plt.rcParams.update({'font.size': 14})

# Total population, N.
N = 10352390
# Initial number of infected and recovered individuals, I0 and R0.
S0 = N-1
I0 = 2
initial_conditions = S0, I0
# Everyone else, S0, is susceptible to infection initially.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R_0 = 4
gamma = 1/18
beta = R_0*gamma
# A grid of time points (in days)

S1 = []
I1 = []
SA = []
IA = []

# Time vector
T = np.linspace(0, 300, 3000000)

# Differential relations
def dSdt(S, I, beta, gamma):
    return -beta * (S*I)/N + gamma*I

def dIdt(S, I, beta, gamma):
    return beta * (S*I)/N - gamma *I

# Analytical solution
def I_analytical(t):
    I_inf = (1-gamma/beta)*N
    xi = beta - gamma
    V = I_inf/I0 - 1
    return I_inf/(1+V*exp(-xi*t))

# Fourth-order Runge-Kutta algorithm 
def RK4(h, S0, I0, i):
    for day in range(0,len(T)):
        k1S = dSdt(S0, I0, beta, gamma)*h
        k2S = dSdt(S0 + k1S/2, I0 + h/2, beta, gamma)*h
        k3S = dSdt(S0 + k2S/2, I0 + h/2, beta, gamma)*h
        k4S = dSdt(S0 + k3S,   I0 + h,   beta, gamma)*h

        k1I = dIdt(S0, I0, beta, gamma)*h
        k2I = dIdt(S0 + h/2, I0 + k1I/2, beta, gamma)*h
        k3I = dIdt(S0 + h/2, I0 + k2I/2, beta, gamma)*h
        k4I = dIdt(S0 + h,   I0 + k3I,   beta, gamma)*h

        S0 += (1/6)*(k1S + 2*k2S + 2*k3S + k4S)
        I0 += (1/6)*(k1I + 2*k2I + 2*k3I + k4I)
        
        if i == 1: 
            S1.append(S0-N+I_analytical(T[day]))
            I1.append(I0-I_analytical(T[day]))
        else:
            S1.append(S0)
            I1.append(I0)


# The SIS model differential equations.

def SIS(initial_conditions, t, N, beta, gamma):
    S2, I2 = initial_conditions
    dS = dSdt(S2, I2, beta, gamma)
    dI = dIdt(S2, I2, beta, gamma)
    return dS, dI

result = odeint(SIS, initial_conditions, T, args=(N,beta,gamma))
S2, I2 = result.T

def cases():
    RK4(0.1, S0, I0, 0)
    for i in range(0, len(T)):
        IA.append(I_analytical(T[i]))
        SA.append(N-I_analytical(T[i]))
    
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax = plt.subplots(1,1)

    ax.plot(T, S1, 'tab:cyan', label='Susceptible - RK4')
    ax.plot(T, I1, 'tab:red', label='Infected - RK4')
    
    #ax.plot(T, S2, 'tab:cyan', label='Susceptible - odeint')
    #ax.plot(T, I2, 'tab:blue', label='Infected - odeint')

    ax.plot(T, SA, 'tab:blue',label='Susceptible - Analytical')
    ax.plot(T, IA, 'tab:orange', label='Infected - Analytical')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Cases')
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()

def error():   
    S3 = []
    I3 = []
    RK4(0.0001, S0, I0, 1)
    
    for i in range(0, len(T)):
        I3.append(I2[i]-I_analytical(T[i]))
        S3.append(S2[i]-N+I_analytical(T[i]))
        
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax = plt.subplots(1,1)        
    ax.plot(T, [abs(i) for i in S1], 'tab:cyan', label='Susceptible')
    ax.plot(T, [abs(i) for i in I1], 'tab:blue', label='Infected')
    #ax.plot(T, [abs(i) for i in S3], 'tab:cyan', label='Susceptible - odeint')
    #ax.plot(T, [abs(i) for i in I3], 'tab:blue', label='Infected - odeint')
    
    plt.title('Deviation from analytical solution', fontsize = 19)
    ax.set_xlabel('Time (Days)', fontsize = 18)
    ax.set_ylabel('Cases', fontsize = 18)
    legend = ax.legend(loc='center right',fontsize = 17)
    legend.get_frame().set_alpha(0.5)
    plt.show()
    
#cases()
error()

