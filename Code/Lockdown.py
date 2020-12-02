import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Total population, N.
N = 10373225
# Initial number of infected and recovered individuals, I0 and R0.
S0 = N-1
E0 = 0 
I0 = 50000
R0 = 0
initial_conditions = S0, E0, I0, R0
# Everyone else, S0, is susceptible to infection initially.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#R_0 = 4
#rho = 0.4
# A grid of time points (in days)

S1 = []
E1 = []
I1 = []
R1 = []
T1 = np.linspace(0,500,500)


def dSdt(S, I, beta, rho):
    return -rho * beta * (S*I/N)
 
def dEdt(S, E, I, alpha, beta, rho):
    return rho * beta * (S*I/N) - alpha*E

def dIdt(E, I, alpha, gamma):
    return alpha * E - gamma * I

def dRdt(I,gamma):
    return gamma * I


# The SIR model differential equations.

def SEIR(initial_conditions, t, N, alpha, beta, gamma, rho):
    S2, E2, I2, R2 = initial_conditions
    dS = dSdt(S2, I2, beta, rho)
    dE = dEdt(S2, E2, I2, alpha, beta, rho)
    dI = dIdt(E2, I2, alpha, gamma)
    dR = dRdt(I2,gamma)
    return dS, dE, dI, dR

def graph(rho, R_0, T):
    gamma = 1/18
    beta = R_0*gamma
    alpha = 1/5.2
    result = odeint(SEIR, initial_conditions, T, args=(N,alpha, beta, gamma, rho))
    S2, E2, I2, R2 = result.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    #fig, ax = plt.subplots(1,1)
    
    #ax.plot(T, [(i/N) for i in S2], label= '$\\rho$ = ' +str(rho))
    ax.plot(T,I2/1000000, label= '$\\rho$ = ' +str(rho))
    plt.title('Infected cases for various $\\rho$', fontsize = 18)
    #plt.title('Susceptibility for various $\\rho$', fontsize = 18)
    ax.set_xlabel('Time (Days)', fontsize = 18)
    ax.set_ylabel('Infected/Millions', fontsize = 18)
    #ax.set_ylabel('Susceptible fraction', fontsize = 18)
    legend = ax.legend(fontsize = 18)
    legend.get_frame().set_alpha(0.5)

rho_list = [0.4, 0.5, 0.6, 0.7, 0.8, 1]
fig, ax = plt.subplots(1,1)
for rho in rho_list:
    graph(rho, 4, T1)
plt.show() 

