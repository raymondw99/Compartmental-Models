import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Total population, N.
N = 10373225
# Initial number of infected and recovered individuals, I0 and R0.
S0 = N-1
E0 = 0 
I0 = 8
R0 = 0
initial_conditions = S0, E0, I0, R0
# Everyone else, S0, is susceptible to infection initially.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R_0 = 4
gamma = 1/18
beta = R_0*gamma
alpha = 1/5.2
# A grid of time points (in days)

S1 = []
E1 = []
I1 = []
R1 = []
T = np.linspace(0,1000,1000)

def dSdt(S, I, beta):
    return -beta * (S*I/N)

def dEdt(S, E, I, alpha, beta):
    return beta * (S*I/N) - alpha*E

def dIdt(E, I, alpha, gamma):
    return alpha * E - gamma * I

def dRdt(I,gamma):
    return gamma * I

# The SIR model differential equations.

def SIR(initial_conditions, t, N, alpha, beta, gamma):
    S2, E2, I2, R2 = initial_conditions
    dS = dSdt(S2, I2, beta)
    dE = dEdt(S2, E2, I2, alpha, beta)
    dI = dIdt(E2, I2, alpha, gamma)
    dR = dRdt(I2,gamma)
    return dS, dE, dI, dR

result = odeint(SIR, initial_conditions, T, args=(N,alpha, beta,gamma))
S2, E2, I2, R2 = result.T

    

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax = plt.subplots(1,1)

ax.plot(S2/1000000, E2/1000000, '--', color = 'm', label='Exposed')
ax.plot(S2/1000000, I2/1000000, '--', color = 'tab:orange', label='Infected')
ax.plot(S2/1000000, R2/1000000, '--', color = 'r', label='Recovered')
plt.title('SEIR - Phase space diagram for $R_0$ = ' + str(R_0),
          fontsize = 18)
ax.set_xlabel('Susceptible (Millions)', fontsize = 18)
ax.set_ylabel('Cases (Millions)', fontsize = 18)
legend = ax.legend(fontsize = 18)
legend.get_frame().set_alpha(0.5)

plt.show()
