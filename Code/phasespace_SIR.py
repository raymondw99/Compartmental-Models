import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# Total population, N.
N = 10373225
# Initial number of infected and recovered individuals, I0 and R0.
S0 = N-1
I0 = 8
R0 = 0
initial_conditions = S0, I0, R0
# Everyone else, S0, is susceptible to infection initially.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R_0 = 4
gamma = 1/18
beta = R_0*gamma
# A grid of time points (in days)

S1 = []
I1 = []
R1 = []
# Time vector
T = np.linspace(0,500,500)

# Differential relations
def dSdt(S, I, beta):
    return -beta * (S*I/N)

def dIdt(S, I, beta, gamma):
    return beta * (S*I/N) - gamma * I

def dRdt(I,gamma):
    return gamma * I


# The SIR model differential equations.
def SIR(initial_conditions, t, N, beta, gamma):
    S2, I2, R2 = initial_conditions
    dS = dSdt(S2, I2, beta)
    dI = dIdt(S2, I2, beta, gamma)
    dR = dRdt(I2,gamma)
    return dS, dI, dR

result = odeint(SIR, initial_conditions, T, args=(N,beta,gamma))
S2, I2, R2 = result.T

    
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax = plt.subplots(1,1)

ax.plot(S2/1000000, I2/1000000, '--', color = 'b', label='Infected')
ax.plot(S2/1000000, R2/1000000, '--', color = 'tab:cyan', label='Recovered')
plt.title('SIR - Phase space diagram for $R_0$ = ' + str(R_0),
          fontsize = 18)
ax.set_xlabel('Susceptible (Millions)', fontsize = 18)
ax.set_ylabel('Cases (Millions)', fontsize = 18)
legend = ax.legend(fontsize = 18)
legend.get_frame().set_alpha(0.5)

plt.show()
