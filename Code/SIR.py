import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# Total population, N.
N = 10373225
# Initial number of infected and recovered individuals, I0 and R0.
p = 0.4 # Vaccination rate
S0 = N-1 #(1-p)*(N-1) #N-1
R0 = 0 #p*(N-1)#0
I0 = 8 #N - S0 - R0
initial_conditions = S0, I0, R0
# Everyone else, S0, is suI0 = 4000000 #8sceptible to infection initially.
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
R_0 = 2
gamma = 1/18
beta = R_0*gamma
# A grid of time points (in days)

S1 = []
I1 = []
R1 = []
T = np.linspace(0,500,500)

# Differential relations
def dSdt(S, I, beta):
    return -beta * (S*I/N)

def dIdt(S, I, beta, gamma):
    return beta * (S*I/N) - gamma * I

def dRdt(I,gamma):
    return gamma * I

# Fourth-order Runge-Kutta algorithm
def RK4(h, days, S0, I0, R0):
    for day in range(0,days):
        k1S = dSdt(S0, I0, beta)*h
        k2S = dSdt(S0 + k1S/2, I0, beta)*h
        k3S = dSdt(S0 + k2S/2, I0, beta)*h
        k4S = dSdt(S0 + k3S, I0, beta)*h

        k1I = dIdt(S0, I0, beta, gamma)*h
        k2I = dIdt(S0, I0 + k1I/2, beta, gamma)*h
        k3I = dIdt(S0, I0 + k2I/2, beta, gamma)*h
        k4I = dIdt(S0, I0 + k3I, beta, gamma)*h

        k1R = dRdt(I0, gamma)*h
        k2R = dRdt(I0+k1R, gamma)*h
        k3R = dRdt(I0+k2R, gamma)*h
        k4R = dRdt(I0+k3R, gamma)*h

        S0 += (1/6)*(k1S + 2*k2S + 2*k3S + k4S)
        I0 += (1/6)*(k1I + 2*k2I + 2*k3I + k4I)
        R0 += (1/6)*(k1R + 2*k2R + 2*k3R + k4R)

        S1.append(S0)
        I1.append(I0)
        R1.append(R0)


# The SIR model differential equations.

def SIR(initial_conditions, t, N, beta, gamma):
    S2, I2, R2 = initial_conditions
    dS = dSdt(S2, I2, beta)
    dI = dIdt(S2, I2, beta, gamma)
    dR = dRdt(I2,gamma)
    return dS, dI, dR

result = odeint(SIR, initial_conditions, T, args=(N,beta,gamma))
S2, I2, R2 = result.T

    
RK4(1, 500, S0, I0, R0)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax = plt.subplots(1,1)

#ax.plot(T, S1, label='Susceptible - RK4')
#ax.plot(T, I1, label='Infected - RK4')
#ax.plot(T, R1, label='Recovered - RK4')

ax.plot(T, S2/1000000, label='Susceptible')
ax.plot(T, I2/1000000, label='Infected')
ax.plot(T, R2/1000000, label='Recovered')
#plt.title('Vaccination rate p = ' + str(p) + ' - $R_0$ = ' + str(R_0),
#          fontsize = 20)
plt.title('SIR-model for COVID-19 - $R_0$ = ' + str(R_0),
          fontsize = 20)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
ax.set_xlabel('Time (Days)', fontsize = 18)
ax.set_ylabel('Cases (Millions)', fontsize = 18)
legend = ax.legend(fontsize = 18)
legend.get_frame().set_alpha(0.5)

plt.show()
