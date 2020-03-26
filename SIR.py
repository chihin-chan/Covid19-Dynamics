###############################################################################
#
# SIR Model for Covid-19
# Data from John-Hopkins University
#
# By Chi Hin, Jean Phillipe
#
###############################################################################

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ODE Function
def sir(t,y):
    
    gamma = 0.08023309
    beta =  0.00003974
    
    S = y[0]
    I = y[1]
    R = y[2]
    return [-beta*I*S, beta*I*S - gamma*I, gamma*I]


c_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
r_df = pd.read_csv('time_series_covid19_recovered_global.csv')

country = "Switzerland"
start_date = "2/26/20"
start_date_2 = "2/26/2020"

confirmed = c_df[c_df['Country/Region'] == country].iloc[0].loc[start_date:]
recovered = r_df[r_df['Country/Region'] == country].iloc[0].loc[start_date_2:]
confirmed = confirmed.values[0:-1]
recovered = recovered.values


# I(t)
infect = confirmed - recovered

# Suceptable = Total Population
N = 15000

# Number of infected at t0
I0 = 2
S0 = N-I0
R0 = 0

# S(t) = N - I(t) - R(t)
S = N - confirmed

# Parameters to optimise
size = 100 # No. of days
t_eval = np.arange(0,size,1)
res = solve_ivp(sir, [0, 100], [S0,I0,R0], t_eval=np.arange(0,size,1))

plt.plot(res.y[0])
plt.plot(res.y[1])
plt.plot(res.y[2])
plt.legend(('S', 'I', 'R'))
plt.show()

