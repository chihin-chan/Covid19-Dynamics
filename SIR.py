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
from scipy.optimize import least_squares

# ODE Function
def sir(t,y):
    
    gamma = 0.013309
    beta =  0.00001774
    
    S = y[0]
    I = y[1]
    R = y[2]
    return [-beta*I*S, beta*I*S - gamma*I, gamma*I]

# Cost function
def cost(x, t, init):
    t_eval = np.arange(0,t,1)
    res = solve_ivp(sir, [0, t], [init[0], init[1], init[2]], t_eval=np.arange(0,t,1))
    
    # Costs/errors
    infect_error = np.zeros(len(x))
    for i in range(0,len(x)):
        infect_error[i] = res.y[1][i] - x[i]


    return infect_error

    


c_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
r_df = pd.read_csv('time_series_covid19_recovered_global.csv')

country = "Switzerland"
start_date = "2/26/20"
start_date_2 = "2/26/2020"

# Parsing confirmed/recovered cases from csv
confirmed = c_df[c_df['Country/Region'] == country].iloc[0].loc[start_date:]
recovered = r_df[r_df['Country/Region'] == country].iloc[0].loc[start_date_2:]
# Truncate last element to fit recovered data
confirmed = confirmed.values[0:-1]  
recovered = recovered.values

# I(t)
infect = confirmed - recovered

# Suceptable = Total Population
N = 20000

# Number of infected at t0
I0 = 2
S0 = N-I0
R0 = 0

# S(t) = N - I(t) - R(t)
S = N - confirmed

# Parameters to optimise
res_lsq = least_squares(cost(infect, len(infect), [S0, I0, R0]), 


plt.plot(res.y[0])
plt.plot(res.y[1])
plt.plot(res.y[2])
plt.plot(infect)
plt.plot(recovered)
plt.legend(('S', 'I', 'R', 'Actual Infected', 'Actual Recovered'))
plt.grid()
plt.show()

