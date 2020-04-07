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
from scipy.optimize import minimize

# Predict function
def predict(x, infect_data, recovered_data, init, beta, gamma):
    
    # ODE Function
    def sir(t, y):
            
        # Outputs
        S = y[0]
        I = y[1]
        R = y[2]
        
        return [-beta*I*S, beta*I*S - gamma*I, gamma*I]
    
    t_extend = len(x) + 100
    t = np.linspace(0,t_extend, t_extend)
    res = solve_ivp(sir, 
                    [0, t_extend], 
                    [init[0], init[1], init[2]],
                    t_eval=np.arange(0,t_extend,1))
    
    plt.plot(t, res.y[0],"--", linewidth=2)
    plt.plot(t, res.y[1],"--", linewidth=2)
    plt.plot(t, res.y[2],"--", linewidth=2)
    plt.plot(x, infect_data, linewidth=4)
    plt.plot(x, recovered_data, linewidth=4)
    plt.legend(("Susceptibles prediction",
                "Infected prediction",
                "Recovered prediciton", 
                "Infected data", 
                "Recoverd data"))
    
    plt.xlabel("Time since the 1st feb", fontsize=18)
    plt.ylabel("Number of people", fontsize=18)
    plt.title("Predicitons for Switzerland", fontsize=22)
    plt.grid()
    plt.show()
    

# Cost function
def cost(point, infect_data, recovered_data, init):
    
    beta, gamma = point
    
    # ODE Function
    def sir(t, y):
            
        # Outputs
        S = y[0]
        I = y[1]
        R = y[2]
        
        return [-beta*I*S, beta*I*S - gamma*I, gamma*I]
    
    res = solve_ivp(sir, 
                    [0, len(infect_data)], 
                    [init[0], init[1], init[2]],
                    t_eval=np.arange(0,len(infect_data),1))
    # Costs/errors
    # Returns RMSE
    rmse_infect = np.sqrt(np.mean((res.y[1] - infect_data)**2)) 
    rmse_recoverd = np.sqrt(np.mean((res.y[2] - recovered_data)**2)) 
    alpha = 0.1
    print(alpha*rmse_infect + (1-alpha)*rmse_recoverd)
    return alpha*rmse_infect + (1-alpha)*rmse_recoverd

c_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
r_df = pd.read_csv('time_series_covid19_recovered_global.csv')

country = "Switzerland"
start_date = "2/10/20"

# Parsing confirmed/recovered cases from csv
confirmed = c_df[c_df['Country/Region'] == country].iloc[0].loc[start_date:]
recovered = r_df[r_df['Country/Region'] == country].iloc[0].loc[start_date:]
# Truncate last element to fit recovered data
confirmed = confirmed.values 
recovered = recovered.values

# I(t)
infect = confirmed - recovered

# Suceptable
N = 22500

# Number of infected at t0
I0 = 2
S0 = N-I0
R0 = 0

# S(t) = N - I(t) - R(t)
S = N - confirmed

# Parameters to optimise
optimal = minimize(cost, 
                   [0.00001, 0.00001], 
                   args=(infect, recovered, [S0, I0, R0]),
                   method='L-BFGS-B',
                   bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
print(optimal.x)

t = np.linspace(0,len(infect), len(infect))

predict(np.transpose(t), infect, recovered, [S0, I0, R0], optimal.x[0], optimal.x[1])


