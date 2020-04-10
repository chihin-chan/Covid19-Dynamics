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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Predict function
def predict(x, infect_data, recovered_data, init, beta, gamma, start, country):
    
    # ODE Function
    def sir(t, y):
            
        # Outputs
        S = y[0]
        I = y[1]
        R = y[2]
        
        return [-beta*I*S, beta*I*S - gamma*I, gamma*I]
    
    # Computing ODE to +100 days
    t_extend = len(x) + 365
    t = np.linspace(0, t_extend, t_extend)
    
    # Solving SIR
    res = solve_ivp(sir, 
                    [0, t_extend], 
                    [init[0], init[1], init[2]],
                    t_eval=np.arange(0,t_extend,1))
    
    # Preparing date axis
    date_format = '%m/%d/%y'
    time = start
    for i in range(t_extend-1):
        time_add = datetime.strptime(start, date_format) + timedelta(days=i+1)
        time = np.append(time, time_add.strftime(date_format))
    
    # Adding NaN values of to official data for plotting
    infect_data = np.concatenate((infect_data, [None] * (t_extend-len(infect_data))))
    recovered_data = np.concatenate((recovered_data, [None] * (t_extend-len(recovered_data))))
    
    # Plotting routines
    df = pd.DataFrame({'Susceptible': res.y[0], 
                        'Infected': res.y[1], 
                        'Recovered': res.y[2], 
                        "Infected (Data)": infect_data, 
                        "Recovered (Data)": recovered_data}, 
                        index=time)
    fig, ax = plt.subplots(figsize = (12,12))
    df['Susceptible'].plot(linestyle = '--', linewidth = 2)
    df['Infected'].plot(linestyle = '--', linewidth = 2)
    df['Recovered'].plot(linestyle = '--', linewidth = 2)
    df['Infected (Data)'].plot(linewidth = 4)
    df['Recovered (Data)'].plot(linewidth = 4)
    plt.ylabel("Number of people", fontsize=18)
    plt.xlabel("mm/dd/yy", fontsize=18)
    plot_title = ('Predictions for ' + country)
    plt.title(plot_title, fontsize=22)
    plt.legend()
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
   
    # Solving ODE 
    res = solve_ivp(sir, 
                    [0, len(infect_data)], 
                    [init[0], init[1], init[2]],
                    t_eval=np.arange(0,len(infect_data),1))
  
    # Returns RMSE
    rmse_infect = np.sqrt(np.mean((res.y[1] - infect_data)**2)) 
    rmse_recoverd = np.sqrt(np.mean((res.y[2] - recovered_data)**2)) 
    
    # Defining weights
    alpha = 0.2
    print(alpha*rmse_infect + (1-alpha)*rmse_recoverd)
    return alpha*rmse_infect + (1-alpha)*rmse_recoverd

plt.close()
c_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
r_df = pd.read_csv('time_series_covid19_recovered_global.csv')

country = "Singapore"
start_date = "1/25/20"

# Parsing confirmed/recovered cases from csv
confirmed = c_df[c_df['Country/Region'] == country].iloc[0].loc[start_date:]
recovered = r_df[r_df['Country/Region'] == country].iloc[0].loc[start_date:]
# Truncate last element to fit recovered data
confirmed = confirmed.values 
recovered = recovered.values

# I(t)
infect = confirmed - recovered

# Suceptable
N = 9000

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
                   bounds=[(0.00000001, 0.8), (0.00000001, 0.8)])
# Prints Beta/Gamma
print(optimal.x)

# Prints Gamma/Beta, if S0 > Gamma/Beta -> exponential growth
print("Gamma/Beta: " + str(optimal.x[1]/optimal.x[0]))

# Prints Repoductive Ratio S0*Beta/Gamma. 
# Measures number of secondary infection from primary infection, 2->5 based on Tom Rocks Math
print("S0*Beta/Gamma: " + str(S0*optimal.x[0]/optimal.x[1]))

# Predicting numbers from optimised gamma & beta
t = np.linspace(0,len(infect), len(infect))
predict(np.transpose(t), infect, recovered, [S0, I0, R0], optimal.x[0], optimal.x[1], start_date, country)


