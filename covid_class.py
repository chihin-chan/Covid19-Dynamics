###############################################################################
#
# SIR Model for Covid-19
# Data from John-Hopkins University
#
# By Chi Hin, Jean Phillipe
#
# Class: COVID
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


# Class COVID
class Covid:
    def __init__(self, country, state, start_date):
        self.country  = country
        self.state = state
        self.start_date = start_date

    def load(self):
        c_df = pd.read_csv('time_series_covid19_confirmed_global.csv')
        r_df = pd.read_csv('time_series_covid19_recovered_global.csv')
        # Saving end date
        self.end_date = c_df.columns[-1]
        
        # Parsing confirmed/recovered cases from csv
        if self.state == "":
            c_df = c_df[c_df['Province/State'].isnull()]
            c_df = c_df[c_df['Province/State'].isnull()]
            self.confirmed = c_df[c_df['Country/Region'] == self.country].iloc[0].loc[self.start_date:]
            self.recovered = r_df[r_df['Country/Region'] == self.country].iloc[0].loc[self.start_date:]
            print("Country Selected: " + self.country)
        elif self.state != 0:
            self.confirmed = c_df[c_df['Province/State'] == self.state].iloc[0].loc[self.start_date:]
            self.recovered = r_df[r_df['Province/State'] == self.state].iloc[0].loc[self.start_date:]
            print("State Selected: " + self.state)
        else:
            print("Invalid Country/Province Enter")            
        
        # Truncate last element to fit recovered data
        self.confirmed = self.confirmed.values 
        self.recovered = self.recovered.values
        
        # I(t)
        self.infected = self.confirmed - self.recovered
        print('Data loaded Successful!')
    
    def find_beta_gamma(self, N, alpha):
        
        # Number of infected at t0 for ODE Solver
        self.I0 = 2
        self.S0 = N-self.I0
        self.R0 = 0
        
        # Finding Beta and Gamma by optimisation optimise
        optimal = minimize(cost, 
                   [0.00001, 0.00001], 
                   args=(self.infected, self.recovered, [self.S0, self.I0, self.R0], alpha),
                   method='L-BFGS-B',
                   bounds=[(0.00000001, 0.8), (0.00000001, 0.8)])
        
        # Saving optimal beta/gamma into private variables
        self.beta = optimal.x[0]
        self.gamma = optimal.x[1]
        
    def predict(self, day):
        # ODE Function
        def sir(t, y):

            # Outputs
            S = y[0]
            I = y[1]
            R = y[2]
            
            return [-self.beta*I*S, self.beta*I*S - self.gamma*I, self.gamma*I]
        
        # Extending simulation based on how many days
        t_extend = len(self.confirmed) + day
        t = np.linspace(0, t_extend, t_extend)
        # Solving SIR
        res = solve_ivp(sir, 
                        [0, t_extend], 
                        [self.S0, self.I0, self.R0],
                        t_eval=np.arange(0,t_extend,1))
        
        # Preparing date axis
        date_format = '%m/%d/%y'
        time = self.start_date
        for i in range(t_extend-1):
            time_add = datetime.strptime(self.start_date, date_format) + timedelta(days=i+1)
            time = np.append(time, time_add.strftime(date_format))
        
        # Adding NaN values of to official data for plotting
        self.infected = np.concatenate((self.infected, [None] * (t_extend-len(self.infected))))
        self.recovered = np.concatenate((self.recovered, [None] * (t_extend-len(self.recovered))))
        
        # Plotting routines
        plt.close()
        df = pd.DataFrame({'Susceptible': res.y[0], 
                            'Infected': res.y[1], 
                            'Recovered': res.y[2], 
                            "Infected (Data)": self.infected, 
                            "Recovered (Data)": self.recovered}, 
                            index=time)
        fig, ax = plt.subplots(figsize = (12,12))
        df['Susceptible'].plot(color='b',linestyle = '--', linewidth = 2)
        df['Infected'].plot(color='r',linestyle = '--', linewidth = 2)
        df['Recovered'].plot(color='g',linestyle = '--', linewidth = 2)
        df['Infected (Data)'].plot(color='r',linewidth = 4)
        df['Recovered (Data)'].plot(color='g',linewidth = 4)
        plt.ylabel("Number of people", fontsize=18)
        plt.xlabel("mm/dd/yy", fontsize=18)
        info = ( "Data Updated:" + self.end_date + '\n' 
                + "Beta: " + str("{:.9f}".format(self.beta)) + '\n' 
                + "Gamma: " + str("{:.5f}".format(self.gamma)) + '\n' 
                + "S0: " + str(self.S0) + '\n' 
                + "Reproduction No: (S0*B/g): "+ str("{:.3f}".format(self.S0*self.beta/self.gamma)) + '\n' 
                + "Max Inf: " + str("{:.0f}".format(max(df['Infected'])))
                )
        plt.text(0.02, 0.8, info, fontsize = 10, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plot_title = ('Predictions for ' + self.country + " " + self.state + " to " + time[-1])
        plt.title(plot_title, fontsize=22)
        plt.legend()
        plt.grid()
        plt.show()

    def predict2(self, day, red):
        # ODE Function
        def sir(t, y):

            # Outputs
            S = y[0]
            I = y[1]
            R = y[2]
            
            return [-self.beta*I*S, self.beta*I*S - self.gamma*red*I, self.gamma*I]
        
        # Extending simulation based on how many days
        t_extend = len(self.confirmed) + day
        t = np.linspace(0, t_extend, t_extend)
        # Solving SIR
        res = solve_ivp(sir, 
                        [0, t_extend], 
                        [self.S0, self.I0, self.R0],
                        t_eval=np.arange(0,t_extend,1))
        
        # Preparing date axis
        date_format = '%m/%d/%y'
        time = self.start_date
        for i in range(t_extend-1):
            time_add = datetime.strptime(self.start_date, date_format) + timedelta(days=i+1)
            time = np.append(time, time_add.strftime(date_format))
        
        # Adding NaN values of to official data for plotting
        self.infected = np.concatenate((self.infected, [None] * (t_extend-len(self.infected))))
        self.recovered = np.concatenate((self.recovered, [None] * (t_extend-len(self.recovered))))
        
        # Plotting routines
        plt.close()
        df = pd.DataFrame({'Susceptible': res.y[0], 
                            'Infected': res.y[1], 
                            'Recovered': res.y[2], 
                            "Infected (Data)": self.infected, 
                            "Recovered (Data)": self.recovered}, 
                            index=time)
        fig, ax = plt.subplots(figsize = (12,12))
        df['Susceptible'].plot(color='b',linestyle = '--', linewidth = 2)
        df['Infected'].plot(color='r',linestyle = '--', linewidth = 2)
        df['Recovered'].plot(color='g',linestyle = '--', linewidth = 2)
        df['Infected (Data)'].plot(color='r',linewidth = 4)
        df['Recovered (Data)'].plot(color='g',linewidth = 4)
        plt.ylabel("Number of people", fontsize=18)
        plt.xlabel("mm/dd/yy", fontsize=18)
        info = ( "Data Updated:" + self.end_date + '\n' 
                + "Beta: " + str("{:.9f}".format(self.beta)) + '\n' 
                + "Gamma: " + str("{:.5f}".format(self.gamma*red)) + '\n' 
                + "S0: " + str(self.S0) + '\n' 
                + "Reproduction No: (S0*B/g): "+ str("{:.3f}".format(self.S0*self.beta/self.gamma/red)) + '\n' 
                + "Max Inf: " + str("{:.0f}".format(max(df['Infected'])))
                )
        plt.text(0.02, 0.8, info, fontsize = 10, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        plot_title = ('Predictions for ' + self.country + " " + self.state + " to " + time[-1])
        plt.title(plot_title, fontsize=22)
        plt.legend()
        plt.grid()
        plt.show()

# Cost function
def cost(point, infect_data, recovered_data, init, alpha):
    
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
    print(alpha*rmse_infect + (1-alpha)*rmse_recoverd)
    return alpha*rmse_infect + (1-alpha)*rmse_recoverd
