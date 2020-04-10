###############################################################################
#
# SIR Model for Covid-19
# Data from John-Hopkins University
#
# By Chi Hin, Jean Phillipe
#
###############################################################################



from covid_class import Covid

# Initialise class Covid("Country", "State", "Start Date")
# If no state leave it is ""
sg = Covid("Singapore", "", "1/22/20")

# Loads Data from John Hopkins
sg.load()

# find_beta_gamma(S0, alpha)
# Input Total Population N s.t S0 = N - 2
# Alpha = (1-alpha)*rmse_recoverd + alpha*rmse_infect
sg.find_beta_gamma(7000, 0.1)

# predict(days) -> days refers to the limits of ODE solver since start date
sg.predict(200)