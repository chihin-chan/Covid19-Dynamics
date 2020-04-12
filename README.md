# Covid19-Dynamics

Modeling the dynamics of the COVID19 disease with SIR model using Python

### Prerequisites

Tested with Python 3.7.4

## Running

An example for running the case of Singapore is shown in the main.py script
```
# Initialise class Covid("Country", "State", "Start Date")
# If no state leave it is "sg = Covid("China", "Hubei", "1/22/20")
sg = Covid("Singapore", "", "1/22/20")

# Loads Data from John Hopkins
sg.load()

# find_beta_gamma(S0, alpha)
# Input Total Population N s.t S0 = N - 2
# Alpha = (1-alpha)*rmse_recoverd + alpha*rmse_infect
sg.find_beta_gamma(8000, 0.1)

# predict(days) -> days refers to the limits of ODE solver since start date
sg.predict(150)

# reduce reproduciton number by 1.5 times with predict2
sg.predict2(150, 1.5)
``` 

## Authors

* **Chan Chi Hin** (https://github.com/CHCFD)
* **Jean-Phillippe Kuntzer** (https://github.com/jphkun)

## References
https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html


