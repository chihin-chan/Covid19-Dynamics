###############################################################################
#
# SIR Model for Covid-19
# Data from John-Hopkins University
#
# By Chi Hin, Jean Phillipe
#
###############################################################################



from covid_class import Covid



sg = Covid("Singapore", "1/25/20")
sg.load()
sg.find_beta_gamma(6000)
sg.predict(200)