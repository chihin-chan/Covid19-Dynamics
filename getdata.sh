#!/bin/bash
cd ../COVID-19/csse_covid_19_data/csse_covid_19_time_series
git pull
cp time_series_covid19_confirmed_global.csv time_series_covid19_recovered_global.csv ../../../Covid19-Dynamics
cd ../../../Covid19-Dynamics
exec bash 
