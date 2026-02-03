# MJO-Seasonal-Cycle-Code
Replicates Roundy 2026c, Zonal Propagation of the Indian Basin MJO in Varying Background Wind and Seasonal Background Wind Regimes

The file speedIOucomppaper.py is written in python 3, and uses data arrays I extracted from the ERA5 reanalysis and the NOAA interpolated OLR dataset (you will need to create your own arrays to replicate the project). 
Data arrays should be daily averages, with the ERA5 data averaged to 1 degree latitude longitude resolution, and the OLR in its native 2.5-degree grid. Time should be in dimension 0 of the arrays. Latitude in my original OLR arrays is 30N to 30S, and longitude is from 0-357.5 in the OLR arrays and from -180E to 179E in the ERA5 data arrays. The code makes these both consistent by altering the ERA5 arrays to begin at 0E. 
You will need to edit the figure and data file paths where needed. 
