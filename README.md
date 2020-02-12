# MSciProject

Code for MSci Project - ESA/NASA Solar Orbiter Electromagnetic Compatibility Testing

[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- [![Forks][forks-shield]][0] -->
<!-- [![Stargazers][stars-shield]][0] -->


## Introduction

In June 2019, ESA/NASA Solar Orbiter spacecraft sent to IABG's facility: MFSA (Magnetfeldsimulationnanlage) in Munich to undergo magnetic testing.

21st June - 1st Powered Day - Instruments tested cumulatively

24th June - 2nd Powered Day - Instruments tested separately

## Navigating the Data

- Results
   - dBdI_data -------------------> contains csv files with the xdata, ydata to plot dB vs dI
      - Day 1
      - Day 2
        - 1Hz ------------> original files sampling MFSA data at 1Hz
        - 1Hz_with_err ---> same files but the random noise for each probe is added
        - 1kHz -----------> sampling MFSA at 1kHz (raw data)
        - mag ------------> for the stowed MAG_OBS
    - dBdB_oldplots ----------------> some saved plots ofusing 1Hz data
    - dI_graphs ---------------------> saved current profiles
    - Gradient_dicts ----------------> contains csv files of B/I proportionality constants
        - Day_1
        - Day_2
        - 1hz_noorigin ----------> gradients where not forced through origin
          - cur --------------> using `scipy.optimise.curve_fit` method
          - lin ---------------> using `scipy.stats.linregress` method
        - bool_check_grads_cur -> df's that contain booleans if grad/intercept signicifant
        - bool_check_grads_lin
    - PowerSpectrum ---------------> powerspectrum for day 2 MFSA - need to be updated
    - Variation -----------------------> contains csv for estimated dB for I variation in EUI and METIS
  
## Code files


### dB_pyfiles
  
```python
dB.py
```
Calculates the change in Magnetic field dB due to a step change in current of an instrument. At the moment, averages 30 seconds of magnetic field measurements either side of step change (with 2sec buffer zone either side as current step change timestamp 2.5sec uncertainty). Outputs dBdI data and Gradient Dicts.

```python
dBmag.py
```

Same as `dB.py` but for MAG_OBS (magnetometer instrument) that was stowed but operating during testing.

```python
dBmag2.py
```

Same as `dBmag.py` but does not output files, only plots.

```python
dBsaveall.py
```

Same as `dB.py` but creates files for all probes, all instruments on a given day - now redunant as this was built into `dB.py` recently.

### Core Files
```python
current.py
```

Finds the datetime objects for each current step change for day 1 & 2. Found by looking at the gradient and setting a threshold. The anomalous peaks are removed. Current LCL data in spacecraft time. MFSA data in German UT local time.

```python
processing.py
```

Contains class `processing` that contains many methods which reads in, cleans the csv raw data files (changes to correct timezone and creates datetimeindex) from MFSA. Also rotates the axis into the desired reference frame and contains the `calculate_db` method that the core of finding dB.


### Analysis/Plotting Files

```python
day_one.py, day_two.py
```
Used to plot, calculate powerspectrum & noise level of MFSA probes for a given timespan, probe and instrument

```python
check_vect_dicts.py
```
Creates the boolean dfs, that checks if grad > 2*sigma or if origin (0,0) within intercept uncertainty

```python
mag.py
```
Plots timeseries data from stowed MAG_OBS instrument during powered testing.  Used to calibrate the timezones looking at the power amplifier peaks.

```python
magplot_1.py, magplot_2.py
```

Time index of MAG_OBS for day 1 and day 2 properly calculated (`mag.py` now redundant). So better for plotting and finding peaks.

```python
metis_var.py
```

Calculates average current (I) variation during METIS scientific operations and estimates dB at MAG_OBS location because of this dI

```python
plot_variation.py
```

Plots bar chart of the Variations csv folders - showing the estimated total dB at each probe (all 3 axis combined) - with errorbars

```python
probe_dist.py
```
Calculates distance of each MFSA probe from the center of the spacecraft and estimates 1/r^3 dipole factor.

```python
read_dBdO.py
```

Reads in the dBdI data, calculates slope using either `scipy.stats.linregress` or `scipy.optimize.curve_fit` method and shows the plots. Very quick.

```python
variation.py
```
Calculates the variation measured in MAG_OBS and MFSA data due to noisy current in EUI and METIS.  Used to validate our estimates.

```python
vect_map.py
```
Plots 3D map showing the vectors of the proportionality constants calculated at every probe for a given instrument. Probe 12 not shown as too far awawy and often dominates the plot.


```python
lowpassdif.py
```
Calculates % dif if a order 10 butterworth low pass filter is applied to the MFSA data before calculating the proportionality constants





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/JonasSinjan/MSciProject/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/JonasSinjan/MSciProject/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/JonasSinjan/MSciProject/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/JonasSinjan/MSciProject/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jonas-sinjan