# MSciProject

Code for MSci Project - ESA/NASA Solar Orbiter Electromagnetic Compatibility Testing

<!-- [![Contributors][contributors-shield]][contributors-url] -->
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
   - dBdI_data -------------------> contains .csv files with the xdata, ydata to plot dB vs dI
     - newdI_copy ---------------> correct dbdI with corrected dI values
        - Day 1
        - Day 2
          - 1Hz ------------> original files sampling MFSA data at 1Hz
          - 1Hz_with_err ---> same files but the random noise for each probe is added
          - 1kHz -----------> sampling MFSA at 1kHz (raw data)
          - mag ------------> for the stowed MAG_OBS
    - dBdB_oldplots ----------------> some saved plots of using 1Hz data
    - dI_graphs ---------------------> saved current profiles
    - Gradient_dicts ----------------> contains csv files of B/I constants proportionality constants
        - newdI_dicts ---------> correct dicts with corrected dI values
          - Day_1
          - Day_2
            - cur -------------> using `scipy.optimise.curve_fit` method
            - bool_cur ---------> boolean df showing if grads significant
        - old_dicts ------------> old dicts, show original methods
          - Day_1
          - Day_2
          - 1hz_noorigin ----------> gradients where not forced through origin
            - cur --------------> using `scipy.optimise.curve_fit` method
            - lin ---------------> using `scipy.stats.linregress` method
          - bool_check_grads_cur -> df's that contain booleans if grad/intercept signicifant
          - bool_check_grads_lin
    - PowerSpectrum ---------------> contains powerspectrum plots and files containing peaks
      - Day_2 ------------------> mainly Day2 power spectrum plots
      - Peak_files ---------------> contains .csv files with the selected peaks for each power spectra
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
current_newdI.py
```

Finds the datetime objects for each current step change for day 1 & 2. Found by looking at the gradient and setting a threshold. The anomalous peaks are removed. Current LCL data in spacecraft time. MFSA data in German UT local time. Exact current dI found averaging either side of step change for higher accuracy.


```python
processing.py
```

Contains class `processing` that contains many methods which reads in, cleans the csv raw data files, changes to correct timezone and creates datetimeindex, from MFSA. Also rotates the axis into the desired reference frame and contains the `calculate_db` method that the core of finding dB.

### Analysis/Plotting Files

```python
mfsa_object.py, deprecated(day_one.py, day_two.py) 
```
Updated class to better organise code (supersedes day_one and day_two). Used to plot, calculate powerspectrum & noise level of MFSA probes for a given timespan, probe and instrument.

```python
load_inflight.py
```

Contains class `burst_data` that creates an object for desired timespan for in-flight Burst mode data - similar analysis methods as used in mfsa_object.py.

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

Calculates average current (I) variation during METIS scientific operations and estimates dB at MAG_OBS location because of this dI.

```python
plot_variation.py
```

Plots bar chart and cubic fit of the variations csv folders - showing the estimated total dB at each probe (all 3 axis combined) - with errorbars

```python
plot_raw_current.py
```

Plots timeseries of raw LCL current data.

```python
probe_dist.py
```
Calculates distance of each MFSA probe from the center of the spacecraft and estimates 1/r^3 dipole factor.

```python
read_dBdI.py
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
vect_dipole_fit.py
```
Calculates magnetic moment in x,y,z for each probes' prop. const. and fits line through, if dipole should be flat line.

```python
vect_over_det_fit.py
```
Uses `numpy.linalg.lstsq` least squares regression to solve over determined system, 33 equations with 3 unknowns, also caculates r_2 to indicate fit.  (Fits data from 11 probes in 3 directions to find one set of magnetic moments for each instrument).

```python
lowpassdif.py
```
Calculates % dif if a order 10 butterworth low pass filter is applied to the MFSA data before calculating the proportionality constants.

```python
adjust_errs.py
```

Corrects the errors on dBdI_data to only be the random noise level by each probe (previously this was added in quadrature with the standard error of mean propagation).

```python
analyse_powerspec.py
```

Comparing different Peak_files .csv's.

```python
changedI_in_dbdi.py
```

Helper script to update gradient dicts with corrected dI values (using current_newdI.py instead of current.py) and save to new files.



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
