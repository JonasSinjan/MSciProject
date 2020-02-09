import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.signal as sps
from datetime import datetime, timedelta
import time
import math
import csv


windows = False

if windows == True:
    path1 = "..."
    path2 = "..."
elif windows == False:
    path1 = os.path.expanduser("~/Documents/MSciProject/Data/mag/Day1MAGBurst1.csv")
    path2 = os.path.expanduser("~/Documents/MSciProject/Data/mag/Day1MAGBurst2.csv")


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)


collist = ['time','X','Y','Z']
df1.columns = collist
df2.columns = collist


end_df1 = list(df1["time"])[-1]
inc_df1 = list(df1["time"])[1]

df2["time"] = df2["time"] + end_df1 + inc_df1


df_tot = {}
for col in collist:
    df_tot[col] = df1[col]
    df_tot[col][775168:1823744] = df2[col]


print (df_tot.head())
print (df_tot.tail())


