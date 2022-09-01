# Cluster-stocks
Clustering companies using their daily stock price movements using KMeans

This repository shows

## Data preprocessing

```python
import glob
import pandas as pd

# get csv files from a folder
path = '/Users/Daniel/Downloads'
csv_files = glob.glob(path + "/*.csv")

# read each csv file and create a list of dataframes
list_of_df = (pd.read_csv(file) for file in csv_files)

df_list_raw = []

# loop through generator object to create a readable list of dataframes 
for df in list_of_df:
    df_list_raw.append(df)

# prepare labels to drop
label = ['Najwyzszy',
         'Najnizszy',
         'Wolumen']

df_list = []

# loop through each dataframe to calculate the PLN difference between the closing and opening prices for each trading day
for df in df_list_raw:
    sub_df = df
    x = sub_df.drop(label, axis = 1)
    y = x.set_index('Data')
    y['diff'] = y['Zamkniecie'] - y['Otwarcie']
    z = y['diff']
    df_list.append(z)
```
