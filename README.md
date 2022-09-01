# Cluster-stocks
Clustering companies using their daily stock price movements using KMeans

This repository shows unsupervised learning approach on risk management. It presents methodology of clustering companies by their stocks, starting from preprocessing data, clustering and ending on visualization. It is worth to notice that this repository does not focus on the quality of predictions but on the methodology itself.

## Data preprocessing
Our data contains daily prices of WIG20 index portfolio over the late 2019-2022 period saved in OHLCV format. Normally this benchmark is based on the value of 20 major and most liquid companies listed on WSE, but in this example we excluded two companies (ALE and PCO) because of their recent IPOs. Stocks are quoted on WSE (Warsaw Stock Exchange) and price data comes from [stooq.pl](https://stooq.pl/)

```python
import glob
import pandas as pd

# get csv files from a folder
path = 'path_to_folder'
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
## Clustering stocks
We will cluster companies using their daily stock price movements (i.e. the PLN difference between the closing and opening prices for each trading day). Notice that some stocks are more expensive than others. To account for this, we will include a Normalizer at the beginning of our pipeline. The Normalizer will separately transform each company's stock price to a relative scale before the clustering begins.

```python
# create a dataframe where each row corresponds to a company and each column corresponds to a trading day
df_raw = pd.concat(df_list, axis=1, ignore_index=False)
df_t = df_raw.transpose()

tickers = ['ACP', 'CCC', 'CDR', 'CPS', 'DNP', 'JSW', 'KGH', 'KTY', 'LPP', 'MBK', 'OPL', 'PEO', 'PGE', 'PGN', 'PKN', 'PKO', 'PZU', 'SPL']

print(len(df_t))
print(len(tickers))

import numpy as np

# convert the price movements to Numpy matrix for fitting the model
df = np.matrix(df_t)

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# create a pipeline that chains normalizer and kmeans then fit the pipeline to the df array
normalizer = Normalizer()
kmeans = KMeans(n_clusters=10)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(df)

# predict the cluster labels for the price movements
labels = pipeline.predict(df)
dataframe = pd.DataFrame({'labels': labels, 'tickers': tickers})

# display dataframe sorted by cluster label
print(dataframe.sort_values('labels'))
```
## Visualizing hierarchies
We will perform hierarchical clustering of the companies using the dendrogram plot.

```python
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# rescale the price movements then calculate the hierarchical clustering, using 'complete' linkage
normalized_df = normalize(df)
mergings = linkage(normalized_df, method='complete')

# plot a dendrogram of the hierarchical clustering, using the company tickers as the labels
dendrogram(mergings,
           labels = tickers,
           leaf_rotation = 90,
           leaf_font_size = 6)

import matplotlib.pyplot as plt

plt.show()
```
