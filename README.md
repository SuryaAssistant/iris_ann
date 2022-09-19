<!-- Title -->
<span align = "center">

# Iris Flower Species Classification using ANN

Oleh: Fandi Adinata @2022

</span>
<!-- End of Title -->

<br>

## Prequerities
- Install Jupyter Notebook. If not, you can use Google Collab and skip prequerities below.
- For Linux :
  - Install Numpy
  ```
  pip install numpy
  ``` 
  - Install Pandas
  ```
  pip install pandas
  ``` 
  - Install sci-kit learn
  ```
  pip install scikit-learn
  ``` 
  - Install Seaborn
  ```
  pip install seaborn
  ``` 
- Download this repository and move to your workspace


## Classification Result

Program dan hasil secara singkat tertera di bawah

### Import Library
```
# General library(s)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Library(s)
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning Result Metric
from sklearn.metrics import confusion_matrix, classification_report

```

### Data Loading
Import dataset dari folder `dataset`

```
# Load dataset to project
df = pd.read_csv("./dataset/iris.csv")
```

### Exploratory Data Analysis
```
sns.countplot(x='Species', data=df)
plt.show()
```
<span align = "center">
  
![Logo](https://github.com/SuryaAssistant/iris_ann/blob/main/export/countplot.png)
  
</span>

Masing-masing spesies memiliki jumlah data yang sama, yaitu 50 sampel data. Sehingga tidak diperlukan penyeimbangan data ketika pembuatan model ANN.

```
## Visualize data based on it species in every feature

sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')
```

<span align = "center">
  
![Logo](https://github.com/SuryaAssistant/iris_ann/blob/main/export/sepal.png)

![Logo](https://github.com/SuryaAssistant/iris_ann/blob/main/export/petal.png)

</span>

Spesies Setosa dapat dibedakan dari dua spesies yang lain menggunakan scatter plot karena posisinya yang berjauhan dari dua spesies yang lain. Di sisi lain, spesies versicolor dengan virginica lokasi datanya tumpang tindah. Namun lebih dapat dibedakan jika menggunakan ukuran petal dari masing-masing spesies.

```
# Show pairplot

sns.pairplot(data=df, hue="Species")
```

<span align = "center">
  
![Logo](https://github.com/SuryaAssistant/iris_ann/blob/main/export/pairplot.png)

</span>

```
# Boxplot to see outlier

df.plot(kind='box', subplots = True, layout = (4,4), figsize = (14,14))
plt.show()
```

Seluruh feature bersih dari outlier kecuali untuk feature sepalWidth

### Data Preprocessing
Olah data seperlunya dan sebelum pembuatan model

Copy feature (input) ke dataframe baru yang terpisah

```
# Copy feature

category = df[["SepalLengthCm",
                           "SepalWidthCm",
                           "PetalLengthCm", 
                           "PetalWidthCm"]]

df_ann = category.copy()

df_ann
```

Gunakan one-hot encoding untuk mengubah data String menjadi angka label

```
# One-hot encoding for 'Species' categorical data
encoding = pd.get_dummies(df.Species, prefix='species')

## Concat data
df_ann = pd.concat([df_ann, encoding], axis=1)

# Rename
df_ann.rename(columns = {'species_Iris-setosa':'setosa',
                         'species_Iris-versicolor':'versicolor',
                         'species_Iris-virginica':'virginica'}, 
              inplace = True)

df_ann
```

Lihat korelasi antar feature dengan target klasifikasi

```
fig = plt.subplots(figsize=(6,6))
sns.heatmap(df_ann.corr(), annot=True)
```

<span align = "center">
  
![Logo](https://github.com/SuryaAssistant/iris_ann/blob/main/export/heatmap.png)

</span>

