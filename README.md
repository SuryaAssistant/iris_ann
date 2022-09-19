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


### Data Modelling
Pembuatan model ANN, training, dan testing

Feature yang digunakan untuk kl;asifikasi adalah `sepalWidthCm`, `sepalLengthCm`, `petalWidthCm`, dan `petalLengthCm`.
Sedangakan, target klasifikasi adalah `setosa`, `versicolor`, dan `virginica`.

Ukuran data untuk training adalah 70% dari dataset dan 30% sisanya untuk testing

```
# Create train and test data

# Determine feature (x) and target (y)
data_vars = df_ann.columns.values.tolist()
y = ['setosa', 'versicolor', 'virginica']
x = [i for i in data_vars if i not in y]

# Use 70% data for training and 30% for testing
x_train, x_test, y_train, y_test = train_test_split(df_ann[x], 
                                                    df_ann[y], 
                                                    test_size=0.3, 
                                                    random_state=0)
```

Scalling data menggunakan StandardScaler

```
# Normalize train data

x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
```

Buat model ANN dengan 3 hidden layer. Tiap layer terdiri dari 10 sub

```
# Sci-kit learn ANN

# Use 3 hidden layer with 10 unit for each hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(10,10, 10), 
                    activation='relu', 
                    solver='sgd', 
                    max_iter=1000)

mlp.fit(x_train,y_train)
print("Training score : ", mlp.score(x_train, y_train))

y_predict_train = mlp.predict(x_train)
y_predict_test = mlp.predict(x_test)
```

Model telah dibuat dan kemudian diperiksa hasil keseluruhan terutama dalam f1 score dan confusion matrix

```
# Print Result


print('ARTIFICIAL NEURAL NETWORK TRAINING RESULT:')
print('======================================================')
print('Accuracy:', mlp.score(x_train, y_train))
print('======================================================')
print('Classification Report --------------------------------')
print(classification_report(y_train, y_predict_train))
print('======================================================')
print('Confussion Matrix Setosa -----------------------------')
print(confusion_matrix(y_train.setosa, y_predict_train[:,0]))
print('Confussion Matrix Versicolor -------------------------')
print(confusion_matrix(y_train.versicolor, y_predict_train[:,1]))
print('Confussion Matrix Virginica --------------------------')
print(confusion_matrix(y_train.virginica, y_predict_train[:,2]))


print('\n======================================================')

print('ARTIFICIAL NEURAL NETWORK TESTING RESULT:')
print('======================================================')
print('Accuracy:', mlp.score(x_test, y_test))
print('======================================================')
print('Classification Report --------------------------------')
print(classification_report(y_test, y_predict_test))
print('======================================================')
print('Confussion Matrix Setosa -----------------------------')
print(confusion_matrix(y_test.setosa, y_predict_test[:,0]))
print('Confussion Matrix Versicolor -------------------------')
print(confusion_matrix(y_test.versicolor, y_predict_test[:,1]))
print('Confussion Matrix Virginica --------------------------')
print(confusion_matrix(y_test.virginica, y_predict_test[:,2]))

print('\n======================================================')


# Test to all data in dataset
x_test_df, x_train_df, y_test_df, y_train_df = train_test_split(df_ann[x], df_ann[y], test_size=1, random_state=99)
x_test_df = StandardScaler().fit_transform(x_test_df)
y_predict_df = mlp.predict(x_test_df)

print('ARTIFICIAL NEURAL NETWORK TESTING TO DATASET RESULT:')
print('======================================================')
print('Accuracy:', mlp.score(x_test_df, y_test_df))
print('======================================================')
print('Classification Report --------------------------------')
print(classification_report(y_test_df, y_predict_df))
print('======================================================')
print('Confussion Matrix Setosa -----------------------------')
print(confusion_matrix(y_test_df.setosa, y_predict_df[:,0]))
print('Confussion Matrix Versicolor -------------------------')
print(confusion_matrix(y_test_df.versicolor, y_predict_df[:,1]))
print('Confussion Matrix Virginica --------------------------')
print(confusion_matrix(y_test_df.virginica, y_predict_df[:,2]))

print('\n======================================================')
```



### Hasil

```
ARTIFICIAL NEURAL NETWORK TRAINING RESULT:
======================================================
Accuracy: 0.9809523809523809
======================================================
Classification Report --------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        34
           1       1.00      0.94      0.97        32
           2       0.95      1.00      0.97        39

   micro avg       0.98      0.98      0.98       105
   macro avg       0.98      0.98      0.98       105
weighted avg       0.98      0.98      0.98       105
 samples avg       0.98      0.98      0.98       105

======================================================
Confussion Matrix Setosa -----------------------------
[[71  0]
 [ 0 34]]
Confussion Matrix Versicolor -------------------------
[[73  0]
 [ 2 30]]
Confussion Matrix Virginica --------------------------
[[64  2]
 [ 0 39]]

======================================================
ARTIFICIAL NEURAL NETWORK TESTING RESULT:
======================================================
Accuracy: 0.9111111111111111
======================================================
Classification Report --------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.78      0.88        18
           2       0.73      1.00      0.85        11

   micro avg       0.91      0.91      0.91        45
   macro avg       0.91      0.93      0.91        45
weighted avg       0.93      0.91      0.91        45
 samples avg       0.91      0.91      0.91        45

======================================================
Confussion Matrix Setosa -----------------------------
[[29  0]
 [ 0 16]]
Confussion Matrix Versicolor -------------------------
[[27  0]
 [ 4 14]]
Confussion Matrix Virginica --------------------------
[[30  4]
 [ 0 11]]

======================================================
ARTIFICIAL NEURAL NETWORK TESTING TO DATASET RESULT:
======================================================
Accuracy: 0.9731543624161074
======================================================
Classification Report --------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       1.00      0.92      0.96        50
           2       0.92      1.00      0.96        49

   micro avg       0.97      0.97      0.97       149
   macro avg       0.97      0.97      0.97       149
weighted avg       0.98      0.97      0.97       149
 samples avg       0.97      0.97      0.97       149

======================================================
Confussion Matrix Setosa -----------------------------
[[99  0]
 [ 0 50]]
Confussion Matrix Versicolor -------------------------
[[99  0]
 [ 4 46]]
Confussion Matrix Virginica --------------------------
[[96  4]
 [ 0 49]]

======================================================
```

Dari model yang telah dibuat, didapatkan akurasi training 91% dan akurasi testing sebesar 91%. Dengan mengujikan model ke keseluruhan dataset, didapatkan akurasi total sebesar 97%. F1 score untuk setiap klasifikasi spesies terlihat seperti pada hasil pengujian di atas.
