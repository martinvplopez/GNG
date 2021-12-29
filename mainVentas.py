import pandas as pd
import tensorflow as tf
from GrowingNeuralGas import GrowingNeuralGas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# # Limpieza e introducción al sistema inteligente del GNG del conjunto de datos: Clientes de ventas al por mayor

df=pd.read_csv("Clientes _Ventas_por_Mayor.csv")
# descripcionData=df.describe().loc[['mean','std']]

## Aprendizaje y analisis del dataset con todas las columnas

# Estandarización de los datos
features1 = list(df.columns)
x = StandardScaler().fit_transform(df)

# standarizeddf = pd.DataFrame(data = x, columns = features1)
# standarizeddfmS=standarizeddf.describe().loc[['mean','std']]
# print(standarizeddfmS)

# Realizacion PCA
pca = PCA(n_components=4)
principal_componentes = pca.fit(x)
Xpca=pca.transform(x)
dfPCA= pd.DataFrame(
    data    = Xpca,
    columns = ['PC1','PC2','PC3','PC4']
)
# print("Shape datos estandarizados", x.shape)
# print("Shape datos con PCA",Xpca.shape)

# Training set y muestras a predecir en Dataset con Canal y región

trainPCA= dfPCA.loc[:351].values #
training_set = tf.constant(trainPCA, dtype=tf.float32)
predictSample1=tf.constant([dfPCA.iloc[432]], dtype=tf.float32) # Canal 2
predictSample2=tf.constant([dfPCA.iloc[439]], dtype=tf.float32) # Canal 1
predictSample3=tf.constant([dfPCA.iloc[438]], dtype=tf.float32) # Región 1
predictSample4=tf.constant([dfPCA.iloc[356]], dtype=tf.float32) # Región 2
predictSample5=tf.constant([dfPCA.iloc[434]], dtype=tf.float32) # Región 3

growingNeuralGas = GrowingNeuralGas()
growingNeuralGas.fit(training_set, 5,7)

print(growingNeuralGas.predict(predictSample1))
print(growingNeuralGas.predict(predictSample2))
print(growingNeuralGas.predict(predictSample3))
print(growingNeuralGas.predict(predictSample4))
print(growingNeuralGas.predict(predictSample5))


# # Mapa de calor para analizar el peso de cada variable por componente

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
# componentes = principal_componentes.components_
# plt.imshow(componentes.T, cmap='viridis', aspect='auto')
# plt.yticks(range(len(standarizeddf.columns)), standarizeddf.columns)
# plt.xticks(range(4), np.arange(principal_componentes.n_components_) + 1)
# plt.grid(False)
# plt.colorbar()
# plt.savefig("heatMapPrincipalComponents.png")
# plt.close()

## Analisis dataset sin tener en cuenta "Región" y "Canal"

df.drop("Region", axis=1, inplace=True)
df.drop("Channel", axis=1, inplace=True)

# Estandarización
features2 = list(df.columns)
x2 = StandardScaler().fit_transform(df)
# standarizeddf2 = pd.DataFrame(data =x2, columns = features2)
# standarizeddfmS2=standarizeddf2.describe().loc[['mean','std']]
# print(standarizeddfmS2)

# Realizacion PCA
pca2 = PCA(n_components=4)
principal_componentes = pca2.fit(x2)
Xpca2=pca2.transform(x2)
dfPCA2= pd.DataFrame(
    data    = Xpca2,
    columns = ['PC1','PC2','PC3','PC4']
)
# print("Shape datos estandarizados", x2.shape)
# print("Shape datos con PCA",Xpca2.shape)

# Training set y muestras a predecir en Dataset sin Canal y región

trainPCA2= dfPCA2.loc[:351].values #
training_set2 = tf.constant(trainPCA2, dtype=tf.float32)
predictSample1=tf.constant([dfPCA2.iloc[432]], dtype=tf.float32) # Canal 2
predictSample2=tf.constant([dfPCA2.iloc[439]], dtype=tf.float32) # Canal 1
predictSample3=tf.constant([dfPCA2.iloc[438]], dtype=tf.float32) # Región 1
predictSample4=tf.constant([dfPCA2.iloc[356]], dtype=tf.float32) # Región 2
predictSample5=tf.constant([dfPCA2.iloc[434]], dtype=tf.float32) # Región 3

growingNeuralGas = GrowingNeuralGas()
growingNeuralGas.fit(training_set2, 5,7)

print(growingNeuralGas.predict(predictSample1))
print(growingNeuralGas.predict(predictSample2))
print(growingNeuralGas.predict(predictSample3))
print(growingNeuralGas.predict(predictSample4))
print(growingNeuralGas.predict(predictSample5))