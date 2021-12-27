import pandas as pd
import tensorflow as tf
from GrowingNeuralGas import GrowingNeuralGas
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Limpieza e introducción al sistema inteligente del GNG del conjunto de datos: Clientes de ventas al por mayor


df=pd.read_csv("Clientes _Ventas_por_Mayor.csv")
descripcionData=df.describe().loc[['mean','std']]
# print(descripcionData)

# df.plot(x ='Region', y='Fresh', kind = 'scatter')
# plt.show()


#Aprendizaje y analisis del dataset con todas las columnas
features1 = list(df.columns)

trainx1 = df.loc[:351,features1].values # Conjunto de entrenamiento
predictx1 = df.loc[351:, features1].values # Conjunto de predicciones

# print(trainx1)

x = StandardScaler().fit_transform(trainx1)
standarizeddf = pd.DataFrame(data = x, columns = features1)
# standarizeddfmS=standarizeddf.describe().loc[['mean','std']]
# print(standarizeddfmS)



# Analisis componentes principales con todas las columnas
pca = PCA(n_components=None)
principal_componentes = pca.fit(x)
dfPCA= pd.DataFrame(
    data    = principal_componentes.components_,
    columns = df.columns,
    index   = ['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8']
)
# Mapa de calor para analizar el peso de cada variable por componente
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = principal_componentes.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(standarizeddf.columns)),standarizeddf.columns)
plt.xticks(range(len(standarizeddf.columns)), np.arange(principal_componentes.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.savefig("heatMapPrincipalComponents.png")
plt.close()

#principalDf = pd.DataFrame(data = principal_componentes ,columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])


#
#
# df.drop("Region", axis=1, inplace=True)
# df.drop("Channel", axis=1, inplace=True)
#
# features2 = list(df.columns)
#
# trainx2 = df.loc[:351,features2].values
# predictx2 = df.loc[351:, features2].values


#
# training_set = tf.constant(trainx1, dtype=tf.float32)
#
# training_set_without_region_channel = tf.constant(trainx2, dtype=tf.float32)
# #
# validation_set = tf.constant(predictx1, dtype= tf.float32)
#
# validation_set2 = tf.constant(predictx2, dtype= tf.float32)
#
#
# growingNeuralGas = GrowingNeuralGas()
# growingNeuralGas.fit(training_set, 5)

# growingNeuralGas.predict(validation_set)