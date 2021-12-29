import pandas as pd
import tensorflow as tf
from GrowingNeuralGas import GrowingNeuralGas
from sklearn.preprocessing import StandardScaler

# Aplicar GNG al conjunto de testeo "Sample Cluster Data 2D"

df=pd.read_csv("Sample_Cluster_Data_2D.csv")
print(df.head())

features1 = list(df.columns)
trainx1 = df.loc[:1500,features1].values # Conjunto de entrenamiento

x = StandardScaler().fit_transform(trainx1)
training_set = tf.constant(x, dtype=tf.float32)
growingNeuralGas = GrowingNeuralGas()
growingNeuralGas.fit(training_set, 1, 7)
