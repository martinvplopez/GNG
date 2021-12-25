import pandas as pd
import tensorflow as tf
# Aplicar GNG al conjunto de testeo "Sample Cluster Data 2D"
df=pd.read_csv("Clientes _Ventas_por_Mayor.csv")

from GrowingNeuralGas import GrowingNeuralGas


training_set = tf.constant(df.to_numpy(), dtype=tf.float32)

growingNeuralGas = GrowingNeuralGas()
growingNeuralGas.fit(training_set, 1)

# growingNeuralGas = GrowingNeuralGas()
# growingNeuralGas.fit(X, 5)