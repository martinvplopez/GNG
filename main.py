import tensorflow as tf

from GrowingNeuralGas import GrowingNeuralGas


tf.random.set_seed(23)
X = tf.concat([tf.random.normal([100, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([0.0, 0.0]),
                tf.random.normal([100, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([0.0, 1.0]),
               tf.random.normal([100, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 1.0])], 0)

sample= tf.random.normal([1,2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 1.0])


growingNeuralGas = GrowingNeuralGas()
growingNeuralGas.fit(X, 1)
print("Predicted cluster:",growingNeuralGas.predict(sample))

