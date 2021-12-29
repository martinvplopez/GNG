import functools

import numpy as np
import tensorflow as tf

from Graph import Graph
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter

class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=10, eta=10, alpha=.1, delta=.1, maxNumberUnits=500):
        self.A = None
        self.N = []
        self.connectedComponents=0
        self.error_ = None
        self.epsilon_a = epsilon_a
        self.epsilon_n = epsilon_n
        self.a_max = a_max
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.maxNumberUnits = maxNumberUnits

    def incrementAgeNeighborhood(self, indexNearestUnit):
        self.N[indexNearestUnit].incrementAgeNeighborhood(1.0)
        for indexNeighbour in self.N[indexNearestUnit].neighborhood:
            self.N[indexNeighbour].incrementAgeNeighbour(indexNearestUnit, 1.0)

    def findNearestUnit(self, xi, A):
        return tf.math.argmin(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1))

    def findSecondNearestUnit(self, xi, A):
        indexNearestUnit = self.findNearestUnit(xi, A)
        error_ = tf.constant(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1), dtype=tf.float32).numpy()
        error_[indexNearestUnit] = np.Inf
        return tf.math.argmin(tf.constant(error_))

    def findIndexNeighbourMaxError(self, indexUnitWithMaxError_):
        index = tf.squeeze(tf.math.argmax(tf.gather(self.error_, self.N[indexUnitWithMaxError_].neighborhood)), 0)
        indexNeighbourMaxError = self.N[indexUnitWithMaxError_].neighborhood[index]
        return indexNeighbourMaxError

    def pruneA(self):
        indexToNotRemove = [index for index in tf.range(self.N.__len__()) if self.N[index].neighborhood.__len__() > 0]
        self.A = tf.Variable(tf.gather(self.A, indexToNotRemove, axis=0))

        for graphIndex in reversed(range(self.N.__len__())):
            if self.N[graphIndex].neighborhood.__len__() == 0:
                for pivot in range(graphIndex + 1, self.N.__len__()):
                    self.N[pivot].id -= 1
                    for indexN in range(self.N.__len__()):
                        for indexNeighbothood in range(self.N[indexN].neighborhood.__len__()):
                            if self.N[indexN].neighborhood[indexNeighbothood] == pivot:
                                self.N[indexN].neighborhood[indexNeighbothood] -= 1
                self.N.pop(graphIndex)

    def getGraphConnectedComponents(self):
        connectedComponentIndeces = list(range(self.N.__len__())) # Indices de los nodos empiezan desde el 0
        for graphIndex in range(self.N.__len__()):
            for neighbourIndex in self.N[graphIndex].neighborhood:
                if connectedComponentIndeces[graphIndex] <= connectedComponentIndeces[neighbourIndex]:
                    connectedComponentIndeces[neighbourIndex] = connectedComponentIndeces[graphIndex]
                else:
                    aux = connectedComponentIndeces[graphIndex]
                    for pivot in range(graphIndex, self.N.__len__()):
                        if connectedComponentIndeces[pivot] == aux:
                            connectedComponentIndeces[pivot] = connectedComponentIndeces[neighbourIndex]
        uniqueConnectedComponentIndeces = functools.reduce(lambda cCI, index: cCI.append(index) or cCI if index not in cCI else cCI, connectedComponentIndeces, [])
        connectedComponents = []
        for connectedComponentIndex in uniqueConnectedComponentIndeces:
            connectedComponent = []
            for index in range(connectedComponentIndeces.__len__()):
                if connectedComponentIndex == connectedComponentIndeces[index]:
                    connectedComponent.append(self.N[index])
            connectedComponents.append(connectedComponent)
        return uniqueConnectedComponentIndeces.__len__(), connectedComponents

    def fit(self, trainingX, numberEpochs, numClusters):
        self.A = tf.Variable(tf.random.normal([2, trainingX.shape[1]], 0.0, 1.0, dtype=tf.float32)) # Creacion 2 nodos aleatorios
        self.N.append(Graph(0))
        self.N.append(Graph(1))
        self.error_ = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
        epoch = 0
        numberProcessedRow = 0
        while epoch < numberEpochs and self.A.shape[0] < self.maxNumberUnits: # Condicion de parada: epochs y numero maximo nodos ( añadir numero de componentes)
                shuffledTrainingX = tf.random.shuffle(trainingX) # Selección muestra aleatoria xi
                for row_ in tf.range(shuffledTrainingX.shape[0]):
                    xi = shuffledTrainingX[row_]
                    # Selección 2 nodos más cercanos a la muestra
                    indexNearestUnit = self.findNearestUnit(xi, self.A)
                    self.incrementAgeNeighborhood(indexNearestUnit)
                    indexSecondNearestUnit = self.findSecondNearestUnit(xi, self.A)

                    # Al nodo más cercano se le incrementara el error en su distancia con xi
                    self.error_[indexNearestUnit].assign(self.error_[indexNearestUnit] + tf.math.reduce_sum(tf.math.squared_difference(xi, self.A[indexNearestUnit])))
                    # Mueves el nodo más cercano a xi
                    self.A[indexNearestUnit].assign(self.A[indexNearestUnit] + self.epsilon_a * (xi - self.A[indexNearestUnit]))
                    # También mover sus vecinos a xi
                    for indexNeighbour in self.N[indexNearestUnit].neighborhood:
                        self.A[indexNeighbour].assign(self.A[indexNeighbour] + self.epsilon_n * (xi - self.A[indexNeighbour]))
                    # Si el 2ndo más cercano es vecino del más cercano -> edad arista=0
                    if indexSecondNearestUnit in self.N[indexNearestUnit].neighborhood:
                        self.N[indexNearestUnit].setAge(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].setAge(indexNearestUnit, 0.0)
                    # Crear arista
                    else:
                        self.N[indexNearestUnit].addNeighbour(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].addNeighbour(indexNearestUnit, 0.0)
                    # Checkeo edades aristas en todx el grafo (eliminar aristas>amax)
                    for graph in self.N:
                        graph.pruneGraph(self.a_max)
                    self.pruneA()
                    self.connectedComponents, self.component = self.getGraphConnectedComponents()
                    print( "GrowingNeuralGas::numberUnits: {} - GrowingNeuralGas::numberGraphConnectedComponents: {}".format(self.A.shape[0], self.connectedComponents))

                    # Criterio de parada según número de clusters
                    if self.connectedComponents>=numClusters:
                        break

                    # Cada lambda-iteracion se insertara un nuevo nodo
                    if not (numberProcessedRow + 1) % self.eta:
                        # Se encuentra la unidad con mayor error y su vecino con mayor error
                        indexUnitWithMaxError_ = tf.squeeze(tf.math.argmax(self.error_), 0)
                        indexNeighbourWithMaxError_ = self.findIndexNeighbourMaxError(indexUnitWithMaxError_)

                        # Insertar un nodo en el medio de los anteriores elegidos
                        self.A = tf.Variable(tf.concat([self.A, tf.expand_dims(0.5 * (self.A[indexUnitWithMaxError_] + self.A[indexNeighbourWithMaxError_]), 0)], 0))
                        # Conectar el nuevo nodo y eliminar la arista anterior
                        self.N.append(Graph(self.A.shape[0] - 1,[indexUnitWithMaxError_, indexNeighbourWithMaxError_], [0.0, 0.0]))
                        self.N[indexUnitWithMaxError_].removeNeighbour(indexNeighbourWithMaxError_)
                        self.N[indexUnitWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)
                        self.N[indexNeighbourWithMaxError_].removeNeighbour(indexUnitWithMaxError_)
                        self.N[indexNeighbourWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)

                        # Disminuir error de los nodos anteriores y establecer uno para el nuevo
                        self.error_[indexUnitWithMaxError_].assign(self.error_[indexUnitWithMaxError_] * self.alpha)
                        self.error_[indexNeighbourWithMaxError_].assign(self.error_[indexNeighbourWithMaxError_] * self.alpha)
                        self.error_ = tf.Variable(tf.concat([self.error_,  tf.expand_dims(self.error_[indexUnitWithMaxError_], 0)], 0))

                    self.error_.assign(self.error_ * self.delta)
                    numberProcessedRow += 1

                epoch += 1
                print("GrowingNeuralGas::epoch: {}".format(epoch))
        print("FIT HAS ENDED")
        self.connectedComponents, self.component = self.getGraphConnectedComponents() # Analisis de la topologia una vez ha finalizado
        # Cluster.clusters(self.component,self.N, self.A)
        GrowingNeuralGasPlotter.plotGraphConnectedComponent(
            r'C://Users//marti//PycharmProjects//Growing Neural Gas//ImagesTest1',
            'graphConnectedComponents_' + '{}_{}'.format(self.A.shape[0], self.connectedComponents),
            self.A,
            self.N, self.component)

    def predict(self, sample): # Dada una entrada y una topología aprendida se debe devolver el cluster donde se encuentra
        indexNearestUnit = self.findNearestUnit(sample, self.A)
        return self.N[indexNearestUnit].getClusterId()

    def GrowingNeuralGasSaver(self):
        self.N = str(self.N)
        self.A = str(self.A)
        n_file = open(r"C:\Users\YO\PycharmProjects\GNG\grafo.txt", "w")
        a_file = open(r"C:\Users\YO\PycharmProjects\GNG\puntos.txt", "w")
        for row in self.A:
            a_file.write(self.A + "\n")
        for row in self.N:
            n_file.write(self.N + "\n")
        a_file.close()
        n_file.close()

    def GrowingNeuralGasLoader(self):
        filename1 = 'grafo.txt'
        filename2 = 'puntos.txt'
        with open(filename1) as file:
            self.N = [[str(digit) for digit in line.split()] for line in file]

        with open(filename2) as file:
            self.A = tf.convert_to_tensor([[str(digit) for digit in line.split()] for line in file])
