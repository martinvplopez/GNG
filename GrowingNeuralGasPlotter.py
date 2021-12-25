import matplotlib.pyplot as plt

import tensorflow as tf

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, A, N, components):
        figure, axis = plt.subplots()
        x = [A[index][0].numpy() for index in tf.range(A.shape[0])]
        y = [A[index][1].numpy() for index in tf.range(A.shape[0])]
        colors = ["b", "g", "k", "r", "m", "c", "y"]
        clusterid = 0
        for cluster in components:
            print("Cluster", clusterid, "color:", colors[clusterid])
            for node in cluster:
                node.setClusterId(clusterid)
                id = node.id
                point = A.numpy()[id]
                x = point[0]
                y = point[1]
                graphZero = axis.scatter(x, y, color=colors[clusterid], marker='.')
                nameFigure='{}_{}'.format(clusterid,id)
                figure.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
                #print(node.getClusterId())
            clusterid += 1

        plt.close(figure)