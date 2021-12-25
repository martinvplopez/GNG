import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
class Cluster(object):
    @staticmethod
    def clusters(ConnectedComponents, N, A):
        figure, axis = plt.subplots()
        # print(ConnectedComponents)
        # print("A", A.numpy())
        # print("A1", A.numpy()[1])
        colors = ["b", "g", "k", "r", "m", "c", "y"]
        clusterid=0
        for cluster in ConnectedComponents:
            for node in cluster:
                node.setCluster=clusterid
                id=node.id
                point=A.numpy()[id]
                x=point[0]
                y=point[1]
                graphZero = axis.scatter(x, y, color=colors[clusterid], marker='.')
                figure.savefig('C://Users//marti//PycharmProjects//Growing Neural Gas//ImagesTest1' + '//' + 'graphConnectedComponents_' + '{}_{}'.format(len(ConnectedComponents), id) + '.png', transparent=False, dpi=80, bbox_inches="tight")


                # print("Cluster", clusterid, "id",id, "posicion",point, "x", point[0], point[1])
            clusterid+=1
        plt.close(figure)