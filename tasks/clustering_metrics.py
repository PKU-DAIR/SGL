import matplotlib
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.manifold import TSNE

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            #            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')

        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore

    @staticmethod
    def plot(X, fig, col, size, true_labels):
        ax = fig.add_subplot(1, 1, 1)
        for i, point in enumerate(X):
            ax.scatter(point[0], point[1], lw=0, s=size, c=col[true_labels[i]])

    def plotClusters(self, tqdm, hidden_emb, true_labels):
        tqdm.write('Start plotting using TSNE...')
        # Doing dimensionality reduction for plotting
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(hidden_emb)
        # Plot figure
        fig = plt.figure()
        self.plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 40, true_labels)
        plt.axis("off")
        fig.savefig("plot.png", dpi=120)
        tqdm.write("Finished plotting")
