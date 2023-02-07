import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples


class PCA_Analysis(object):

    def __init__(self, dataset, standarization, method_PCA):
        self.dataset = dataset
        self.standarization = standarization
        self.method_PCA = method_PCA

    def preprocess(self):
        if self.standarization == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(self.dataset)
        df_preprocess = scaler.transform(self.dataset)
        return scaler, df_preprocess

    def pca(self, dataset_preprocess):
        pca = PCA()
        pca.fit_transform(dataset_preprocess)
        return pca, dataset_preprocess

    def optimal_PCA(self, dataset_preprocess, pca):
        if self.method_PCA == 'eigenvalues':
            S = np.cov(dataset_preprocess.T)
            autovalores, autovectores = np.linalg.eigh(S)
            optCP = pd.DataFrame({'Number of CP': list(range(1, len(autovalores) + 1)),
                                  'autovalores': sorted(autovalores, reverse=True)})
            optCP['mean_autovalores'] = autovalores.mean()
            nCP = np.count_nonzero(sorted(autovalores, reverse=True) > autovalores.mean())
        else:  # % variabilidad
            optCP = pd.DataFrame(
                {'Number of components': list(range(1, len(pca.explained_variance_ratio_.cumsum()) + 1)),
                 '% Variabilidad': pca.explained_variance_ratio_.cumsum()})
            nCP = np.count_nonzero(pca.explained_variance_ratio_.cumsum() <= int(self.method_PCA))
        return optCP, nCP

    def train(self, nCP, dataset_preprocess):
        pca = PCA(n_components=nCP)
        scores_pca = pca.fit_transform(dataset_preprocess)
        scores_pca = pd.DataFrame(scores_pca, columns=['Comp' + str(i) for i in range(1, len(scores_pca[0]) + 1)])
        scores_pca.index = self.dataset.index
        corr_var_PCA = pca.components_.T * np.sqrt(pca.explained_variance_)
        corr_var_PCA = pd.DataFrame(corr_var_PCA, columns=['Comp' + str(i) for i in range(1, len(corr_var_PCA[0]) + 1)])
        corr_var_PCA.index = self.dataset.columns
        return pca, scores_pca, corr_var_PCA


class KMEANS(object):

    def __init__(self, scores, max_clusters, method_kMeans):
        self.scores = scores
        self.max_clusters = 10 #max_clusters
        self.method_kMeans = method_kMeans

    def optimal_kMeans(self):
        dict_metric_cluster = dict()
        for n_cluster in range(2, int(self.max_clusters) + 1):
            kmeans = KMeans(n_clusters=n_cluster, random_state=12345).fit(self.scores)
            centroids = kmeans.cluster_centers_
            assignments = kmeans.labels_
            if self.method_kMeans == 'silhouette':
                metric = silhouette_score(self.scores, assignments)
            else:
                metric = kmeans.inertia_
            dict_metric_cluster[n_cluster] = metric
        if self.method_kMeans == 'silhouette':
            nClusters = max(dict_metric_cluster, key=dict_metric_cluster.get)
        else:
            nClusters = max(dict_metric_cluster, key=dict_metric_cluster.get)
        return dict_metric_cluster, nClusters

    def train_kMeans(self, nClusters):
        kmeans = KMeans(n_clusters=nClusters, random_state=12345).fit(self.scores)
        df_original_final = pd.DataFrame({'cluster': kmeans.labels_}, index=pd.DataFrame(self.scores).index)
        return kmeans, df_original_final


def main_cluster(data, signal):
    try:
        # Pivot data
        pivot_data = pd.pivot_table(data, index='cntid', columns='period', values=signal, aggfunc='median')

        if len(pivot_data) <= 2:
            return {'code': 200, 'message': 'Hay muy pocos contadores.'}

        # Principal Component Analysis
        pca_obj = PCA_Analysis(pivot_data, 'standard', 'eigenvalues')
        scaler, df_preprocess_std = pca_obj.preprocess()
        pca, df_preprocess_pca = pca_obj.pca(df_preprocess_std)
        dataCP, nCP = pca_obj.optimal_PCA(df_preprocess_pca, pca)
        pca, scores_pca, corr_var_PCA = pca_obj.train(nCP, df_preprocess_std)

        # Algoritmo KMeans
        kmeans_obj = KMEANS(scores_pca, len(scores_pca) - 1, 'silhouette')
        ## Numero optimo de clusters
        summary_kMeans, nClusters = kmeans_obj.optimal_kMeans()
        ## Entrenamiento
        kmeans, df_predict = kmeans_obj.train_kMeans(nClusters)


        # Second clusterization if needed
        if (df_predict.value_counts(normalize=True)[0] > 0.8):
            # Cluster with more than the 80% of the data
            cl_full = df_predict.value_counts(normalize=True).reset_index()['cluster'][0]
            # Contadores list
            cnt_list = list(df_predict[df_predict['cluster'] == cl_full].index)
            # Algoritmo KMeans
            kmeans_obj2 = KMEANS(scores_pca[scores_pca.index.isin(cnt_list)], len(scores_pca[scores_pca.index.isin(cnt_list)]) - 1, 'silhouette')
            ## Numero optimo de clusters
            summary_kMeans, nClusters = kmeans_obj2.optimal_kMeans()
            ## Entrenamiento
            kmeans2, df_predict2 = kmeans_obj2.train_kMeans(nClusters)
            ## Merge both results
            df_predict['cluster'] = df_predict['cluster'] + df_predict2['cluster'].max()
            df_predict.update(df_predict2)
            df_predict['cluster'] = df_predict['cluster'].astype(int)

        ## Mergear la prediccion y los perfiles
        df_final = pd.merge(pivot_data, df_predict, left_index=True, right_index=True)

        # print(pivot_data)
        # print(df_predict)

        return {'code': 200, 'message': 'OK', 'data': df_final}

    except Exception as e:
        return {'code': 500, 'message': 'Error in cluster main. Details: '+str(e)}

