import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MakeClusters(object):
    """
    INPUT:
    -df_to_cluster(Pandas DataFrame) - This data frame contains different lat/longs that you would like to cluster together
    -threshold(int) - This is the minimum amount of rides that a certain location needs to meet in order to qualify for clustering. Otherwise those rows get deleted.

    OUTPUT:
    -CSV of all points clustered and labeled
    """

    def __init__(self, df_to_cluster, threshold=500, norm=True):
        self.df = df_to_cluster
        self.threshold = threshold
        self.filter_no_rides()
        if norm:
            self.year_total_array = self.get_year_total_array()
            self.normalize()


    def filter_no_rides(self):
        """
        INPUT:
        -None

        OUTPUT:
        -None

        DOC:
        -Removes those labels that don't meet a threshold

        """
        # total for the year, returning those with 500 ridse or more
        x = self.df[['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']]
        self.df = self.df[np.sum(x, axis=1) > self.threshold]

    def get_year_total_array(self):
        """
        INPUT:
        -None

        OUTPUT:
        -Year_total_array(array) - A list of total rides drop off at every latitude and longitude.
        -This will be used for normalizing the data
        """

        self.x = self.df.drop(['lat', 'long'], axis=1)
        year_total_array = (self.x.d0 + self.x.d1 + self.x.d2 + self.x.d3 + self.x.d4 + self.x.d5 + self.x.d6).reshape(len(self.x), 1).astype(float)
        return year_total_array

    def normalize(self):
        """
        INPUT:
        OUTPUT:
        DOC:
        """
        self.x = np.divide(self.x, self.year_total_array)

    def cluster(self, cluster_num=6, random_state=True, return_normed=True):
        """
        INPUT:
        -cluster_num(int) - Number of clusters that you'd like to make
        -random_state(boolean) - If you wanted to have the same output each time.
        -return_normed(boolean) - If you want your df to return normed

        OUTPUT:
        -Normalized or non-normalized df
        """
        if random_state:
            # In case you want the same results each time
            cluster_model = KMeans(n_clusters=cluster_num, random_state=1)
        else:
            cluster_model = KMeans(n_clusters=cluster_num)

        cluster_model.fit(self.x)

        #apply your clustered labels to your original df
        self.df['label'] = cluster.labels_

        #put the lat and long of your original df back on to your normed tabel
        self.x['lat'] = self.df['lat']
        self.x['long'] = self.df['long']
        self.x['label'] = self.df['label']

        #user chooses if they want the normed table or not as the output
        if return_normed:
            return self.x
        else:
            return self.df

    def export_df(self, df, path):
        """
        INPUT:
        -df(Pandas DataFrame) - User defined
        -path(string) - User defined - Where to put the file you're saving

        DOC: Simple little function to save your df to csv
        """
        df.to_csv(path)

    def plot_results(self, df):
        """
        DOC:
        -Plot the clusters you have just made. 
        -More colors are supplied than needed just in case you want to increase the number of clusters.
        """
        colors = ['#3385AD', '#8533FF', '#FF5CD6', '#FF7373', '#FFDB4D', '#B8FF70', '#70DB70', '#335C33', \
                '#75FFFF', '#D6FFD6', '#141A14','#3385AD', '#8533FF', '#FF5CD6', '#3385AD', '#8533FF', '#FF5CD6', \
                '#3385AD', '#8533FF', '#FF5CD6']
        plt.figure(figsize=(12,14))
        for ind, label in enumerate(np.unique(df.label)):
            y = df[df.label == label]
            plt.scatter(y['long'], y.lat, c=colors[ind], s=45, label=label)
        plt.legend()
        plt.show()

    def get_cluster_means(self, df):
        """
        DOC
        -Group your clusters by label and then take the mean. This will get your the distributions for each cluster.
        """
        means_df = df.groupby(['label']).mean()
        return means_df

if __name__ == '__main__':
    df = pd.read_csv('data/manhatten_point_features.csv', index_col=0)
    mc = MakeClusters(df)