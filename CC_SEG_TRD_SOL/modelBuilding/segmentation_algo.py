from sklearn.cluster import KMeans

class DistBasedAlgo:
    
    def __init__(self):
        pass
    
    def k_means(self, df_scaled, cluster):
        self.df_scaled = df_scaled
        self.cluster = cluster
        km = KMeans(n_clusters=cluster, random_state=123).fit(df_scaled)
        return km
        