from sklearn import metrics

class EMSegmentation:
    
    def __init__(self):
        pass
    
    def elbow_analysis(self):
        pass
    
    def check_silhouette_score(self, scaled_data_frame, km_labels):
        self.scaled_data_frame = scaled_data_frame
        self.km_labels = km_labels
        return metrics.silhouette_score(scaled_data_frame, km_labels)
    