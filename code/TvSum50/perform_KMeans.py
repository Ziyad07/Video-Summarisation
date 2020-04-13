import numpy as np
from sklearn.cluster import KMeans

class perform_KMeans(object):

    def perform_Kmeans(self, points):
        #create k-means object
        kmeans = KMeans(n_clusters=4)
        # fit kmeans object to data
        kmeans.fit(points)
    
        return kmeans

    def samplePointsfromCluster(self, videoNumpy, frameTargets, amount_to_sample):
        
        videoNumpy = videoNumpy / 255.0
        number_of_frames = videoNumpy.shape[0]
        numberofPixels = [number for number in videoNumpy.shape]
        print numberofPixels
        pixels = videoNumpy.flatten().reshape(number_of_frames, 62720) # try generalise this number 
        kmeans = self.perform_Kmeans(pixels)
        #kmeans = perform_Kmeans(principalComponents)
        
        indicies_per_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
        
        indicies = np.array([], dtype=np.int32)
        for i in range(len(indicies_per_cluster)):
            num_samples = int(amount_to_sample * len(indicies_per_cluster[i]))
            idx = np.random.permutation(len(indicies_per_cluster[i]))
            index = [x for x in indicies_per_cluster[i][idx]]
            indicies = np.concatenate((indicies,index[:num_samples]))
        
        videoNumpy = videoNumpy[indicies]
        frameTargets = frameTargets[indicies]
    
        return videoNumpy, frameTargets










