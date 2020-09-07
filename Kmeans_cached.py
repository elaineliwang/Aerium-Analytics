from sklearn.cluster import KMeans
import joblib
import os
import numpy as np
import cv2

class KMeans_Cached:

    def __init__(self, K, filename):
        self.joblib_file = "Cached_Models/"+filename.split('/')[-1].split('.')[0]+"_"+str(K)+".pkl"

        self.K = K
        self.img = cv2.imread(filename)
        self.img_data = np.array(self.img)
        self.n = self.img_data.shape[0]
        if os.path.isfile(self.joblib_file):
            self.model = joblib.load(self.joblib_file)
            self.model_exists=True
        else:
            self.model = KMeans(n_clusters=K)
            self.model_exists=False

    def fit(self):
        if not self.model_exists:
            pixels = self.img_data.reshape((-1, 3))
            self.model.fit(pixels)

            os.makedirs(os.path.dirname(self.joblib_file), exist_ok=True)
            joblib.dump(self.model, self.joblib_file)
            self.model_exists=True

    def output_image(self):
        labels=self.model.labels_
        values=self.model.cluster_centers_.squeeze()

        output_name = "Output/"+self.joblib_file.split('/')[-1].split('.')[0]+".jpg"
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        cv2.imwrite(output_name, values[labels].reshape((self.n,self.n,3)))

    def output_mask(self, color):
        labels=self.model.labels_

        color_label = self.model.predict(color.reshape((1, -1)))
        mask = (labels==color_label).reshape((self.n,self.n,1))

        output_name = "Output/"+self.joblib_file.split('/')[-1].split('.')[0]+"mask.jpg"
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        cv2.imwrite(output_name, self.img_data*mask)
