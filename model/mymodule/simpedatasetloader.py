import numpy as np
import cv2
import os

class SimpleDatasetLoader:

    def __init__(self,prepocessors = None):
        self.prepocessors = prepocessors
        if self.prepocessors is None:
            self.prepocessors = []
        

    
    def load(self,imagePaths,verbose = 1):

        data =[]
        labels =[]

        for(i,imagePath) in enumerate(imagePaths):

            image = cv2.imread(imagePath)

            label = imagePath.split(os.path.sep)[-2]

            if self.prepocessors != None:
                for p in self.prepocessors:
                    image = p.prepocessors(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        
        return (np.array(data), np.array(labels))