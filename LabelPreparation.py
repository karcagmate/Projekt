import os
import pandas as pd
import numpy as np


class LabelPrep:

    def __init__(self,labels_path,video_name) -> None:
        self.labels_path=labels_path
        self.video_name=video_name

    def prep_label(self):
        labels_file_path=os.path.join(self.labels_path,f"label_{self.video_name}.txt")
        labels_df=pd.read_csv(labels_file_path,delim_whitespace=True)
        labels_df=labels_df.drop(columns=('Frame'),axis=1)
        return labels_df

import cv2
import os
class Train_Test:
    def __init__(self,file_path) -> None:
        self.file_path=file_path

        
    def into_array(self):
        file_list=os.listdir(self.file_path)
        images=[]
        for file_name in file_list:
            file=os.path.join(self.file_path,file_name)
            image=cv2.imread(file)
            images.append(image)
        images=np.array(images)
        return images
    

        

