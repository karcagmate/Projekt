import os
import pandas as pd
import numpy as np
#import cv2

class LabelPrep:

    def __init__(self,labels_path) -> None:
        self.labels_path=labels_path
        #self.video_name=video_name

    def prep_label(self):
        all_labels=[]
        labels_path=sorted(os.listdir(self.labels_path))
        for label_file in labels_path:
            labels_file_path=os.path.join(self.labels_path,label_file)
            labels_df=pd.read_csv(labels_file_path,delim_whitespace=True)
            labels_df=labels_df.drop(columns=['Frame'])
            all_labels.append(labels_df)
        combined=pd.concat(all_labels,ignore_index=True)
        return combined


        

import cv2
import os
class Train_Test:
    def __init__(self,file_path) -> None:
        self.file_path=file_path

        
    def into_array(self):
        all_frames = []
        for video_folder in sorted(os.listdir(self.file_path)):
            video_path = os.path.join(self.file_path, video_folder)
            
            frame_files = sorted(os.listdir(video_path))
            for frame_file in frame_files:
                frame_path = os.path.join(video_path, frame_file)
                image = cv2.imread(frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                all_frames.append(image)
        frames_array = np.array(all_frames)
        return frames_array
    
    

        

