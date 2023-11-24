import cv2
import os
import pandas as pd

class DataPreparation:
    def __init__(self, videos_path,save_path,labels_path):
        self.videos_path=videos_path
        self.save_path=save_path
        self.labels_path=labels_path
    
    def cutting_image(self,img):
     #convert into grayscale
     to_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     #binary thresholding
     ret,threshold1=cv2.threshold(to_gray,15,255,cv2.THRESH_BINARY)
     #apply median blur
     #find contours in the image
     threshold1=cv2.medianBlur(threshold1,19)
     #find edges
     x=threshold1.shape[0]
     y=threshold1.shape[1]

     edges_x=[]
     edges_y=[]
     for i in range(x):
        for j in range(y):
            if threshold1.item(i,j)!=0:
                edges_x.append(i)
                edges_y.append(j)
     #if no edges foud return original image
     if not edges_x:
        return img
     #bounding box
     left=min(edges_y) 
     right=max(edges_y)
     bottom=min(edges_x)
     top=max(edges_x)
     #crop image
     out_img=img[bottom:top,left:right]
     return out_img
    
    def get_frames(self):
     video_path=sorted(os.listdir(self.videos_path))

     for videos in video_path:
        video_path=self.videos_path+videos
        video_name=videos[:2]


        if not os.path.exists(self.save_path+video_name):
            os.mkdir(self.save_path+video_name)
        #load labels
        labels_file_path=os.path.join(self.labels_path, f"label_{video_name}.txt")
        labels_df=pd.read_csv(labels_file_path,delim_whitespace=True)
        #open video file
        cap=cv2.VideoCapture(video_path)
        frame_num=0
        while cap.isOpened():
            ret,frame=cap.read()
            #if no more frames available
            if not ret:
                break
            #process every 25th frame
            if frame_num%25==0:
               
                #resize frame
                dim=(int(frame.shape[1]/frame.shape[0]*300),300)
                frame=cv2.resize(frame,dim,cv2.INTER_AREA)
                cut_img=self.cutting_image(frame)
                #resize result
                result=cv2.resize(cut_img,(250,250),cv2.INTER_AREA)
                #get label for current frame
                label=labels_df.iloc[frame_num//25]['Phase']
                #save result
                #path to save frame
                img_save_path=os.path.join(self.save_path,video_name,f"{frame_num//25+1:04d}.jpg")
                cv2.imwrite(img_save_path,result)
                print(f"Saved frame {img_save_path} with label {label} ret {ret}")
                cv2.waitKey(1)
            frame_num+=1
        cap.release()
        print("Video {:s}: Totally have {:d} frames".format(video_name, frame_num))   