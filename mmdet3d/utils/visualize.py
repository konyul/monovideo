import cv2            
import numpy as np
import time
import os

def visualize(x, direc_name):  # input shape : (channel, H, W)
    idx = str(len(os.listdir(direc_name))).zfill(6)
    image_feature = normalize(x)
    #image_1 = cv2.imread(img_metas[0]['filename'], cv2.COLOR_BGR2RGB)
    #cv2.imwrite("pictures/images/"+idx+".jpg",image_1)
    max_image_feature = np.max(np.transpose(image_feature.astype("uint8"),(1,2,0)),axis=2)
    max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
    #cv2.imwrite(f"{direc_name}/{idx}.jpg",max_image_feature)
    cv2.imwrite(f"{direc_name}/{idx}.jpg",max_image_feature)
    

def normalize(image_features):
    image_features = image_features.cpu().numpy()
    image_features -= image_features.mean()
    image_features /= image_features.std()
    image_features *= 64
    image_features += 128
    return image_features
