import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
import os
import yaml


CONFIG_PATH = "/home/office5/Image_Code/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config("config.yaml")
# print(config['measure_pixel'])


image = "/home/office5/Image_Code/Ketan_sir/data/Anna2.bmp"
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]

t_lower = 100  # Lower Threshold
t_upper = 200  # Upper threshold
aperture_size = 7  # Aperture size
L2Gradient = True
edge = cv2.Canny(thresh, t_lower, t_upper, L2gradient = L2Gradient)
y, x = edge.shape

n = 8
start_x = 2
start_y = 7
edge1 = edge[start_y * y // n: (start_y + 1) * y // n, start_x * x // n:(start_x + 1) * x // n]
output = cv2.connectedComponentsWithStats(edge1, 4, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output


# loop over the number of unique connected component labels
for i in range(0, numLabels):
    if i == 0:
        text = "examining component {}/{} (background)".format(
            i + 1, numLabels)
    # otherwise, we are examining an actual connected component
    else:
        text = "examining component {}/{}".format( i + 1, numLabels)
    x_ = stats[i, cv2.CC_STAT_LEFT]
    y_ = stats[i, cv2.CC_STAT_TOP]
    w_ = stats[i, cv2.CC_STAT_WIDTH]
    h_ = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

    keepWidth = w_ > 2
    keepHeight = h_ > 2
    keepArea = area > 30

    if all((keepWidth, keepHeight, keepArea)):
        componentMask = (labels == i).astype("uint8") * 255
        cv2.imwrite(f"/home/office5/Image_Code/try_connected_component/{i}.png",componentMask)
        

get_min_max_hei_wid_value = []

for img1 in glob.glob('/home/office5/Image_Code/try_connected_component/*.png'):
    if img1 != '/home/office5/Image_Code/try_connected_component/0.png':
        img_name = img1.split('/')[-1]
        img = cv2.imread(img1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        for i in range(len(corners)):
            for j in range(1,config['measure_pixel']):
                hei_inc = int(np.int0(corners[i])[1]) + j
                wid_inc = int(np.int0(corners[i])[0]) + j
                hei_dec = int(np.int0(corners[i])[1]) - j
                wid_dec = int(np.int0(corners[i])[0]) - j
                same_hei = int(np.int0(corners[i])[1])
                same_wid = int(np.int0(corners[i])[0])

                check_extreme_point = []
                try:
                    if(gray[same_hei][same_wid]!=255):
                        break
                    else:
                        if( (gray[hei_inc][same_wid]!=255 or gray[same_hei][wid_inc]!=255) and (gray[hei_dec][same_wid]!=255 or gray[same_hei][wid_dec]!=255) and (gray[hei_dec][same_wid]!=255 or gray[same_hei][wid_inc]!=255) and (gray[hei_inc][same_wid]!=255 or gray[same_hei][wid_dec]!=255)):
                            break
                        else:
                            check_extreme_point.append(1)        
                except:
                    print("out of image")

            if(len(check_extreme_point)==1):
                # gray[same_hei][same_wid]=128
                # ori_img_loc_y = start_y * y // n + same_hei
                # ori_img_loc_x = start_x * x // n  + same_wid
                # cv2.imwrite(f"/home/office5/Image_Code/extremen_point/out_put{img_name}",gray)

                points = []
                points.append(img_name)
                # For decrement height and width value
                lw,th,rw,bh=1,1,1,1

                # For storing count of every side pixel value such as left-width,right-width,top-height,bottom-height
                countlw,countth,countrw,countbh=0,0,0,0

                # For left width count store value
                wid_dec = same_wid- lw
                while(gray[same_hei][wid_dec]==255):
                    wid_dec = same_wid - lw
                    countlw+=1
                    lw+=1
                points.append(countlw)

                # For top height count store value
                hei_dec = same_hei- th
                while(gray[hei_dec][same_wid]==255):
                    hei_dec = same_hei - th
                    countth+=1
                    th+=1
                points.append(countth)

                # For right width count store value
                wid_inc = same_wid + rw
                while(gray[same_hei][wid_inc]==255):
                    wid_inc = same_wid + rw
                    countrw+=1
                    rw+=1
                points.append(countrw)

                # For bottom height count store value
                hei_inc = same_hei + bh
                while(gray[hei_inc][same_wid]==255):
                    hei_inc = same_hei + bh
                    countbh+=1
                    bh+=1
                points.append(countbh)

                get_min_max_hei_wid_value.append(points)

print("point",get_min_max_hei_wid_value)

























































# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import glob
# import time
# import os
# import yaml


# CONFIG_PATH = "/home/office5/Image_Code/"

# # Function to load yaml configuration file
# def load_config(config_name):
#     with open(os.path.join(CONFIG_PATH, config_name)) as file:
#         config = yaml.safe_load(file)

#     return config

# config = load_config("config.yaml")
# print(config['measure_pixel'])


# image = "/home/office5/Image_Code/Ketan_sir/data/Anna2.bmp"
# img = cv2.imread(image)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]

# # # Setting All parameters
# t_lower = 100  # Lower Threshold
# t_upper = 200  # Upper threshold
# aperture_size = 7  # Aperture size
# L2Gradient = True
    
# # # Applying the Canny Edge filter
# # # with Custom Aperture Size
# edge = cv2.Canny(thresh, t_lower, t_upper, L2gradient = L2Gradient)
# y, x = edge.shape

# n = 8
# start_x = 2
# start_y = 7
# edge1 = edge[start_y * y // n: (start_y + 1) * y // n, start_x * x // n:(start_x + 1) * x // n]
# # print(start_y * y // n, (start_y + 1) * y // n, start_x * x // n ,(start_x + 1) * x // n)
# # print(edge)
# # print(edge1.shape)
# # show(edge1)

# output = cv2.connectedComponentsWithStats(edge1, 4, cv2.CV_32S)
# (numLabels, labels, stats, centroids) = output


# # loop over the number of unique connected component labels
# for i in range(0, numLabels):
#     if i == 0:
#         text = "examining component {}/{} (background)".format(
#             i + 1, numLabels)
#     # otherwise, we are examining an actual connected component
#     else:
#         text = "examining component {}/{}".format( i + 1, numLabels)
#     # print a status message update for the current connected
#     # component
#     # print("[INFO] {}".format(text))
#     # extract the connected component statistics and centroid for
#     # the current label
#     x_ = stats[i, cv2.CC_STAT_LEFT]
#     y_ = stats[i, cv2.CC_STAT_TOP]
#     w_ = stats[i, cv2.CC_STAT_WIDTH]
#     h_ = stats[i, cv2.CC_STAT_HEIGHT]
#     area = stats[i, cv2.CC_STAT_AREA]
#     (cX, cY) = centroids[i]

#     keepWidth = w_ > 2
#     keepHeight = h_ > 2
#     keepArea = area > 30
#     # ensure the connected component we are examining passes all
#     # three tests
#     if all((keepWidth, keepHeight, keepArea)):
#         # print("[INFO] keeping connected component '{}'".format(i))
#         componentMask = (labels == i).astype("uint8") * 255
#         cv2.imwrite(f"/home/office5/Image_Code/try_connected_component/{i}.png",componentMask)
	

# get_extreme_point = []
# for img1 in glob.glob('/home/office5/Image_Code/try_connected_component/*.png'):
#     if img1 != '/home/office5/Image_Code/try_connected_component/0.png':
#         img_name = img1.split('/')[-1]
#         img = cv2.imread(img1)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         gray = np.float32(gray)
#         dst = cv2.cornerHarris(gray,2,3,0.04)
#         ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
#         dst = np.uint8(dst)

#         ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#         corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#         res = np.hstack((centroids,corners))
#         res = np.int0(res)

#         img[res[:,3],res[:,2]] = [0,255,0]


#         for i in range(len(corners)):
#             for j in range(1,config['measure_pixel']):
#                 hei_inc = int(np.int0(corners[i])[1]) + j
#                 wid_inc = int(np.int0(corners[i])[0]) + j
#                 hei_dec = int(np.int0(corners[i])[1]) - j
#                 wid_dec = int(np.int0(corners[i])[0]) - j
#                 same_hei = int(np.int0(corners[i])[1])
#                 same_wid = int(np.int0(corners[i])[0])

#                 l = []
#                 try:
#                     if(gray[same_hei][same_wid]!=255):
#                         break
#                     else:
#                         if( (gray[hei_inc][same_wid]!=255 or gray[same_hei][wid_inc]!=255) and (gray[hei_dec][same_wid]!=255 or gray[same_hei][wid_dec]!=255) and (gray[hei_dec][same_wid]!=255 or gray[same_hei][wid_inc]!=255) and (gray[hei_inc][same_wid]!=255 or gray[same_hei][wid_dec]!=255)):
#                             break
#                         else:
#                             l.append(1)        
#                 except:
#                     print("out of image")

#             if(len(l)==1):
#                 # print(img_name)
#                 # print(same_hei,same_wid,)
#                 # gray[same_hei][same_wid]=128
#                 # ori_img_loc_y = start_y * y // n + same_hei
#                 # ori_img_loc_x = start_x * x // n  + same_wid
#                 get_extreme_point.append([same_hei, same_wid,img_name])
#                 # get_extreme_point.append([ori_img_loc_y, ori_img_loc_x])
#                 # zip(ori_img_loc_y,ori_img_loc_x)
#                 # print(get_extreme_point)
#                 cv2.imwrite(f"/home/office5/Image_Code/extremen_point/out_put{img_name}",gray)


# for i in range(len(get_extreme_point)):
#     path = f'/home/office5/Image_Code/extremen_point/out_put{get_extreme_point[i][2]}' 
#     # print(path) 
#     img_name = path.split('/')[-1]
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     l = []

#     same_hei = get_extreme_point[i][0]
#     same_wid = get_extreme_point[i][1]
#     hei_dec = get_extreme_point[i][0]
#     hei_inc = get_extreme_point[i][0]
#     wid_dec = get_extreme_point[i][1]
#     wid_inc = get_extreme_point[i][1]

#     print(img_name,same_hei,same_wid)
#     j=1
#     count=0
#     while(gray[same_hei][wid_dec]==255):
#         wid_dec = get_extreme_point[i][1] - j
#         count+=1
#         j+=1
#     l.append(count)

#     j=1
#     count=0
#     while(gray[hei_dec][same_wid]==255):
#         hei_dec = get_extreme_point[i][0] - j
#         count+=1
#         j+=1
#     l.append(count)


#     j=1
#     count=-1
#     while(gray[same_hei][wid_inc]==255):
#         wid_inc = get_extreme_point[i][1] + j
#         count+=1
#         j+=1
#     l.append(count)

#     j=1
#     count=-1
#     while(gray[hei_inc][same_wid]==255):
#         hei_inc = get_extreme_point[i][0] + j
#         count+=1
#         j+=1
#     l.append(count)

#             # break
        
#     print(l)
#         # pass






