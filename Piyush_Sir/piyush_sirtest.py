import cv2
import pandas as pd
import numpy as np


class Master_Img:

    def align(self,ref_x, ref_y, img_x, img_y, img):
        rows,cols = img.shape

        # IF both midx and midy of csv image with reference iamge are same then it will not change anything simply return same image
        if((ref_x==img_x) and (ref_y==img_y)):
            return img

        # If any of the midx or midy change then perform below operation
        else:
            
            diff_x = ref_x - img_x
            diff_y = ref_y - img_y

            # For horizontal rows to be shifted
            if(diff_y > 0):
                crop_img = img[0:rows,(cols-diff_y):cols]
                rem_img = img[0:rows,0:(cols-diff_y)]
                img_h = cv2.hconcat([crop_img, rem_img])
                img = img_h
            elif(diff_y<0):
                diff_y = abs(diff_y)
                crop_img = img[0:rows,0:diff_y]
                rem_img = img[0:rows,diff_y:cols]
                img_h = cv2.hconcat([rem_img,crop_img])
                img = img_h
            else:
                img = img
            
            # For vertical image to be shifted
            if(diff_x>0):
                crop_img = img[0:diff_x,0:cols]
                rem_img = img[diff_x:rows,0:cols]
                im_v = cv2.vconcat([rem_img,crop_img])
                img = im_v
            elif(diff_x<0):
                diff_x = abs(diff_x)
                crop_img = img[(rows-diff_x):rows,0:cols]
                rem_img = img[0:(rows-diff_x),0:cols]
                im_v = cv2.vconcat([crop_img, rem_img])
                img = im_v
            else:
                img = img
            
        return img


    def create_master1(self,excel_path1, img_pth, kernel):
    
        df = pd.read_csv(excel_path1)
        ref_x, ref_y = df['midx'].iloc[0], df['midy'].iloc[0]

        align_img = []
        for i in range(len(df.index)):
            img_name = df['File name'].iloc[i]
            img = f'{img_pth}/{img_name}'
            img_x, img_y = df['midx'].iloc[i], df['midy'].iloc[i]
            gray_img = cv2.imread(img,0)
            gray_img_blur = cv2.medianBlur(gray_img,kernel)
            gray_img_blur = gray_img_blur.astype(np.int8)
            align_img.append(self.align(ref_x,ref_y,img_x,img_y,gray_img_blur))

        hei,wid = align_img[0].shape
        master_img1 = np.zeros([hei, wid])
        N = len(align_img)

        # Getting master image
        for im in align_img:
            master_img1=master_img1 + im / N

        return master_img1
    
    def create_master2(self,excel_path1, img_pth, kernel):
    
        df = pd.read_csv(excel_path1)
        ref_x, ref_y = df['midx'].iloc[0], df['midy'].iloc[0]

        align_img = []
        for i in range(len(df.index)):
            img_name = df['File name'].iloc[i]
            img = f'{img_pth}/{img_name}'
            img_x, img_y = df['midx'].iloc[i], df['midy'].iloc[i]
            gray_img = cv2.imread(img,0)
            gray_img_blur = cv2.medianBlur(gray_img,kernel)
            gray_img_blur = gray_img_blur.astype(np.int8)
            align_img.append(self.align(ref_x,ref_y,img_x,img_y,gray_img_blur))

        img_diff_lst = []
        hei,wid = align_img[0].shape
        for i in range(len(df.index)-1):
            align_img_diff = np.abs(align_img[i] - align_img[i+1])
            img_diff_lst.append(align_img_diff)

        hei,wid = img_diff_lst[0].shape
        master_img2 = np.zeros([hei, wid])
        N = len(img_diff_lst)


        # Getting master image
        for im in img_diff_lst:
            master_img2=master_img2 + im / N

        # cv2.imwrite("master_image1.png",master_img)
        return master_img2


    def get_align_csv2(self,excel_path2,img_pth,kernel,mst_img1,mst_img2):

        df_2 = pd.read_csv(excel_path2)
        ref_x, ref_y = df_2['midx'].iloc[0], df_2['midy'].iloc[0]

        align_img = []
        for i in range(len(df_2.index)):
            img_name = df_2['File name'].iloc[i]
            img = f'{img_pth}/{img_name}'
            img_x, img_y = df_2['midx'].iloc[i], df_2['midy'].iloc[i]
            gray_img = cv2.imread(img,0)
            gray_img_blur = cv2.medianBlur(gray_img,kernel)
            gray_img_blur = gray_img_blur.astype(np.int8)
            align_img.append(self.align(ref_x,ref_y,img_x,img_y,gray_img_blur))
        
            output_val = []
            error_mst1 = mst_img1-align_img[i]
            error_mst2 = np.abs(mst_img2 - error_mst1)

            # print(error_mst2)
            cv2.imwrite(f'{img_pth}/output_{img_name}',error_mst2)


obj = Master_Img()

excel_path1 = '/home/office5/Image_Code/Piyush_Sir/testing.csv'
excel_path2 = '/home/office5/Image_Code/Piyush_Sir/points_3.csv'
img_pth = '/home/office5/Image_Code/Piyush_Sir/Images_p/'
kernel = 3

mst_img1 = obj.create_master1(excel_path1,img_pth,kernel)
mst_img2 = obj.create_master2(excel_path1,img_pth,kernel)

# From csv2 file
obj.get_align_csv2(excel_path2,img_pth,kernel,mst_img1,mst_img2)