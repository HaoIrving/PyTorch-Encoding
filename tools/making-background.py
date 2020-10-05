import numpy as np
import os
from scipy.io import loadmat,savemat
import cv2
###
# lightblue label=1
# red label=5
# white label=6
### 
#storage red lightblue white pixels
def find_pixels():
    desired_label=[1,5,6]
    red_pixel=np.zeros((0,3))
    lightblue_pixel=np.zeros((0,3))
    white_pixel=np.zeros((0,3))
    pixel_per=np.zeros((1,3))
    for i in range(400):
        if i%20==0:
            print(i)
        for j in range(512):
            for k in range(512):
                pixel_per[0,0]=j
                pixel_per[0,1]=k
                pixel_per[0,2]=i
                # if label[j,k,i]==1:
                #     lightblue_pixel=np.vstack((lightblue_pixel,pixel_per))
                if label[j,k,i]==5:
                    red_pixel=np.vstack((red_pixel,pixel_per))
                # if label[j,k,i]==6:
                #     white_pixel=np.vstack((white_pixel,pixel_per))
    np.save('C:\\Users\\dell\\Desktop\\裁剪图片原图\\redpixels.npy',red_pixel)
    # np.save('C:\\Users\\dell\\Desktop\\裁剪图片原图\\lightbluepixels.npy',lightblue_pixel)
    # np.save('C:\\Users\\dell\\Desktop\\裁剪图片原图\\whitepixels.npy',white_pixel)
    return
label=loadmat('D:\\learning\\contestprepare\\data\\data\\label.mat')
label=label['label']
find_pixels()
def making_background(path_rep):
    desired_label=[2,3,4]
    list_rep=os.listdir(path_rep)
    red=np.load('C:\\Users\\dell\\Desktop\\裁剪图片原图\\redpixels.npy')
    lightblue=np.load('C:\\Users\\dell\\Desktop\\裁剪图片原图\\lightbluepixels.npy')
    white=np.load('C:\\Users\\dell\\Desktop\\裁剪图片原图\\whitepixels.npy')
    high_red=red.shape[0]
    high_lightblue=lightblue.shape[0]
    high_white=white.shape[0]
    label_new=np.zeros((512,512,200))
    for i in range(len(list_rep)):
        index_red=np.random.randint(0,high_red,1)
        index_lightblue=np.random.randint(0,high_lightblue,1)
        index_white=np.random.randint(0,high_white,1)
        #
        path_HH_red=path_tif+'HH\\'+str(red[index_red,2]+1)+'_HH.tiff'
        path_HV_red=path_tif+'HV\\'+str(red[index_red,2]+1)+'_HV.tiff'
        path_VH_red=path_tif+'VH\\'+str(red[index_red,2]+1)+'_VH.tiff'
        path_VV_red=path_tif+'VV\\'+str(red[index_red,2]+1)+'_VV.tiff'
        red_HH=cv2.imread(path_HH_red,-1)
        red_HV=cv2.imread(path_HV_red,-1)
        red_VH=cv2.imread(path_VH_red,-1)
        red_VV=cv2.imread(path_VV_red,-1)
        #
        path_HH_white=path_tif+'HH\\'+str(red[index_white,2]+1)+'_HH.tiff'
        path_HV_white=path_tif+'HV\\'+str(red[index_white,2]+1)+'_HV.tiff'
        path_VH_white=path_tif+'VH\\'+str(red[index_white,2]+1)+'_VH.tiff'
        path_VV_white=path_tif+'VV\\'+str(red[index_white,2]+1)+'_VV.tiff'
        white_HH=cv2.imread(path_HH_white,-1)
        white_HV=cv2.imread(path_HV_white,-1)
        white_VH=cv2.imread(path_VH_white,-1)
        white_VV=cv2.imread(path_VV_white,-1)
        #
        path_HH_lightblue=path_tif+'HH\\'+str(red[index_lightblue,2]+1)+'_HH.tiff'
        path_HV_lightblue=path_tif+'HV\\'+str(red[index_lightblue,2]+1)+'_HV.tiff'
        path_VH_lightblue=path_tif+'VH\\'+str(red[index_lightblue,2]+1)+'_VH.tiff'
        path_VV_lightblue=path_tif+'VV\\'+str(red[index_lightblue,2]+1)+'_VV.tiff'
        lightblue_HH=cv2.imread(path_HH_lightblue,-1)
        lightblue_HV=cv2.imread(path_HV_lightblue,-1)
        lightblue_VH=cv2.imread(path_VH_lightblue,-1)
        lightblue_VV=cv2.imread(path_VV_lightblue,-1)


        imnum=int(list_rep[i].split('_')[0])-1
        path_tif='D:\\learning\\contestprepare\\data\\data\\'
        path_part=['HH','HV','VH','VV']
        path_HH=path_tif+'HH\\'+str(imnum+1)+'_HH.tiff'
        path_HV=path_tif+'HV\\'+str(imnum+1)+'_HV.tiff'
        path_VH=path_tif+'VH\\'+str(imnum+1)+'_VH.tiff'
        path_VV=path_tif+'VV\\'+str(imnum+1)+'_VV.tiff'
        HH=cv2.imread(path_HH,-1)
        HV=cv2.imread(path_HV,-1)
        VH=cv2.imread(path_VH,-1)
        VV=cv2.imread(path_VV,-1)
        for j in range(512):
            for k in range(512):
                if label[j,k,imnum]==2:
                    HH[j,k]=white_HH[white[index_white,0],white[index_white,1]]
                    HV[j,k]=white_HV[white[index_white,0],white[index_white,1]]
                    VH[j,k]=white_VH[white[index_white,0],white[index_white,1]]
                    VV[j,k]=white_VV[white[index_white,0],white[index_white,1]]
                    label[j,k,imnum]==label[white[index_white,0],white[index_white,1],white[index_white,2]]
                    assert(label[white[index_white,0],white[index_white,1],white[index_white,2]]==6)
                if label[j,k,imnum]==3:
                    HH[j,k]=lightblue_HH[lightblue[index_lightblue,0],lightblue[index_lightblue,1]]
                    HV[j,k]=lightblue_HV[lightblue[index_lightblue,0],lightblue[index_lightblue,1]]
                    VH[j,k]=lightblue_VH[lightblue[index_lightblue,0],lightblue[index_lightblue,1]]
                    VV[j,k]=lightblue_VV[lightblue[index_lightblue,0],lightblue[index_lightblue,1]]
                    label[j,k,imnum]==label[white[index_lightblue,0],white[index_lightblue,1],white[index_lightblue,2]]
                    assert(label[lightblue[index_lightblue,0],lightblue[index_lightblue,1],lightblue[index_lightblue,2]]==1)
                if label[j,k,imnum]==4:
                    HH[j,k]=red_HH[red[index_red,0],red[index_red,1]]
                    HV[j,k]=red_HV[red[index_red,0],red[index_red,1]]
                    VH[j,k]=red_VH[red[index_red,0],red[index_red,1]]
                    VV[j,k]=red_VV[red[index_red,0],red[index_red,1]]
                    label[j,k,imnum]==label[red[index_red,0],white[index_red,1],white[index_red,2]]
                    assert(label[red[index_red,0],red[index_red,1],red[index_red,2]]==5)
        label_new[:,:,i]=label[:,:,imnum]

        path_out_HH=path_tif+'out\\HH\\'+str(501+i)+'_HH.tiff'
        path_out_HV=path_tif+'out\\HV\\'+str(501+i)+'_HV.tiff'
        path_out_VH=path_tif+'out\\VH\\'+str(501+i)+'_VH.tiff'
        path_out_VV=path_tif+'out\\VV\\'+str(501+i)+'_VV.tiff'
        cv2.imwrite(path_out_HH,HH)
        cv2.imwrite(path_out_HV,HV)
        cv2.imwrite(path_out_VH,VH)
        cv2.imwrite(path_out_VV,VV)
    path_mat=path_tif+'out\\label_new.mat'
    savemat(path_mat,label_new)
    return    
        


        

    