#coding:utf-8
'''
Created on 2018年2月1日

@author: YHB
'''

from PIL import  Image
import os
import numpy as np
import tensorflow as tf

# 彩色图片转为黑白图片(3通道转换为1通道)
def rgb2gray(rgb):
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #return gray
    height,width,_ = rgb.shape[0],rgb.shape[1],rgb.shape[2] #[height,width,channel]
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    gray = gray.reshape(height,width,1)
    return gray

def getImages(dstPath):
    files = []
    for filename in os.listdir(dstPath):
        dstFile=os.path.join(dstPath,filename)
        if os.path.isfile(dstFile):
            files.append(dstFile)
        if os.path.isdir(dstFile):
            files_ = getImages(dstFile)
            files.extend(files_)
    return files

def convertImageToGray(dstPath):
    files = getImages(dstPath)
    #file_contents = tf.placeholder(tf.float32, [120,160,3], name="image_content")
    #decode_png = tf.image.decode_png(file_contents, channels=3)
    #encode_png = tf.image.encode_png(decode_png)
    with tf.Session() as sess:
        for filename in files:
            file_contents = tf.read_file(filename)
            decode_png = tf.image.decode_png(file_contents, channels=3)
            #encode_png = tf.image.encode_png(decode_png)
            image_array = decode_png.eval()
            gray = rgb2gray(image_array)
            encode_png = tf.image.encode_png(gray)
            f = open(filename, "wb+")
            f.write(encode_png.eval())
            f.close()
    
#图片压缩批处理 
def compressImage(srcPath,dstPath):  
    start = srcPath.rindex("/") + 1
    end = len(srcPath)
    dir_name = srcPath[start:end]
    
    test_set_num = 40 
    for filename in os.listdir(srcPath):  
        #拼接完整的文件或文件夹路径
        srcFile= srcPath + "/" + filename #os.path.join(srcPath,filename)
        
        #如果是文件就处理
        if os.path.isfile(srcFile):    
            if test_set_num >0:
                dstPath_ = dstPath + "/test/" + dir_name + "/"
            else:
                dstPath_ = dstPath + "/train/" + dir_name + "/"
            #如果不存在目的目录则创建一个，保持层级结构
            if not os.path.exists(dstPath_):
                    os.makedirs(dstPath_)
                    
            dstFile= dstPath_ + "/" + filename #os.path.join(dstPath,filename)
            try: 
                #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
                sImg=Image.open(srcFile)  
                #w,h=sImg.size  
                #dImg=sImg.resize((int(w/2),int(h/2)),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
                dImg=sImg.resize((200,260),Image.ANTIALIAS) #[width,height]
                dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
                print(dstFile+" compressed succeeded")
                test_set_num = test_set_num - 1
            except Exception as e:
                print(dstFile + " compressed failed")
                if os.path.exists(dstFile):
                    os.remove(dstFile)
                print(e)

        #如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstPath)

if __name__=='__main__':  
    #compressImage("../../../star_image/crawl","../../../star_image_dst/crawl")
    convertImageToGray("../../../star_image_dst/crawl")