#coding:utf-8
'''
Created on 2018年2月1日

@author: YHB
'''
from PIL import  Image
import os
import numpy as np
import face_recognition
import logging
import time
import tensorflow as tf

console_log = True
currentTime = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
log_path = "../log"
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename= log_path + '/image_face_process_'+ currentTime + '.log',
                filemode='w')
#输出到控制台
if console_log:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_face(src_path,dst_path):
    '''
    获取图片中的人脸，生成人脸图片
    '''
    for filename in os.listdir(src_path):
        filepath = os.path.join(src_path,filename)
        dstpath = os.path.join(dst_path,filename)
        
        if os.path.isfile(filepath):
            try:
                image_file = face_recognition.load_image_file(filepath)
                loc = face_recognition.face_locations(image_file) #(top, right, bottom, left)
                if len(loc) == 1: #图片上只有一个头像
                    loc = loc[0]
                    top = loc[0]
                    right = loc[1]
                    bottom = loc[2]
                    left = loc[3]
                    im = Image.open(filepath)
                    # 截取图片中脸部
                    region = im.crop((left, top, right, bottom))  #(left, upper, right, lower)
                    region.save(dstpath)
                    logging.info(filepath + "   获取人脸成功")
            except Exception as e:
                logging.exception(e)
                logging.exception(filepath + "   获取人脸失败")
                os.remove(dstpath)
        #循环子目录        
        if os.path.isdir(filepath):
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)
            get_face(filepath,dstpath)
            

def rgb2gray(rgb):
    '''
    灰度处理：彩色图片转为黑白图片(3通道转换为1通道)
    '''
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #return gray
    height,width,_ = rgb.shape[0],rgb.shape[1],rgb.shape[2] #[height,width,channel]
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    gray = gray.reshape(height,width,1)
    return gray

def getImages(dstPath):
    '''
    获取目录下的所有文件路径
    '''
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
    '''
      图片灰度处理
    '''
    files = getImages(dstPath)
    #file_contents = tf.placeholder(tf.float32, [120,160,3], name="image_content")
    #decode_png = tf.image.decode_png(file_contents, channels=3)
    #encode_png = tf.image.encode_png(decode_png)
    with tf.Session() as sess:
        for filename in files:
            file_contents = tf.read_file(filename)
            decode = tf.image.decode_jpeg(file_contents, channels=3)
            #encode_png = tf.image.encode_png(decode_png)
            image_array = decode.eval()
            gray = rgb2gray(image_array)
            encode = tf.image.encode_jpeg(gray)
            f = open(filename, "wb+")
            f.write(encode.eval())
            f.close()
            logging.info("灰度处理:" + filename)
    
def compressImage(srcPath,dstPath,width,height):  
    '''
    图片压缩批处理 ，将图片压缩为指定大小
    '''
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
                dImg=sImg.resize((width,height),Image.ANTIALIAS) #[width,height]
                dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
                logging.info(dstFile+" compressed succeeded")
                test_set_num = test_set_num - 1
            except Exception as e:
                logging.exception(dstFile + " compressed failed")
                if os.path.exists(dstFile):
                    os.remove(dstFile)
                logging.exception(e)

        #如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstPath)

if __name__=='__main__':  
    get_face('../image/crawl/','../image/face/')
    compressImage("../image/face","../image_dst/face",80,80)
    convertImageToGray("../image_dst/face")