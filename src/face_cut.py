#coding:utf-8
'''
Created on 2018年2月6日
截取图片中的人脸
@author: youhuabin
'''

import face_recognition
from PIL import Image
import os

def get_face(src_path,dst_path):
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
                    x = left #- 20
                    y = top #- 20
                    w = right #+ 20
                    h = bottom #+ 20
                    region = im.crop((x, y, w, h))  #(left, upper, right, lower)
                    region.save(dstpath)
                    print(filepath + "   get face successed")
            except Exception as e:
                print(e)
                print(filepath + "   get face failed")
                os.remove(dstpath)
        #循环子目录        
        if os.path.isdir(filepath):
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)
            get_face(filepath,dstpath)
if __name__ == '__main__':
    get_face('../image/crawl/','../image/face/')
   