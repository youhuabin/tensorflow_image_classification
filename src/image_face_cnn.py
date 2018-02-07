#coding:utf-8
'''
Created on 2018年2月1日

@author: YHB
使用卷积神经网络对10位女明星进行分类
'''
import tensorflow as tf
import os
import numpy as np
from builtins import len
import random
import logging
import time

console_log = True
currentTime = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
log_path = "../log"
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename= log_path + '/face_classifier_'+ currentTime + '.log',
                filemode='w')
#输出到控制台
if console_log:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

star_name = {0:"迪丽热巴",1:"范冰冰",2:"关晓彤",3:"李小冉",4:"刘亦菲",5:"马苏",6:"全智贤",7:"杨幂",8:"赵丽颖",9:"Angelababy"}
stars = {"dlrb":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"fbb":[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             "gxt":[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"lxr":[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
             "lyf":[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],"ms":[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
             "qzx":[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],"ym":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
             "zly":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],"zy":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
            }

def read_train_images(num_channel):
    train_path = "../image_dst/face/train/";
    images = []
    categories = []
    sess = tf.InteractiveSession()
    
    file_path = os.path.join(train_path+"dlrb/","dlrb1.jpg")
    file_contents = tf.read_file(file_path)
    image = tf.image.decode_jpeg(file_contents, channels=num_channel)
    image_array = image.eval()  #[height, width, channels]
    images.append(image_array.astype(np.float32))
    categories.append(stars["dlrb"])
    
    file_path = os.path.join(train_path+"fbb/","fbb1.jpg")
    file_contents = tf.read_file(file_path)
    image = tf.image.decode_jpeg(file_contents, channels=num_channel)
    image_array = image.eval()  #[height, width, channels]
    images.append(image_array.astype(np.float32))
    categories.append(stars["fbb"])
    sess.close()
    return images,categories
    

def read_images(dir_path,category,num_channel):
    images = []
    categories = []
    sess = tf.InteractiveSession()
    for filename in os.listdir(dir_path): 
        file_path = os.path.join(dir_path,filename)
        file_contents = tf.read_file(file_path)
        image = tf.image.decode_jpeg(file_contents, channels=num_channel)
        image_array = image.eval()  #[height, width, channels]
        images.append(image_array.astype(np.float32))
        categories.append(category)
        #logging.info("read:" + file_path)
    sess.close()
    return images,categories

def get_images(path,num_channel):
    '''
    获取图片
    '''
    images = []
    labels = []
    
    for k,v in stars.items():
        logging.info("读取 " + k + "图片")
        dir_path = path + k
        images_,labels_ = read_images(dir_path,v,num_channel) 
        images.extend(images_)
        labels.extend(labels_)
        logging.info("读取 " + k + "图片完成")
    return images,labels

def get_batch(batch_size,images,labels):
    max_number = len(images)
    lst_number = random.sample(range(0,max_number),batch_size) 
    batch_images = []
    batch_labels = []
    for i in lst_number:
        batch_images.append(images[i])
        batch_labels.append(labels[i])
    return batch_images,batch_labels

# 创建卷积层默认步幅大小为1
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 创建最大池化层 过滤器默认窗口大小为 2*2
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# 创建网络
def conv_net(x, weights, biases, dropout):
    
    # 第一个卷积层 [Batch Size, Height, Width, Channel]
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # 第一个池化层（下采样）
    pool1 = maxpool2d(conv1, k=2)

    # 第二个卷积层
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # 第二个池化层
    pool2 = maxpool2d(conv2, k=2)
    
    # 第三个卷积层
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    # 第三个池化层
    pool3 = maxpool2d(conv3, k=2)

    # Reshape conv2 output to fit fully connected layer input
    # 全连接层1，将池化层的数据转换成一维
    fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # 保持网络的连接数（为防止过拟合需要适当断开部分连接）
    fc1 = tf.nn.dropout(fc1, dropout) 
    
    # 全连接层2
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # 保持网络的连接数（为防止过拟合需要适当断开部分连接）
    fc2 = tf.nn.dropout(fc2, dropout)
    
    # 输出层，分类预测
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

if __name__ == '__main__':
    
    # 训练参数
    learning_rate = 0.001
    num_steps = 200000
    display_step = 10
    
    # Network Parameters
    num_classes = 10 # 分类数 (0-9数字)
    
    # tf Graph input
    graph_with = 80 #52
    graph_height = 80 #60
    num_channel = 1 #黑白图片 (图片通道数)
    X = tf.placeholder(tf.float32, [None, graph_height,graph_with, num_channel])
    Y = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    
    # 每层的权重与偏置
    feature1_size = 32
    feature2_size = 64
    feature3_size = 128
    full_input_size = int((graph_with//(2*2*2))*(graph_height//(2*2*2))*feature3_size)
    full1_size = 1024
    full2_size = 512
    weights = {
        # 5x5 卷积, 输入1通道(特征Map), 输出32特征Map
        'wc1': tf.Variable(tf.random_normal([5, 5, num_channel, feature1_size])),
        # 5x5 卷积, 输入32特征Map,  输出64特征Map
        'wc2': tf.Variable(tf.random_normal([5, 5, feature1_size, feature2_size])),
        # 5x5 卷积, 输入64特征Map,  输出128特征Map
        'wc3': tf.Variable(tf.random_normal([5, 5, feature2_size, feature3_size])),
        # 全连接层1, 输入节点10*10*128, 1024输出节点
        'wd1': tf.Variable(tf.random_normal([full_input_size, full1_size])),
        # 全连接层2, 输入节点1024, 512输出节点
        'wd2': tf.Variable(tf.random_normal([full1_size, full2_size])),
        # 输出层：输入节点512, 输出节点 2 (类型预测)
        'out': tf.Variable(tf.random_normal([full2_size, num_classes]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([feature1_size])),
        'bc2': tf.Variable(tf.random_normal([feature2_size])),
        'bc3': tf.Variable(tf.random_normal([feature3_size])),
        'bd1': tf.Variable(tf.random_normal([full1_size])),
        'bd2': tf.Variable(tf.random_normal([full2_size])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # 构造网络
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)
    
    # 定义损失函数与优化器
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y),name='loss_op')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,name='train_op')
    
    # 评估模型
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1),name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
    
    # 初始化变量
    init = tf.global_variables_initializer()
    
    # 获取训练集图片
    logging.info("开始获取训练集图片")
    train_path = "../image_dst/face/train/";
    images,labels = get_images(train_path,num_channel)
    logging.info("获取训练集图片结束")
    #train_images,train_labels = read_train_images(num_channel)
    # 获取测试集图片
    logging.info("开始获取测试集图片")
    test_path = "../image_dst/face/test/";
    test_images,test_labels = get_images(test_path,num_channel)
    logging.info("获取测试集图片结束")
    # 模型训练
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, num_steps+1):
            train_batch_images,train_batch_labels = get_batch(100, images,labels)
            test_batch_images,test_batch_labels = get_batch(50, test_images,test_labels)
            sess.run(train_op, feed_dict={X: train_batch_images, Y: train_batch_labels, keep_prob: 0.8})
            if step % display_step == 0 or step == 1:
                # 计算训练集损失与准确率
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_batch_images,Y: train_batch_labels,keep_prob: 1.0})
                # 计算测试集准确率
                test_acc = sess.run(accuracy, feed_dict={X: test_batch_images,Y: test_batch_labels,keep_prob: 1.0})
                #print("Step " + str(step) + ", 训练集损失= " + "{:.4f}".format(loss) + ", 训练集准确率= " + "{:.3f}".format(acc) + ", 测试集准确率= " + "{:.3f}".format(test_acc))
                logging.info("Step %s, 训练集损失= %.4f , 训练集准确率=  %.4f, 测试集准确率= %.4f",str(step),loss,acc,test_acc)
        saver = tf.train.Saver()    
        saver.save(sess,  "../model/face-classifier-model" )
        logging.info("模型训练完成!")
        #correct = sess.run(correct_pred,feed_dict={X: train_images,Y: train_labels,keep_prob: 1.0})
        #print(train_labels)
        #print(correct)
        # 计算测试集准确率
