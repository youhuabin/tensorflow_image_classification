'''

@author: YHB
'''
from skimage import io
import numpy as np
import tensorflow as tf
#import Image
if __name__ == '__main__':
    file_path = "../image/face/dlrb/dlrb103.jpg"
    #img=io.imread(file_path)
    #img = np.asarray(img,np.float32)
    #img = Image.fromarray(img)
    #io.imsave("../image/face/dlrb/dlrb103_bak.jpg", img)
    '''
    sess = tf.InteractiveSession()
    
    file_contents = tf.read_file(file_path)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image_array = image.eval()  #[height, width, channels]
        
    sess.close()
    '''