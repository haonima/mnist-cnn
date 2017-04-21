import tensorflow as tf
from PIL import Image
import numpy as np
import os


sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def gen_data_array(pic_dir):
    res=[]
    for i in os.listdir(pic_dir):
        im = Image.open(os.path.join(pic_dir,i))
        img=np.array(im)
        img=(255-img)/255.0
        res.append(img)
    np.save('train_data',res)

def next_batch(n):
    pass

if __name__ == '__main__':
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x,[-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    b_conv2 = bias_variable([64])

    W_conv2 = weight_variable([3,3,32,64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


