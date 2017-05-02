#encoding:utf8
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import local_data
sess = tf.InteractiveSession()



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def upsampling2d(shape):
    shape4D = [batsiz, rows, cols, out_ch]
    linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                                    strides=[1, 2, 2, 1], padding='SAME') + self.b
    return tf.nn.conv2d_transpose()

batch_cur = 0
def next_batch(num):
    global batch_cur


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

x_image = tf.reshape(x, [-1,28,28,1])

#auto encoding
W_ae1 = weight_variable([5,5,1,16])
b_ae1 = bias_variable([16])
h_ae1 = tf.nn.relu(conv2d(x_image,W_ae1) + b_ae1)
h_ae1_pool = max_pool_2x2(h_ae1)

W_ae2 = weight_variable([5,5,1,8])
b_ae2 = bias_variable([8])
h_ae2 = tf.nn.relu(conv2d(h_ae1_pool,W_ae2) + b_ae2)
h_ae2_pool = max_pool_2x2(h_ae2)

W_ae3 = weight_variable([5,5,1,8])
b_ae3 = bias_variable([8])
h_ae3 = tf.nn.relu(conv2d(h_ae2_pool,W_ae3) + b_ae3)

encoded = max_pool_2x2(h_ae3)
#decoding
W_de1 = weight_variable([2,2,1,8])
b_de1 = bias_variable([8])
h_de1 = tf.nn.conv2d_transpose(encoded,W_de1,[tf.shape(W_de1)[0],7,7,8],strides=[1,2,2,1],padding='SAME')+b_de1

W_de1 = weight_variable([2,2,1,8])
b_de1 = bias_variable([8])
h_de1 = tf.nn.conv2d_transpose(encoded,W_de1,[tf.shape(W_de1)[0],7,7,8],strides=[1,2,2,1],padding='SAME')+b_de1

W_de1 = weight_variable([2,2,1,8])
b_de1 = bias_variable([8])
h_de1 = tf.nn.conv2d_transpose(encoded,W_de1,[tf.shape(W_de1)[0],7,7,8],strides=[1,2,2,1],padding='SAME')+b_de1





#conv_1
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#pool_1
h_pool1 = max_pool_2x2(h_conv1)

#conv_2
W_conv2 = weight_variable([5, 5, 64, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#conv_3
W_conv3 = weight_variable([5, 5, 32, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

#pool_2
h_pool2 = max_pool_2x2(h_conv3)

#conv_4
W_conv4 = weight_variable([5, 5, 16, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool2, W_conv4) + b_conv4)

#pool_3
h_pool3 = max_pool_2x2(h_conv4)

#conv_5
W_conv5 = weight_variable([2, 2, 32, 64])
b_conv5 = bias_variable([64])
h_conv5 = tf.nn.relu(conv2d(h_pool3, W_conv5) + b_conv5)

#pool_4
#h_pool4 = max_pool_2x2(h_conv5)



#fully-connected
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

#h_conv5_flat = tf.reshape(h_conv5, [-1, 4 * 4 * 64])
#h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 32])
h_pool2_flat = tf.reshape(h_conv5, [-1, 4 * 4 * 64])
#h_pool2_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
#h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool2_flat,W_fc1,b_fc1))
keep_prob = tf.placeholder(tf.float32) #drop out
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer()) # 变量初始化

dt = local_data.DataSet('dataset/data')


for i in range(10000):
#    batch = mnist.train.next_batch(500)
    batch = dt.random_next_batch(50)
    if i%100 == 0:
        # print(batch[1].shape)
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("test accuracy %g"%accuracy.eval(feed_dict={x: dt.test_image(), y_: dt.test_label(), keep_prob: 1.0}))

print 'inferencing...'
prediction = tf.argmax(y_conv,1)
label = sess.run([prediction], feed_dict={x: dt.inference_image(),keep_prob:1.0})
with open('inference','w') as f:
    s = [str(int(c)) for c in label ]
    f.write(','.join(s))