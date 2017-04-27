#!encoding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function


def map_func(args, ctx):
    from yahoo.ml.tf import TFNode
    from datetime import datetime
    import math
    import numpy
    import time
    import tensorflow as tf

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Parameters
    batch_size   = 100

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

    def print_log(worker_num, arg):
        print("{0}: {1}".format(worker_num, arg))

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

    def feed_dict():
        # Get a batch of examples from spark data feeder job
        batch = TFNode.next_batch(ctx.mgr, 100)

        # Convert from [(images, labels)] to two numpy arrays of the proper type
        images = []
        labels = []
        for item in batch:
            images.append(item[0])
            labels.append(item[1])
        xs = numpy.array(images)
        xs = xs.astype(numpy.float32)
        xs = 1 - xs/255.0
        ys = numpy.array(labels)
        ys = ys.astype(numpy.uint8)
        keep_prob = 0.9
        return (xs, ys, keep_prob)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
            x = tf.placeholder(tf.float32, [None, 784])
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            # conv_1
            W_conv1 = weight_variable([5, 5, 1, 64])
            b_conv1 = bias_variable([64])
            h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)

            # pool_1
            h_pool1 = max_pool_2x2(h_conv1)

            # conv_2
            W_conv2 = weight_variable([5, 5, 64, 32])
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)

            # conv_3
            W_conv3 = weight_variable([5, 5, 32, 16])
            b_conv3 = bias_variable([16])
            h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3) + b_conv3)

            # pool_2
            h_pool2 = max_pool_2x2(h_conv3)

            # conv_4
            W_conv4 = weight_variable([5, 5, 16, 32])
            b_conv4 = bias_variable([32])
            h_conv4 = tf.nn.elu(conv2d(h_pool2, W_conv4) + b_conv4)

            # pool_3
            h_pool3 = max_pool_2x2(h_conv4)

            # conv_5
            W_conv5 = weight_variable([2, 2, 32, 64])
            b_conv5 = bias_variable([64])
            h_conv5 = tf.nn.elu(conv2d(h_pool3, W_conv5) + b_conv5)

            # pool_4
            h_pool4 = max_pool_2x2(h_conv5)

            # fully-connected
            W_fc1 = weight_variable([2 * 2 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
            # h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1 = tf.nn.elu(tf.nn.xw_plus_b(h_pool2_flat, W_fc1, b_fc1))
            keep_prob = tf.placeholder(tf.float32)  # drop out
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数
            y_ = tf.placeholder(tf.float32, [None, 10])

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  # 损失函数，交叉熵
            train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用adam优化
            #correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 计算准确度
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            global_step = tf.Variable(0)

            # Test trained model
            label = tf.argmax(y_, 1, name="label")
            prediction = tf.argmax(y_conv, 1,name="prediction")
            correct_prediction = tf.equal(prediction, label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()


        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = TFNode.hdfs_path(ctx, args.model)
        print("tensorflow model path: {0}".format(logdir))
        summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

        if args.mode == "train":
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                   logdir=logdir,
                                   init_op=init_op,
                                   summary_op=summary_op,
                                   saver=saver,
                                   global_step=global_step,
                                   summary_writer=summary_writer,
                                   stop_grace_secs=300,
                                   save_model_secs=10)
        else:
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                   logdir=logdir,
                                   saver=saver,
                                   global_step=global_step,
                                   stop_grace_secs=300,
                                   save_model_secs=0)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("{0} session ready".format(datetime.now().isoformat()))

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            count = 0
            while not sv.should_stop() and step < args.steps:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                # using feed_dict
                batch_xs, batch_ys, kp_prob = feed_dict()
                feed = {x: batch_xs, y_: batch_ys, keep_prob: kp_prob}

                if len(batch_xs) != batch_size:
                  print("done feeding")
                  break
                else:
                  if args.mode == "train":
                    _, step = sess.run([train_op, global_step], feed_dict=feed)
                    # print accuracy and save model checkpoint to HDFS every 100 steps
                    if (step % 100 == 0):
                      print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, sess.run(accuracy,{x: batch_xs, y_: batch_ys, keep_prob: 1.0})))
                  else: # args.mode == "inference"
                      labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

                      results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
                      TFNode.batch_results(ctx.mgr, results)
                      print("acc: {0}".format(acc))

            if sv.should_stop() or step >= args.steps:
                TFNode.terminate(ctx.mgr)

        # Ask for all the services to stop.
        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()