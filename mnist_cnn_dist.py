# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from yahoo.ml.tf import TFCluster
import dist_mapfun

sc = SparkContext(conf=SparkConf().setAppName("mnist_spark"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="mnist/mnist_model/")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()
print("args:",args)

print("{0} ===== Start".format(datetime.now().isoformat()))


images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')]).repartition(1)
labels = sc.textFile(args.labels).map(lambda ln: [float(x) for x in ln.split(',')]).repartition(1)

print("zipping images and labels")
dataRDD = images.zip(labels)
print("zipping done")

cluster = TFCluster.reserve(sc, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
cluster.start(dist_mapfun.map_func, args)
if args.mode == "train":
  cluster.train(dataRDD, args.epochs)
else:
  labelRDD = cluster.inference(dataRDD)
  labelRDD.saveAsTextFile(args.output)
cluster.shutdown()

print("{0} ===== Stop".format(datetime.now().isoformat()))

