import numpy as np
import os
from PIL import Image

def toCSV(vec):
  """Converts a vector/array into a CSV string"""
  return ','.join([str(i) for i in vec])

def writeData(sc, input_images_dir, input_labels_dir, output_images,output_labels, num_partitions):
    images = []
    for i in os.listdir(input_images_dir):
        im = Image.open(os.path.join(input_images_dir,i))
        img = np.array(im)
        images.append(img)
    images = np.array(images)
    shape=images.shape

    labels = []
    with open(input_labels_dir) as f:
        for i in f:
            tmp = np.zeros([10])
            tmp[int(i)]=1
            labels.append(tmp)
    labels = np.array(labels)

    imageRDD = sc.parallelize(images.reshape(shape[0],shape[1]*shape[2]),num_partitions)
    labelRDD = sc.parallelize(labels,num_partitions)

    imageRDD.map(toCSV).saveAsTextFile(output_images)
    labelRDD.map(toCSV).saveAsTextFile(output_labels)



def readData():
    pass

if __name__ == '__main__':
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("mnist_parallelize"))
    writeData(sc,'dataset/train/TrainImage','dataset/train/train_labels.txt','compete_mnist/images','compete_mnist/label',5)