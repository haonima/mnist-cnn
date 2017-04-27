import numpy as np
import os
import random
import logging
class DataSet:
    def __init__(self,prefix='data'):
        self._epoch_num = 0
        ftrain_image = os.path.join(prefix,'train/train_image')
        ftrain_label = os.path.join(prefix,'train/train_label')
        ftest_image = os.path.join(prefix,'test/test_image')
        ftest_label = os.path.join(prefix,'test/test_label')
        inference_image = os.path.join(prefix,'inference_image')

        trimage = []
        print 'reading train images...'
        with open(ftrain_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp,dtype='float32')
                trimage.append(tmp)
        self.trimage = np.array(trimage,dtype='float32')
        self.trimage = 1-self.trimage/255
        print 'done'

        trlabel = []
        print 'reading label images...'
        with open(ftrain_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp,dtype='float32')
                trlabel.append(tmp)
        self.trlabel = np.array(trlabel,dtype='float32')
        print 'done'

        teimage = []
        print 'reading test images...'
        with open(ftest_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                teimage.append(tmp)
        self.teimage = np.array(teimage, dtype='float32')
        self.teimage = 1-self.teimage/255
        print 'done'

        telabel = []
        print 'reading test labels...'
        with open(ftest_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp,dtype='float32')
                telabel.append(tmp)
        self.telabel = np.array(telabel,dtype='float32')
        print 'done'

        infimage = []
        print 'reading inference images...'
        with open(inference_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                infimage.append(tmp)
        self.infimage = np.array(infimage, dtype='float32')
        self.infimage = 1-self.infimage/255
        print 'done'

    def next_batch(self,data_type,num):
        if data_type == 'train':
            res =  (self.trimage[self._epoch_num: self._epoch_num + num], self.trlabel[self._epoch_num: self._epoch_num + num])
            self._epoch_num += num
            return res
        else :
            res = (self.teimage[self._epoch_num: self._epoch_num + num], self.telabel[self._epoch_num: self._epoch_num + num])
            self._epoch_num += num
            return res

    def random_next_batch(self,data_type,num):
        if data_type == 'train':
            mx = len(self.trimage)
            rdm = random.randint(0,mx)

            if rdm + num > mx:
                res0 = self.trimage[rdm: ]
                res1 = self.trimage[:rdm + num - mx]
                res2 = self.trlabel[rdm:]
                res3 = self.trlabel[:rdm + num - mx]

                res = (np.vstack((res0,res1)),np.vstack((res2,res3)))
            else :
                res0 = self.trimage[rdm:rdm + num]
                res1 = self.trlabel[rdm:rdm + num]
                res = (res0,res1)

            return res
        else :
            mx = len(self.teimage)
            rdm = random.randint(0,mx)

            if rdm + num > mx:
                res0 = self.teimage[rdm: ]
                res1 = self.teimage[:rdm + num - mx]
                res2 = self.telabel[rdm:]
                res3 = self.telabel[:rdm + num - mx]

                res = (res0+res1,res2+res3)
                res = (np.vstack((res0,res1)),np.vstack((res2,res3)))
            else :
                res0 = self.trimage[rdm:rdm + num]
                res1 = self.telabel[rdm:rdm + num]
                res = (res0,res1)

            return res

    def test_image(self):
        return self.teimage
    def test_label(self):
        return self.telabel

    def inference_image(self):
        return self.infimage

