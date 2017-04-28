import numpy as np
import os
import random
import logging
class DataSet:
    def __init__(self,prefix='data'):
        self._epoch_num = 0
        ftrain_image = os.path.join(prefix,'train/train_image')
        ftrain_label = os.path.join(prefix,'train/train_label')
        frotate_15_image = os.path.join(prefix, 'train/rotated_images_15')
        frotate_15_label = os.path.join(prefix, 'train/train_label')
        frotate_30_image = os.path.join(prefix, 'train/rotated_images_30')
        frotate_30_label = os.path.join(prefix, 'train/train_label')
        frotate_m15_image = os.path.join(prefix, 'train/rotated_images_-15')
        frotate_m15_label = os.path.join(prefix, 'train/train_label')
        frotate_m30_image = os.path.join(prefix, 'train/rotated_images_-30')
        frotate_m30_label = os.path.join(prefix, 'train/train_label')
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

        trotate_15image = []
        print 'reading rotate_15 images...'
        with open(frotate_15_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_15image.append(tmp)
        self.trotate_15image = np.array(trotate_15image, dtype='float32')
        self.trotate_15image = 1 - self.trotate_15image / 255
        print 'done'

        trotate_15label = []
        print 'reading rotate_15 label ...'
        with open(ftrain_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_15label.append(tmp)
        self.trotate_15label = np.array(trotate_15label, dtype='float32')
        print 'done'

        trotate_m15image = []
        print 'reading rotate -15 images...'
        with open(frotate_m15_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_m15image.append(tmp)
        self.trotate_m15image = np.array(trotate_m15image, dtype='float32')
        self.trotate_m15image = 1 - self.trotate_m15image / 255
        print 'done'

        trotate_m15label = []
        print 'reading rotate -15 label...'
        with open(ftrain_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_m15label.append(tmp)
        self.trotate_m15label = np.array(trotate_m15label, dtype='float32')
        print 'done'

        trotate_30image = []
        print 'reading rotate 30 images...'
        with open(frotate_30_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_30image.append(tmp)
        self.trotate_30image = np.array(trotate_30image, dtype='float32')
        self.trotate_30image = 1 - self.trotate_30image / 255
        print 'done'

        trotate_30label = []
        print 'reading rotate 30 label...'
        with open(ftrain_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_30label.append(tmp)
        self.trotate_30label = np.array(trotate_30label, dtype='float32')
        print 'done'

        trotate_m30image = []
        print 'reading rotate -30 images...'
        with open(frotate_m30_image) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_m30image.append(tmp)
        self.trotate_m30image = np.array(trotate_m30image, dtype='float32')
        self.trotate_m30image = 1 - self.trotate_m30image / 255
        print 'done'

        trotate_m30label = []
        print 'reading rotate -30 label...'
        with open(ftrain_label) as f:
            for i in f:
                tmp = [x for x in i.split(',')]
                tmp = np.array(tmp, dtype='float32')
                trotate_m30label.append(tmp)
        self.trotate_m30label = np.array(trotate_m30label, dtype='float32')
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

        self.dtset = np.vstack((self.trimage,self.trotate_15image,self.trotate_30image,self.trotate_m15image,self.trotate_m30image))
        self.lbset = np.vstack((self.trlabel,self.trlabel,self.trlabel,self.trlabel,self.trlabel))

    def next_batch(self,data_type,num):
        if data_type == 'train':
            res =  (self.trimage[self._epoch_num: self._epoch_num + num], self.trlabel[self._epoch_num: self._epoch_num + num])
            self._epoch_num += num
            return res
        else :
            res = (self.teimage[self._epoch_num: self._epoch_num + num], self.telabel[self._epoch_num: self._epoch_num + num])
            self._epoch_num += num
            return res

    def random_next_batch(self,num):

        mx = len(self.dtset)
        rdm = random.randint(0,mx)

        if rdm + num > mx:
            res0 = self.dtset[rdm: ]
            res1 = self.dtset[:rdm + num - mx]
            res2 = self.lbset[rdm:]
            res3 = self.lbset[:rdm + num - mx]

            res = (np.vstack((res0,res1)),np.vstack((res2,res3)))
        else :
            res0 = self.dtset[rdm:rdm + num]
            res1 = self.lbset[rdm:rdm + num]
            res = (res0,res1)

        return res
        '''
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
'''

    def test_image(self):
        return self.teimage
    def test_label(self):
        return self.telabel

    def inference_image(self):
        return self.infimage

