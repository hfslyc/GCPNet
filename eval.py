import caffe
import numpy as np
import os, sys

sys.path.insert(0, "../python_layers")
import score
import time
from PIL import Image

import cv2



weights = sys.argv[2]
save_dir =sys.argv[3]
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

net = caffe.Net('prototxt/ade20k_val.prototxt',weights,caffe.TEST)


val_num = 2000
val = np.zeros((val_num,),dtype='S16')
for i in range(val_num):
    val[i] = 'ADE_val_{0:0>8}'.format(i+1)

#scale = [513, 385, 321]
scale = [513]
for i in range(val_num):
    #score_all = np.zeros((scale[0], scale[0], 150))
    for s in scale:
        net.blobs['scale_index_param'].data[...] = s, i, 0
        print >> sys.stderr, "evaluating iter {} {}, size {}".format(sys.argv[2], val[i], s)
        start = time.time()
        net.forward()        
        score = net.blobs['score'].data[0].copy()
       
        net.blobs['scale_index_param'].data[...] = s, i, 1
        net.forward()
        score = (score + net.blobs['score'].data[0][:,:,::-1]) / 2.
       
        end = time.time()
        print >> sys.stderr, "forward time {}".format(end-start)

        if s == scale[0]:
            score_all = np.zeros(score.shape)
   
        if s != scale[0]:
            score = cv2.resize(score, score_all.shape[0:2], interpolation=cv2.INTER_LINEAR)
            score = np.divide(score, np.sum(score,2)[:,:,np.newaxis])
        score_all += score
    predict = np.squeeze(score.argmax(2).astype(np.uint8)) + 1
    im = Image.fromarray(predict.astype(np.uint8), mode='P')
    im.save(os.path.join(save_dir, val[i] + '.png'))


