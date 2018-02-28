import caffe
import numpy as np
import argparse
import os, sys
sys.path.insert(0, "./python_layers")
import time
from PIL import Image


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="GCPNet Evaluation Script")
    parser.add_argument("--model", type=str, default='models/ade20k_full.caffemodel',
                        help="pretrained caffe models")
    parser.add_argument("--prototxt", type=str, default='prototxt/ade20k_val.prototxt',
                        help="caffe prototxt for evaluation")
    parser.add_argument("--save-dir", type=str, default='results/ade20k/val/',
                        help="directory for saving results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for evaluation")
    return parser.parse_args()

args = get_arguments()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

#init
caffe.set_device(args.gpu)
caffe.set_mode_gpu()

net = caffe.Net(args.prototxt, args.model, caffe.TEST)


val_num = 2000
val = np.zeros((val_num,),dtype='S16')
for i in range(val_num):
    val[i] = 'ADE_val_{0:0>8}'.format(i+1)

scale = 513
for i in range(val_num):
    net.blobs['scale_index_param'].data[...] = scale, i, 0
    print >> sys.stderr, "evaluating model {} on {}".format(args.model, val[i])
    start = time.time()
    net.forward()        
    score = net.blobs['score'].data[0].copy()
       
    net.blobs['scale_index_param'].data[...] = scale, i, 1
    net.forward()
    score = (score + net.blobs['score'].data[0][:,:,::-1]) / 2.
    score = score.transpose((1,2,0))
       
    end = time.time()

    predict = np.squeeze(score.argmax(2).astype(np.uint8)) + 1
    im = Image.fromarray(predict.astype(np.uint8), mode='P')
    im.save(os.path.join(args.save_dir, val[i] + '.png'))


