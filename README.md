# Scene Parsing with Global Context Embedding

This repo is the caffe implementation of the following paper:

[Scene Parsing with Global Context Embedding](https://arxiv.org/abs/1710.06507) <br/>
[Wei-Chih Hung](http://hfslyc.github.io), 
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), 
[Xiaohui Shen](https://research.adobe.com/person/xiaohui-shen/), 
[Zhe Lin](https://research.adobe.com/person/zhe-lin/), 
[Kalyan Sunkavalli](https://research.adobe.com/person/kalyan-sunkavalli/), 
Xin Lu, 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/). In ICCV, 2017.


Please cite our paper if you find it useful for your research.

```
@inproceedings{hung2017scene,
  title={Scene Parsing With Global Context Embedding},
  author={Hung, Wei-Chih and Tsai, Yi-Hsuan and Shen, Xiaohui and Lin, Zhe and Sunkavalli, Kalyan and Lu, Xin and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2017}
}
```

## Prerequisite

* DeepLab-v2 caffe. You will need the this updated version [[link]](https://github.com/hfslyc/Caffe-DeepLab-v2-Reduced) for most recent machine setups.
* A GPU with at least 12GB


## Test  on ADE20k validation set

* Download the [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset and put it in ```data/```.

The directories should be like this:
```
./data/ADE20k/annotations/validation
             /images/validation
```

* Download pretrained model

```
bash get_model.sh
```

* Download precomputed context features/priors of ADE20k val set.

```
bash get_prior.sh
```

* Execute evaluation script:

```
python eval.py --prototxt prototxt/ade20k_val.prototxt \
               --model models/ade20k_full.caffemodel \
               --save-dir results/ade20k/val/ \
               --gpu 0
```
The result images will be saved at ```results/ade20k/val/```.


