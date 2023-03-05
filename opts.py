# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', default=8, type=int, help='Number of classes')
    
    parser.add_argument('--model', default='multimodalcnn', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')

    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')

    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    parser.add_argument('--pretrain_path', default='EfficientFace_Trained_on_AffectNet7.pth.tar', type=str, help='Pretrained model (.pth), efficientface')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='lt', type=str, help='fusion type: lt | it | ia')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='softhard')
    args = parser.parse_args()

    return args
