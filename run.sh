#!/bin/bash -ex

CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'MUTAG' --lr 0.001 --suffix 0 --epochs 500 --eta 1.0 --n_percents 5