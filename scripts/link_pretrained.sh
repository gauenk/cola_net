#!/bin/bash

cd ./output/checkpoints/
ln -s ../../weights/checkpoints/DN_Gray/res_cola_v2_6_3_15_l4/model/model_best.pt pretrained_s15.pt
ln -s ../../weights/checkpoints/DN_Gray/res_cola_v2_6_3_30_l4/model/model_best.pt pretrained_s30.pt
ln -s ../../weights/checkpoints/DN_Gray/res_cola_v2_6_3_50_l4/model/model_best.pt pretrained_s50.pt
ln -s ../../weights/checkpoints/DN_Gray/res_cola_v2_6_3_70_l4/model/model_best.pt pretrained_s70.pt
