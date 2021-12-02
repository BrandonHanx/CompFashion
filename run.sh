#!/bin/bash

Home="/vol/research/xmodal_dl"
echo $Home

echo 'args:' $@

python $Home/CompFashion/train_net.py $@
