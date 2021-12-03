#!/bin/bash

PYTHONHOME="/user/HS500/xh00414/miniconda3/envs/py37/bin"
HOME="/vol/research/xmodal_dl/CompFashion"

echo $HOME
echo 'args:' $@

$PYTHONHOME/python $HOME/train_net.py --root $HOME $@
