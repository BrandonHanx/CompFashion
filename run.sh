#!/bin/bash

PYTHONHOME="/vol/research/xmodal_dl/compfashion-env/bin/"
HOME="/vol/research/xmodal_dl/CompFashion/"

echo $HOME
echo 'args:' $@

$PYTHONHOME/python $Home/train_net.py --root $HOME $@
