#!/bin/bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace/$1 \
  -c client1 \
  -gpu 0\
  jobs/$1
