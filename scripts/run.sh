#!/bin/bash
{
source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh
conda activate cuda11

python search_aug.py

exit
}