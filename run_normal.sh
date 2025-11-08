#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate comfyui
python main.py --cache-none --disable-xformers --fast fp16_accumulation
