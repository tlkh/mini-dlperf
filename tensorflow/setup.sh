#!/usr/bin/env bash

echo "Start script"

echo "[1/4] Install requirements.txt"

pip install -r ../requirements.txt

echo "[2/4] Update TF Addons, Datasets"

pip uninstall tensorflow-addons tensorflow-datasets -y
pip install tensorflow-addons tensorflow-datasets -U --user

echo "[3/4] Cache datasets"

echo "1. imagenette/160px"

python3 -c 'import os; import tensorflow_datasets as tfds; tfds.load("imagenette/160px",data_dir=os.environ["TFDS_DIR"])'

echo "2. glue"

python3 -c 'import os; import tensorflow_datasets as tfds; tfds.load("glue/mrpc",data_dir=os.environ["TFDS_DIR"])'
python3 -c 'import os; import tensorflow_datasets as tfds; tfds.load("glue/qqp",data_dir=os.environ["TFDS_DIR"])'

echo "3. imagenet2012"

python3 -c 'import os; import tensorflow_datasets as tfds; tfds.load("imagenet2012",data_dir=os.environ["TFDS_DIR"])'

echo "[4/4] Cache models"

#CUDA_VISIBLE_DEVICES=-1 python3 -c 'from common import xfmer_models; xfmer_models.create_model("xlm-mlm-en-2048")'

echo "Done!"
