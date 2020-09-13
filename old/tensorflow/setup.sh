#!/usr/bin/env bash

echo "Start script"

echo "[1/2] Install requirements.txt"

python3 -m pip install -r ../requirements.txt

echo "[2/2] Update TF Addons, Datasets"

python3 -m pip uninstall tensorflow-addons tensorflow-datasets -y
python3 -m pip install tensorflow-addons tensorflow-datasets -U --user

echo "Done!"
