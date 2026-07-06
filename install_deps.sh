#!/bin/bash

pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ./IsaacLab-v1.2.0
cd ./IsaacLab-v1.2.0
git checkout tags/v1.2.0
./isaaclab.sh -i
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

pip install -r baselines/navdp/requirements.txt
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
