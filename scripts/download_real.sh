#!/usr/bin/env bash
# Real-scene benchmarks used by 3DGS / Scaffold-GS:
#   data/mipnerf360/{bicycle,garden,stump,flowers,treehill,room,counter,kitchen,bonsai}
#   data/tandt/{truck,train}   data/db/{playroom,drjohnson}
set -euo pipefail
mkdir -p data && cd data
if [ ! -d mipnerf360/bicycle ]; then
  wget -q http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
  wget -q https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
  mkdir -p mipnerf360
  unzip -q 360_v2.zip -d mipnerf360 && unzip -q -o 360_extra_scenes.zip -d mipnerf360
  rm 360_v2.zip 360_extra_scenes.zip
fi
if [ ! -d tandt/truck ]; then
  wget -q https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
  unzip -q tandt_db.zip && rm tandt_db.zip
fi
ls mipnerf360 tandt db
du -sh mipnerf360 tandt db
