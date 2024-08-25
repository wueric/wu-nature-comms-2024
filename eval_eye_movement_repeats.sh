#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Evaluate repeats eye movements reconstructions with MS-SSIM and LPIPS


SYNOPSIS
${SCRIPTNAME} <config> <recons_path> <metrics_path>

Eric Wu, 2023-03-21
"

usage() {
  echo "$USAGESTR"
}

if [[ $1 == "" ]]; then
  echo "You must provide command line arguments to run $SCRIPTNAME."
  echo "Use $SCRIPTNAME -h for help."
  exit 1
fi

#################
# Positional arguments
CONFIG="";
RECONS_PATH=""
METRICS_PATH="";

while [ "$1" != "" ]; do
  case $1 in
  *)
    CONFIG=$1
    shift
    RECONS_PATH=$1
    shift
    METRICS_PATH=$1
    shift
    ;;
  esac
done

echo python compute_eye_movements_shuffled_reconstruction_quality.py $CONFIG $RECONS_PATH $METRICS_PATH
python compute_eye_movements_shuffled_reconstruction_quality.py $CONFIG $RECONS_PATH $METRICS_PATH
