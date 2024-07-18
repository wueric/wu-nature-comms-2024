#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Evaluate flashed reconstructions with MS-SSIM and LPIPS

Assumes that reconstructions have been previously produced using the
flash_reconstruct_grid_and_generate.sh shell script, and are
in the corresponding locations produced by that script


SYNOPSIS
${SCRIPTNAME} <config> <output_base> <metrics_base>

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
RECONS_BASE=""
METRICS_BASE="";
WAS_LINEAR="";

while [ "$1" != "" ]; do
  case $1 in
  -l)
    WAS_LINEAR="$1"
    shift
    ;;
  *)
    CONFIG=$1
    shift
    RECONS_BASE=$1
    shift
    METRICS_BASE=$1
    shift
    ;;
  esac
done


RECONS_PATH=$RECONS_BASE/flashed_reconstructions

RECONSTRUCTION_TEST_PATH=$RECONS_PATH/nolinear_test_reconstructions.p
RECONSTRUCTION_TEST_METRICS_PATH=$METRICS_BASE/nolinear_test_metrics.p
RECONSTRUCTION_HELDOUT_PATH=$RECONS_PATH/nolinear_heldout_reconstructions.p
RECONSTRUCTION_HELDOUT_METRICS_PATH=$METRICS_BASE/nolinear_heldout_metrics.p
if [ "$WAS_LINEAR" == "-l" ]; then
  RECONSTRUCTION_TEST_PATH=$RECONS_PATH/test_reconstructions.p
  RECONSTRUCTION_TEST_METRICS_PATH=$METRICS_BASE/test_metrics.p

  RECONSTRUCTION_HELDOUT_PATH=$RECONS_PATH/heldout_reconstructions.p
  RECONSTRUCTION_HELDOUT_METRICS_PATH=$METRICS_BASE/heldout_metrics.p
fi

echo python compute_static_metrics_by_piece.py $CONFIG $RECONSTRUCTION_TEST_PATH "glm_cropped"  $RECONSTRUCTION_TEST_METRICS_PATH
python compute_static_metrics_by_piece.py $CONFIG $RECONSTRUCTION_TEST_PATH "glm_cropped"  $RECONSTRUCTION_TEST_METRICS_PATH

echo python compute_static_metrics_by_piece.py $CONFIG $RECONSTRUCTION_HELDOUT_PATH "glm_cropped"  $RECONSTRUCTION_HELDOUT_METRICS_PATH
python compute_static_metrics_by_piece.py $CONFIG $RECONSTRUCTION_HELDOUT_PATH "glm_cropped"  $RECONSTRUCTION_HELDOUT_METRICS_PATH
