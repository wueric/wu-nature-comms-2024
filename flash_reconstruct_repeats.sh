#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Repeats flashed reconstruction grid search and generation

Assumes that the grid search has already been completed using
flash_reconstruct_grid_and_generate.sh

Assumes that the linear reconstruction filter training has already been
completed using flash_reconstruct_grid_and_generate.sh

SYNOPSIS
${SCRIPTNAME} <config> <output_base> [optional one-letter arguments]

OPTIONS

-f                     Feedback only model

-s <n_shuffles>        Number of shuffled repeats to simulate
                       If not set, use data repeats

Eric Wu, 2023-03-18
"

usage() {
  echo "$USAGESTR"
}

if [[ $1 == "" ]]; then
  echo "You must provide command line arguments to run $SCRIPTNAME."
  echo "Use $SCRIPTNAME -h for help."
  exit 1
fi

############################################################
# Global constants
MAXITER_HQS=5
############################################################

#################
# Positional arguments
CONFIG="";
OUTPUT_BASE="";

#################
# Flag arguments
FB_ONLY="" # -f
SHUFFLE="" # -o
N_SHUFFLED=""
PREFIX_FOLDER=""

while [ "$1" != "" ]; do
  case $1 in
  -f)
    FB_ONLY="$1"
    shift
    ;;
  -p)
    shift
    PREFIX_FOLDER="$1"
    shift
    ;;
  -i)
    shift
    MAXITER_HQS="$1"
    shift
    ;;
  -s)
    SHUFFLE="$1"
    shift
    N_SHUFFLED="$1"
    shift
    ;;
  -l)
    DO_LINEAR_INIT="$1"
    shift
    BINOM_MODEL_CONFIG="$1"
    shift
    ;;
  *)
    CONFIG=$1
    shift
    OUTPUT_BASE=$1
    shift
    ;;
  esac
done

MODEL_ROOT=$OUTPUT_BASE/models
YAMLPATH=$MODEL_ROOT/models.yaml

GRID_PATH=$OUTPUT_BASE/$PREFIX_FOLDER/flashed_reconstruction_grid_search
RECONS_PATH=$OUTPUT_BASE/$PREFIX_FOLDER/flashed_reconstructions
mkdir -p $RECONS_PATH

LINEAR_MODEL_PATH=$RECONS_PATH/linear_model.pth
FLASHED_RECONS_GRID_PATH=$GRID_PATH/nolinear_grid.p
if [ "$DO_LINEAR_INIT" == "-l" ]; then
  FLASHED_RECONS_GRID_PATH=$GRID_PATH/grid.p
fi

echo python tellme_best_recons_grid.py $FLASHED_RECONS_GRID_PATH
BEST_HYPERPARAMS_STATIC_STR=$(python tellme_best_recons_grid.py $FLASHED_RECONS_GRID_PATH)
IFS=';' read -ra BEST_HYPERPARAMS_STATIC_ARR <<<"$BEST_HYPERPARAMS_STATIC_STR"

OPT_LAMBDA_START=${BEST_HYPERPARAMS_STATIC_ARR[0]}
OPT_LAMBDA_END=${BEST_HYPERPARAMS_STATIC_ARR[1]}
OPT_PRIOR_WEIGHT=${BEST_HYPERPARAMS_STATIC_ARR[2]}

echo "OPTIMAL RECONS. HYPERPARAMETERS $OPT_LAMBDA_START, $OPT_LAMBDA_END, $OPT_PRIOR_WEIGHT";
printf "\n"

if [ "$DO_LINEAR_INIT" == "-l" ]; then
  REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/data_repeats.p
  SHUFFLE_RECONS_FLAGS=""
  if [ "$SHUFFLE" == "-s" ]; then
    SHUFFLE_RECONS_FLAGS="-sh $N_SHUFFLED"
    REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/shuffled_repeats.p
  fi
else
  REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/nolinear_data_repeats.p
  SHUFFLE_RECONS_FLAGS=""
  if [ "$SHUFFLE" == "-s" ]; then
    SHUFFLE_RECONS_FLAGS="-sh $N_SHUFFLED"
    REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/nolinear_shuffled_repeats.p
  fi

fi

LINEAR_GRID_FLAGS=""
if [ "$DO_LINEAR_INIT" == "-l" ]; then
  LINEAR_GRID_FLAGS="-l $LINEAR_MODEL_PATH -bb 250 -aa 151"
fi
echo python generate_repeats_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $REPEATS_RECONSTRUCTION_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -i $MAXITER_HQS -lam $OPT_PRIOR_WEIGHT -m $SHUFFLE_RECONS_FLAGS $FB_ONLY
python generate_repeats_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $REPEATS_RECONSTRUCTION_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -i $MAXITER_HQS -lam $OPT_PRIOR_WEIGHT -m $SHUFFLE_RECONS_FLAGS $FB_ONLY
