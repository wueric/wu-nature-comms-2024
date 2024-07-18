#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Repeats eye movements reconstruction grid search and generation

Assumes that the grid search has already been completed using
eye_movements_reconstruct_grid_and_generate.sh

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

USE_MANUAL_YAML="" # -y
MANUAL_YAML_PATH=""

USE_MANUAL_GRID_PATH="" # -p
MANUAL_GRID_PATH=""

while [ "$1" != "" ]; do
  case $1 in
  -f)
    FB_ONLY="$1"
    shift
    ;;
  -s)
    SHUFFLE="$1"
    shift
    N_SHUFFLED="$1"
    shift
    ;;
  -y)
    USE_MANUAL_YAML="$1"
    shift
    MANUAL_YAML_PATH="$1"
    shift
    ;;
  -p)
    USE_MANUAL_GRID_PATH="$1"
    shift
    MANUAL_GRID_PATH="$1"
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

if [[ "$USE_MANUAL_YAML" != "-y" ]]; then
  YAMLPATH=$MODEL_ROOT/models.yaml
else
  YAMLPATH=$MANUAL_YAML_PATH
fi


GRID_PATH=$OUTPUT_BASE/reconstruction_grid_search
mkdir -p $GRID_PATH

RECONS_PATH=$OUTPUT_BASE/reconstructions
mkdir -p $RECONS_PATH


if [ "$USE_MANUAL_GRID_PATH" == "-p" ]; then
  STATIC_RECONS_HYPERPARAMS_PATH=$MANUAL_GRID_PATH/known_eye_movements_hyperparams.txt
  JOINT_RECONS_HYPERPARAMS_PATH=$MANUAL_GRID_PATH/joint_eye_movements_hyperparams.txt
else
  STATIC_RECONS_HYPERPARAMS_PATH=$GRID_PATH/known_eye_movements_hyperparams.txt
  JOINT_RECONS_HYPERPARAMS_PATH=$GRID_PATH/joint_eye_movements_hyperparams.txt
fi

BEST_HYPERPARAMS_STATIC_STR=$(<$STATIC_RECONS_HYPERPARAMS_PATH)
IFS=';' read -ra BEST_HYPERPARAMS_STATIC_ARR <<<"$BEST_HYPERPARAMS_STATIC_STR"

LAMBDA_START=${BEST_HYPERPARAMS_STATIC_ARR[0]}
LAMBDA_END=${BEST_HYPERPARAMS_STATIC_ARR[1]}
PRIOR_WEIGHT=${BEST_HYPERPARAMS_STATIC_ARR[2]}

echo "OPTIMAL HYPERPARAMETERS $LAMBDA_START, $LAMBDA_END, $PRIOR_WEIGHT";
printf "\n"

EYE_MVMT_WEIGHT=$(<$JOINT_RECONS_HYPERPARAMS_PATH)

REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/data_repeats.p
SHUFFLE_RECONS_FLAGS=""
if [ "$SHUFFLE" == "-s" ]; then
  SHUFFLE_RECONS_FLAGS="-sh $N_SHUFFLED"
  REPEATS_RECONSTRUCTION_PATH=$RECONS_PATH/shuffled_repeats.p
fi

echo python generate_eye_movements_joint_repeats_reconstructions.py $CONFIG $YAMLPATH $REPEATS_RECONSTRUCTION_PATH -st $LAMBDA_START -en $LAMBDA_END -i $MAXITER_HQS -lam $PRIOR_WEIGHT -eye $EYE_MVMT_WEIGHT $FB_ONLY $SHUFFLE_RECONS_FLAGS
python generate_eye_movements_joint_repeats_reconstructions.py $CONFIG $YAMLPATH $REPEATS_RECONSTRUCTION_PATH -st $LAMBDA_START -en $LAMBDA_END -i $MAXITER_HQS -lam $PRIOR_WEIGHT -eye $EYE_MVMT_WEIGHT $FB_ONLY $SHUFFLE_RECONS_FLAGS

