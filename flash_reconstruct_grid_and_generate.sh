#!/bin/bash

USAGESTR="
NAME
${SCRIPTNAME} Flashed reconstruction grid search and generation

SYNOPSIS
${SCRIPTNAME} <config> <output_base> [optional one-letter arguments]

OPTIONS

-j <jitter-sd>         Standard deviation of Gaussian for perturbing spike times
                       In units of electrical samples at 20 kHz

-f                     Feedback only model

-o                     Just run the generation portion only; assumes that the grid search
                       has already completed successfully.

-hh                    Just run the heldout generation portion only

-l <binom-config>      Do linear initialization with linear model training

-p <prefix>            Folder to put the reconstructions in (if we do some sort of mod)

-i <iter>              Number of HQS iterations to run

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

###################################
# Global flags
LAMBDA_START_FLAGS="-ls -2 0 5"
LAMBDA_END_FLAGS="-le 0 3 7"
PRIOR_FLAGS="-p 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3"
#PRIOR_FLAGS="-p 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5"

HQS_MAX_ITER=10
######################################

#################
# Positional arguments
CONFIG="";
BASE_FOLDER="";
BINOM_MODEL_CONFIG="";

#################
# Flag arguments
JITTER=0.0
DO_LINEAR_INIT=""
FB_ONLY="" # -f
OPTIMIZE_ONLY="" # -o
HELDOUT_ONLY="" #-hh
PREFIX_FOLDER=""
USE_MANUAL_YAML="" # -y
MANUAL_YAML_PATH=""

while [ "$1" != "" ]; do
  case $1 in
  -j)
    shift
    JITTER="$1"
    shift
    ;;
  -f)
    FB_ONLY="$1"
    shift
    ;;
  -o)
    OPTIMIZE_ONLY="$1"
    shift
    ;;
  -hh)
    HELDOUT_ONLY="$1"
    shift
    ;;
  -l)
    DO_LINEAR_INIT="$1"
    shift
    BINOM_MODEL_CONFIG="$1"
    shift
    ;;
  -i)
    shift
    HQS_MAX_ITER="$1"
    shift
    ;;
  -p)
    shift
    PREFIX_FOLDER="$1"
    shift
    ;;
  -y)
    USE_MANUAL_YAML="$1"
    shift
    MANUAL_YAML_PATH="$1"
    shift
    ;;
  *)
    CONFIG=$1
    shift
    BASE_FOLDER=$1
    shift
    ;;
  esac
done

MODEL_ROOT=$BASE_FOLDER/models
YAMLPATH=$MODEL_ROOT/models.yaml

OUTPUT_BASE=$BASE_FOLDER
if [ "$PREFIX_FOLDER" != "" ]; then
  OUTPUT_BASE=$BASE_FOLDER/$PREFIX_FOLDER
  mkdir -p $OUTPUT_BASE
fi

GRID_PATH=$OUTPUT_BASE/flashed_reconstruction_grid_search
mkdir -p $GRID_PATH

RECONS_PATH=$OUTPUT_BASE/flashed_reconstructions
mkdir -p $RECONS_PATH

LINEAR_MODEL_PATH=$RECONS_PATH/linear_model.pth

RECONSTRUCTION_TEST_PATH=$RECONS_PATH/nolinear_test_reconstructions.p
RECONSTRUCTION_HELDOUT_PATH=$RECONS_PATH/nolinear_heldout_reconstructions.p
FLASHED_RECONS_GRID_PATH=$GRID_PATH/nolinear_grid.p
if [ "$DO_LINEAR_INIT" == "-l" ]; then
  RECONSTRUCTION_TEST_PATH=$RECONS_PATH/test_reconstructions.p
  RECONSTRUCTION_HELDOUT_PATH=$RECONS_PATH/heldout_reconstructions.p
  FLASHED_RECONS_GRID_PATH=$GRID_PATH/grid.p
fi

#######################################################
# Fit linear reconstruction model first
if [ "$DO_LINEAR_INIT" == "-l" ]; then
  if [ "$OPTIMIZE_ONLY" != "-o" ]; then
    echo python fit_linear_model.py $BINOM_MODEL_CONFIG $LINEAR_MODEL_PATH
    python fit_linear_model.py $BINOM_MODEL_CONFIG $LINEAR_MODEL_PATH
    rc=$?
    if [[ $rc != 0 ]]; then
      echo "Linear model fitting failed"
      exit $rc
    fi
  fi
fi

CT_FILE_NAMES_ARGS=""
CELL_TYPES=("ON parasol" "OFF parasol" "ON midget" "OFF midget")

MIXNMATCH_FLAG=""
if [[ "$USE_MANUAL_YAML" != "-y" ]]; then
  CT_FILE_NAMES_ARGS=""
  CELL_TYPES=("ON parasol" "OFF parasol" "ON midget" "OFF midget")
  YAMLPATH=$MODEL_ROOT/models.yaml
  for ct in "${CELL_TYPES[@]}"; do
    CT_FNAME_STR=$(echo $ct | sed 's/ /_/' | tr '[:upper:]' '[:lower:]')
    MODEL_FILE_NAME="wn_${CT_FNAME_STR}_glm_fits.p"
     CT_FILE_NAMES_ARGS+="${MODEL_FILE_NAME} "
  done

  # write YAML
  echo python write_model_yaml_file.py $YAMLPATH $MODEL_ROOT $CT_FILE_NAMES_ARGS
  python write_model_yaml_file.py $YAMLPATH $MODEL_ROOT $CT_FILE_NAMES_ARGS
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "YAML write failed"
    exit $rc
  fi

else
  YAMLPATH=$MANUAL_YAML_PATH
  MIXNMATCH_FLAG="--mixnmatch"
fi

LINEAR_GRID_FLAGS=""
if [ "$DO_LINEAR_INIT" == "-l" ]; then
  LINEAR_GRID_FLAGS="-l $LINEAR_MODEL_PATH -bb 250 -aa 151"
fi

if [ "$OPTIMIZE_ONLY" != "-o" ]; then
  #do the reconstruction grid search for static images

  echo python grid_search_hqs_cropped_glm.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $LINEAR_GRID_FLAGS $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER -j $JITTER -m $FB_ONLY $MIXNMATCH_FLAG
  python grid_search_hqs_cropped_glm.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $LINEAR_GRID_FLAGS $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER -j $JITTER -m $FB_ONLY $MIXNMATCH_FLAG
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Flashed grid search failed"
    exit $rc
  fi
fi

printf "\n"

BEST_HYPERPARAMS_STATIC_STR=$(python tellme_best_recons_grid.py $FLASHED_RECONS_GRID_PATH)
IFS=';' read -ra BEST_HYPERPARAMS_STATIC_ARR <<<"$BEST_HYPERPARAMS_STATIC_STR"

OPT_LAMBDA_START=${BEST_HYPERPARAMS_STATIC_ARR[0]}
OPT_LAMBDA_END=${BEST_HYPERPARAMS_STATIC_ARR[1]}
OPT_PRIOR_WEIGHT=${BEST_HYPERPARAMS_STATIC_ARR[2]}

echo "OPTIMAL RECONS. HYPERPARAMETERS $OPT_LAMBDA_START, $OPT_LAMBDA_END, $OPT_PRIOR_WEIGHT";
printf "\n"

if [ "$HELDOUT_ONLY" != "-hh" ]; then

  echo python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m $FB_ONLY $MIXNMATCH_FLAG
  python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m $FB_ONLY $MIXNMATCH_FLAG
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Test partition flashed reconstruction failed"
    exit $rc
  fi
  printf "\n"
fi

echo python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m -hh $FB_ONLY $MIXNMATCH_FLAG
python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH $LINEAR_GRID_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m -hh $FB_ONLY $MIXNMATCH_FLAG
rc=$?
if [[ $rc != 0 ]]; then
  echo "Heldout partition flashed reconstruction failed"
  exit $rc
fi
printf "\n"
