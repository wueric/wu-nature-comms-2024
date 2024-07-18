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
LAMBDA_START_FLAGS="-ls -2 -1 3"
LAMBDA_END_FLAGS="-le 0 2 5"
PRIOR_FLAGS="-p 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3"

HQS_MAX_ITER=10
######################################

#################
# Positional arguments
CONFIG="";
OUTPUT_BASE="";

#################
# Flag arguments
JITTER=0.0
FB_ONLY="" # -f
OPTIMIZE_ONLY="" # -o
HELDOUT_ONLY="" #-hh


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

GRID_PATH=$OUTPUT_BASE/flashed_reconstruction_grid_search
mkdir -p $GRID_PATH

RECONS_PATH=$OUTPUT_BASE/flashed_reconstructions
mkdir -p $RECONS_PATH

RECONSTRUCTION_TEST_PATH=$RECONS_PATH/test_reconstructions.p
RECONSTRUCTION_HELDOUT_PATH=$RECONS_PATH/heldout_reconstructions.p

FLASHED_RECONS_GRID_PATH=$GRID_PATH/grid.p

CT_FILE_NAMES_ARGS=""
CELL_TYPES=("ON parasol" "OFF parasol" "ON midget" "OFF midget")

for ct in "${CELL_TYPES[@]}"; do
  CT_FNAME_STR=$(echo $ct | sed 's/ /_/' | tr '[:upper:]' '[:lower:]')
  MODEL_FILE_NAME="wn_joint_${CT_FNAME_STR}_fits.p"
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

if [ "$OPTIMIZE_ONLY" != "-o" ]; then
  #do the reconstruction grid search for static images
  echo python grid_search_hqs_cropped_glm.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER -j $JITTER -m $FB_ONLY
  python grid_search_hqs_cropped_glm.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER -j $JITTER -m $FB_ONLY
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
  echo python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m $FB_ONLY
  python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m $FB_ONLY
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Test partition flashed reconstruction failed"
    exit $rc
  fi
  printf "\n"
fi

echo python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m -hh $FB_ONLY
python generate_cropped_glm_hqs_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -j $JITTER -m -hh $FB_ONLY
rc=$?
if [[ $rc != 0 ]]; then
  echo "Heldout partition flashed reconstruction failed"
  exit $rc
fi
printf "\n"
