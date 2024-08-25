#!/bin/bash

USAGESTR="
NAME
${SCRIPTNAME} Flashed reconstruction grid search and generation

SYNOPSIS
${SCRIPTNAME} <config> <output_base> [optional one-letter arguments]

OPTIONS

-y <path-to-yaml>      Use specified YAML, rather than auto-generate

-o                     Just run the generation portion only; assumes that the grid search
                       has already completed successfully.

-hh                    Just run the heldout generation portion only

Eric Wu, 2024-08-24
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
NOISE_INIT_FLAGS="-n 1e-3"
#LAMBDA_START_FLAGS="-ls -2 -1 3"
#LAMBDA_END_FLAGS="-le 0 2 5"
LAMBDA_START_FLAGS="-ls -5 -1 9"
LAMBDA_END_FLAGS="-le -1 1 9"
PRIOR_FLAGS="-p 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3"

HQS_MAX_ITER=10
######################################

#################
# Positional arguments
CONFIG="";
OUTPUT_BASE="";

#################
# Flag arguments
YAMLPATH=""
OPTIMIZE_ONLY="" # -o
HELDOUT_ONLY="" #-hh
FIXATIONAL_EM="" #-f

while [ "$1" != "" ]; do
  case $1 in
  -y)
    shift
    YAMLPATH="$1"
    shift
    ;;
  -o)
    OPTIMIZE_ONLY="$1"
    shift
    ;;
  -f)
    FIXATIONAL_EM="$1"
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

if [[ $YAMLPATH == "" ]]; then
  CELL_TYPES=("ON parasol" "OFF parasol" "ON midget" "OFF midget")
  CAT_TYPE_FNAME_ARR=()
  for ct in "${CELL_TYPES[@]}"; do
    CT_FNAME_STR=$(echo $ct | sed 's/ /_/' | tr '[:upper:]' '[:lower:]')
    MODEL_FILE_NAME="wn_joint_${CT_FNAME_STR}_fits.p"
    CAT_TYPE_FNAME_ARR=("${CAT_TYPE_FNAME_ARR[@]}" "$ct" $MODEL_FILE_NAME)
  done
  # write YAML
  echo python write_model_yaml_file.py $YAMLPATH $MODEL_ROOT "${CAT_TYPE_FNAME_ARR[@]}"
  python write_model_yaml_file.py $YAMLPATH $MODEL_ROOT "${CAT_TYPE_FNAME_ARR[@]}"
fi

rc=$?
if [[ $rc != 0 ]]; then
  echo "YAML write failed"
  exit $rc
fi

if [ "$OPTIMIZE_ONLY" != "-o" ]; then
  #do the reconstruction grid search for static images
  echo python grid_search_flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $NOISE_INIT_FLAGS $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER $FIXATIONAL_EM
  python grid_search_flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $FLASHED_RECONS_GRID_PATH $NOISE_INIT_FLAGS $LAMBDA_START_FLAGS $LAMBDA_END_FLAGS $PRIOR_FLAGS -it $HQS_MAX_ITER $FIXATIONAL_EM
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
  echo python flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH $NOISE_INIT_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -m $FIXATIONAL_EM
  python flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_TEST_PATH $NOISE_INIT_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -m $FIXATIONAL_EM
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Test partition flashed reconstruction failed"
    exit $rc
  fi
  printf "\n"
fi

echo python flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH $NOISE_INIT_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -hh -m $FIXATIONAL_EM
python flashed_noiseless_rf_reconstructions.py $CONFIG $YAMLPATH $RECONSTRUCTION_HELDOUT_PATH $NOISE_INIT_FLAGS -st $OPT_LAMBDA_START -en $OPT_LAMBDA_END -lam $OPT_PRIOR_WEIGHT -i $HQS_MAX_ITER -hh -m $FIXATIONAL_EM

rc=$?
if [[ $rc != 0 ]]; then
  echo "Heldout partition flashed reconstruction failed"
  exit $rc
fi
printf "\n"
