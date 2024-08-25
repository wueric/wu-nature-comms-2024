#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Evaluate eye movements reconstructions in frequency domain

Assumes that reconstructions have been previously produced using the
eye_movements_Reconstruct_grid_and_generate.sh shell script, and are
in the corresponding locations produced by that script


SYNOPSIS
${SCRIPTNAME} <config> <output_base> <metrics_base>

OPTIONS

-k                     Only compute for known/zero eye movements

-s                     Only compute for joint estimation


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
PATCH_SIZE="";

FIXED_EM="" # -k
SIMUL_ONLY="" # -s


while [ "$1" != "" ]; do
  case $1 in
  -k)
    FIXED_EM="$1"
    shift
    ;;
  -s)
    SIMUL_ONLY="$1"
    shift
    ;;
  *)
    CONFIG=$1
    shift
    RECONS_BASE=$1
    shift
    METRICS_BASE=$1
    shift
    PATCH_SIZE=$1
    shift
    ;;
  esac
done


RECONS_PATH=$RECONS_BASE/reconstructions


# do the reconstructions
if [ "$FIXED_EM" != "-k" ]; then

  EYE_MOVEMENT_JOINT_PATH=$RECONS_PATH/joint_estim.p
  EYE_MOVEMENT_JOINT_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_joint_estim_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $EYE_MOVEMENT_JOINT_PATH "joint_estim" $PATCH_SIZE $EYE_MOVEMENT_JOINT_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $EYE_MOVEMENT_JOINT_PATH "joint_estim" $PATCH_SIZE $EYE_MOVEMENT_JOINT_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Joint estim test partition eval failed"
    exit $rc
  fi

  HELDOUT_EYE_MOVEMENT_JOINT_PATH=$RECONS_PATH/heldout_joint_estim.p
  HELDOUT_EYE_MOVEMENT_JOINT_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_heldout_joint_estim_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_EYE_MOVEMENT_JOINT_PATH "joint_estim" $PATCH_SIZE $HELDOUT_EYE_MOVEMENT_JOINT_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_EYE_MOVEMENT_JOINT_PATH "joint_estim" $PATCH_SIZE $HELDOUT_EYE_MOVEMENT_JOINT_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Joint estim heldout partition eval failed"
    exit $rc
  fi
fi


if [ "$SIMUL_ONLY" != "-s" ]; then
  RECONSTRUCT_KNOWN_PATH=$RECONS_PATH/known_eye_movements_recons.p
  RECONSTRUCT_KNOWN_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_known_eye_movements_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $RECONSTRUCT_KNOWN_PATH "joint_estim" $PATCH_SIZE $RECONSTRUCT_KNOWN_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $RECONSTRUCT_KNOWN_PATH "joint_estim" $PATCH_SIZE $RECONSTRUCT_KNOWN_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Known movements test partition eval failed"
    exit $rc
  fi


  HELDOUT_RECONSTRUCT_KNOWN_PATH=$RECONS_PATH/heldout_known_eye_movements.p
  HELDOUT_RECONSTRUCT_KNOWN_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_heldout_known_eye_movements_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_RECONSTRUCT_KNOWN_PATH "joint_estim" $PATCH_SIZE $HELDOUT_RECONSTRUCT_KNOWN_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_RECONSTRUCT_KNOWN_PATH "joint_estim" $PATCH_SIZE $HELDOUT_RECONSTRUCT_KNOWN_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Known movements heldout partition eval failed"
    exit $rc
  fi


  RECONSTRUCT_ZERO_PATH=$RECONS_PATH/zero_eye_movements_recons.p
  RECONSTRUCT_ZERO_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_zero_eye_movements_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $RECONSTRUCT_ZERO_PATH "joint_estim" $PATCH_SIZE $RECONSTRUCT_ZERO_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $RECONSTRUCT_ZERO_PATH "joint_estim" $PATCH_SIZE $RECONSTRUCT_ZERO_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Zero movements test partition eval failed"
    exit $rc
  fi


  HELDOUT_RECONSTRUCT_ZERO_PATH=$RECONS_PATH/heldout_zero_eye_movements.p
  HELDOUT_RECONSTRUCT_ZERO_METRICS_PATH=$METRICS_BASE/FREQ_DOMAIN_heldout_zero_eye_movements_metrics.p
  echo python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_RECONSTRUCT_ZERO_PATH "joint_estim" $PATCH_SIZE $HELDOUT_RECONSTRUCT_ZERO_METRICS_PATH
  python compute_brownian_error_by_frequency.py $CONFIG $HELDOUT_RECONSTRUCT_ZERO_PATH "joint_estim" $PATCH_SIZE $HELDOUT_RECONSTRUCT_ZERO_METRICS_PATH
  rc=$?
  if [[ $rc != 0 ]]; then
    echo "Zero movements heldout partition eval failed"
    exit $rc
  fi
fi