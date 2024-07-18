#!/bin/bash

SCRIPTNAME=$(basename $0)

USAGESTR="
NAME
${SCRIPTNAME} Does GLM hyperparameter grid search and GLM fitting for a single cell type.

SYNOPSIS
${SCRIPTNAME} <fit_type> <config> <cell_type> <output_base> [optional one-letter arguments]

OPTIONS

-j <jitter-sd>         Standard deviation of Gaussian for perturbing spike times
                       In units of electrical samples at 20 kHz

-f                     Feedback only model

-n <iter>              Number of outer iterations to run

-m <seconds>           Seconds of white noise data to use

-w <weight>            Relative weight of the WN contribution to the log-likelihood loss

-o                     Just run the optimization portion only; assumes that the grid search
                       has already completed successfully.

-p                     Train with Poisson spiking loss (default Bernoulli or binomial loss)


-r                     Train on repeats data only (mini-training for noise correlations eval)

-sh                    Train on shuffled repeats data only (must use in conjunction with -r or ignored)

Eric Wu, 2022-11-15
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
# CHANGE THESE DEPENDING ON WHAT PLATFORM YOU RUN THE SCRIPT ON
# there currently aren't any
############################################################

#################
# Positional arguments
MODEL_TYPE="";
CONFIG="";
CELL_TYPE="";
OUTPUT_BASE="";

#################
# Flag arguments
WN_WEIGHT=0.01
N_OUTER_ITER=2
SECONDS_WN=300
JITTER=0.0

FB_ONLY="" # -f
OPTIMIZE_ONLY="" # -o
POISSON_FLAG="" # -p
STUPID_FLAG="" # -s
TRAIN_REPEATS_FLAG="" # -r
TRAIN_SHUFFLE_REPEATS_FLAGS="" # -sh


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
  -w)
    shift
    WN_WEIGHT="$1"
    shift
    ;;
  -m)
    shift
    SECONDS_WN="$1"
    shift
    ;;
  -n)
    shift
    N_OUTER_ITER="$1"
    shift
    ;;
  -p)
    POISSON_FLAG="$1"
    shift
    ;;
  -s)
    STUPID_FLAG="$1"
    shift
    ;;
  -r)
    TRAIN_REPEATS_FLAG="$1"
    shift
    ;;
  -sh)
    TRAIN_SHUFFLE_REPEATS_FLAGS="$1"
    shift
    ;;
  *)
    MODEL_TYPE=$1
    shift
    CONFIG=$1
    shift
    CELL_TYPE=$1
    shift
    OUTPUT_BASE=$1
    shift
    ;;
  esac
done


##################
# check validity of inputs; fail loud
if [ "$MODEL_TYPE" != "flash" ] && [ "$MODEL_TYPE" != "combine" ] && [ "$MODEL_TYPE" != "jitter" ]; then
  echo "Model type must be either flash, combine, or jitter. Got $MODEL_TYPE"
  exit 1;
fi

###################

GLM_GRID_PATH=$OUTPUT_BASE/glm_grid
mkdir -p $GLM_GRID_PATH

MODEL_ROOT=$OUTPUT_BASE/models
mkdir -p $MODEL_ROOT

CT_FILENAME_STR=$(echo $CELL_TYPE | sed 's/ /_/' | tr '[:upper:]' '[:lower:]')
GRID_FILE_NAME="wn_grid_search_${CT_FILENAME_STR}.p"
MODEL_FILE_NAME="wn_${CT_FILENAME_STR}_glm_fits.p"

GRID_OUTPUT=$GLM_GRID_PATH/$GRID_FILE_NAME
MODEL_FIT_PATH=$MODEL_ROOT/$MODEL_FILE_NAME

FIT_FLAGS="-n ${N_OUTER_ITER} -w ${WN_WEIGHT} -j ${JITTER} -m ${SECONDS_WN}"

if [ "$MODEL_TYPE" == "flash" ]; then
  if [ "$FB_ONLY" == "-f" ]; then

    if [ "$TRAIN_REPEATS_FLAG" == "-r" ]; then
      echo "Flash FB only repeats training not supported"
      exit 1;
    else

      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python joint_wn_flashed_nscenes_fb_only_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS -hp $POISSON_FLAG
        python joint_wn_flashed_nscenes_fb_only_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS -hp $POISSON_FLAG
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      L1_CONST=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT -fb)

      echo python fit_amp_joint_wn_flashed_fb_only_glm.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -hp $POISSON_FLAG
      python fit_amp_joint_wn_flashed_fb_only_glm.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -hp $POISSON_FLAG
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi

    fi

  else

    if [ "$TRAIN_REPEATS_FLAG" == "-r" ]; then
      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python flashed_glm_repeats_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT -n $N_OUTER_ITER -hp $POISSON_FLAG $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
        python flashed_glm_repeats_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT -n $N_OUTER_ITER -hp $POISSON_FLAG $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      BEST_HYPERPARAMS_GLM=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT)
      IFS=';' read -ra BEST_HYPERPARAMS_GLM_ARR <<<"$BEST_HYPERPARAMS_GLM"
      L1_CONST=${BEST_HYPERPARAMS_GLM_ARR[0]}
      L21_CONST=${BEST_HYPERPARAMS_GLM_ARR[1]}

      echo python fit_amp_flashed_glm_repeats.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH -n $N_OUTER_ITER -l $L1_CONST -g $L21_CONST -hp $POISSON_FLAG $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
      python fit_amp_flashed_glm_repeats.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH -n $N_OUTER_ITER -l $L1_CONST -g $L21_CONST -hp $POISSON_FLAG $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi

    else
      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python joint_wn_flashed_nscenes_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS -hp $POISSON_FLAG
        python joint_wn_flashed_nscenes_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS -hp $POISSON_FLAG
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      BEST_HYPERPARAMS_GLM=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT)
      IFS=';' read -ra BEST_HYPERPARAMS_GLM_ARR <<<"$BEST_HYPERPARAMS_GLM"
      L1_CONST=${BEST_HYPERPARAMS_GLM_ARR[0]}
      L21_CONST=${BEST_HYPERPARAMS_GLM_ARR[1]}

      echo python fit_amp_joint_wn_flashed_nscenes_glm.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -g $L21_CONST -hp $POISSON_FLAG
      python fit_amp_joint_wn_flashed_nscenes_glm.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -g $L21_CONST -hp $POISSON_FLAG
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi

    fi
  fi

elif [ "$MODEL_TYPE" == "jitter" ]; then

  if [ "$FB_ONLY" == "-f" ]; then

    if [ "$TRAIN_REPEATS_FLAG" == "-r" ]; then
      echo "Jitter FB only repeats training not supported"
      exit 1;
    else

      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python joint_eye_movements_wn_fb_only_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS
        python joint_eye_movements_wn_fb_only_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      L1_CONST=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT -fb)

      echo python fit_wn_regularized_ct_jitter_movie_fb_only_glms_mp.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST
      python fit_wn_regularized_ct_jitter_movie_fb_only_glms_mp.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi
    fi

  else

    if [ "$TRAIN_REPEATS_FLAG" == "-r" ]; then

      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python eye_movements_repeats_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT -n $N_OUTER_ITER $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
        python eye_movements_repeats_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT -n $N_OUTER_ITER $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      BEST_HYPERPARAMS_GLM=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT)
      IFS=';' read -ra BEST_HYPERPARAMS_GLM_ARR <<<"$BEST_HYPERPARAMS_GLM"
      L1_CONST=${BEST_HYPERPARAMS_GLM_ARR[0]}
      L21_CONST=${BEST_HYPERPARAMS_GLM_ARR[1]}

      echo python fit_ct_jitter_movie_repeats.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH -n $N_OUTER_ITER -l $L1_CONST -g $L21_CONST $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
      python fit_ct_jitter_movie_repeats.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH -n $N_OUTER_ITER -l $L1_CONST -g $L21_CONST $TRAIN_SHUFFLE_REPEATS_FLAGS -t 15
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi
    else

      if [ "$OPTIMIZE_ONLY" != "-o" ]; then
        echo python joint_eye_movements_wn_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS
        python joint_eye_movements_wn_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS
        rc=$?
        if [[ $rc != 0 ]]; then
          echo "Grid search failed"
          exit $rc
        fi
      fi

      BEST_HYPERPARAMS_GLM=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT)
      IFS=';' read -ra BEST_HYPERPARAMS_GLM_ARR <<<"$BEST_HYPERPARAMS_GLM"
      L1_CONST=${BEST_HYPERPARAMS_GLM_ARR[0]}
      L21_CONST=${BEST_HYPERPARAMS_GLM_ARR[1]}

      echo python fit_wn_regularized_ct_jitter_movie_glms_mp.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -g $L21_CONST
      python fit_wn_regularized_ct_jitter_movie_glms_mp.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS -l $L1_CONST -g $L21_CONST
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "GLM fit failed"
        exit $rc
      fi
    fi

  fi

else
  if [ "$FB_ONLY" == "-f" ]; then

    if [ "$OPTIMIZE_ONLY" != "-o" ]; then
      echo python joint_everything_wn_flashed_jittered_FB_only_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS
      python joint_everything_wn_flashed_jittered_FB_only_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "Grid search failed"
        exit $rc
      fi
    fi

    L1_CONST=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT -c -fb)

    echo python fit_combined_flashed_jitter_wn_FB_only_glm_mp.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS
    python fit_combined_flashed_jitter_wn_FB_only_glm_mp.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS
    rc=$?
    if [[ $rc != 0 ]]; then
      echo "GLM fit failed"
      exit $rc
    fi

  else

    if [ "$OPTIMIZE_ONLY" != "-o" ]; then
      echo python joint_everything_wn_flashed_jittered_glm_grid_search.py $CONFIG "\"$CELL_TYPE\"" $GRID_OUTPUT $FIT_FLAGS
      python joint_everything_wn_flashed_jittered_glm_grid_search.py $CONFIG "$CELL_TYPE" $GRID_OUTPUT $FIT_FLAGS
      rc=$?
      if [[ $rc != 0 ]]; then
        echo "Grid search failed"
        exit $rc
      fi
    fi

    BEST_HYPERPARAMS_GLM=$(python tellmebest_glm_grid_search.py $GRID_OUTPUT -c)
    IFS=';' read -ra BEST_HYPERPARAMS_GLM_ARR <<<"$BEST_HYPERPARAMS_GLM"
    L1_CONST=${BEST_HYPERPARAMS_GLM_ARR[0]}
    L21_CONST=${BEST_HYPERPARAMS_GLM_ARR[1]}

    echo python fit_combined_flashed_jitter_wn_glm_mp.py $CONFIG "\"$CELL_TYPE\"" $MODEL_FIT_PATH $FIT_FLAGS
    python fit_combined_flashed_jitter_wn_glm_mp.py $CONFIG "$CELL_TYPE" $MODEL_FIT_PATH $FIT_FLAGS
    rc=$?
    if [[ $rc != 0 ]]; then
      echo "GLM fit failed"
      exit $rc
    fi

  fi

fi

