#!/usr/bin/env bash

#for flist in tr05_simu tr05_real dt05_simu dt05_real et05_simu et05_real; do
#    python beamform.py $flist "$@"
#done

export PATH="/talc/multispeech/calcul/users/zwang/Software/matlab2014a/bin:/talc/multispeech/calcul/users/zwang/anaconda2/bin:$PATH"
export PYTHONPATH="$PYTHONPATH:/talc/multispeech/calcul/users/zwang/chainer"

python train.py data FW
