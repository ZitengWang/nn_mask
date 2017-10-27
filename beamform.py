import argparse
import os

import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers
from tqdm import tqdm
import string
from chainer import using_config

from chime_data import gen_flist_simu, \
    gen_flist_real, get_audio_data, get_audio_data_with_context, \
    get_audio_data_after_corr_check
from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import audiowrite, stft, istft
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator, SimpleFWMaskEstimator

parser = argparse.ArgumentParser(description='NN beamforming')
parser.add_argument('flist',
                    help='Name of the flist to process (e.g. tr05_simu)')
parser.add_argument('chime_dir',
                    help='Base directory of the CHiME challenge.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

serializers.load_hdf5(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

stage = args.flist[:2]
scenario = args.flist.split('_')[-1]

# CHiME data handling
if scenario == 'simu':
    flist = gen_flist_simu(args.chime_dir, stage)
elif scenario == 'real':
    flist = gen_flist_real(args.chime_dir, stage)
else:
    raise ValueError('Unknown flist {}'.format(args.flist))

for env in ['caf', 'bus', 'str', 'ped']:
    mkdir_p(os.path.join(args.output_dir, '{}05_{}_{}'.format(
            stage, env, scenario
    )))

# get the correlation coefficients for reference channel selection
with open('mic_error') as a_file:
    xcorr=dict()
    for a_line in a_file:
        A=a_line.split()
        B=np.zeros((6,1),dtype=np.float)        
        for ind in range(1,7):
            B[ind-1]=string.atof(A[ind])
        C={
            A[0]:B
        }
        xcorr.update(C)

output_setup = {}
# output types: gev; mvdr; sdw-mwf; r1-mwf; vs 
output_setup['output_type'] = 'sdw-mwf'
output_setup['gev_ban'] = False
output_setup['mwf_mu'] = 1  # trade-off parameter: a scalar value or 'rnn'
output_setup['evd'] = False # rank-1 reconstruction or not
output_setup['gevd'] = True
output_setup['vs_Qrank'] = 1 # 
outfile_postfix = ''

 
t_io = 0
t_net = 0
t_beamform = 0
# Beamform loop
for cur_line in tqdm(flist, miniters=1000):
    with Timer() as t:
        if scenario == 'simu':
            audio_data = get_audio_data(cur_line)
            context_samples = 0
            corr_info = None
        elif scenario == 'real':
            #audio_data, context_samples = get_audio_data_with_context_2ch(
            #        cur_line[0], cur_line[1], cur_line[2], cur_line[3], args.flist)
            # get the wavname
            spk = cur_line[0][-18:-14]
            env = cur_line[0][-4:]
            file_name = spk + cur_line[3] + env
            corr_info = xcorr[file_name]
            audio_data, context_samples = get_audio_data_after_corr_check(
                cur_line[0], cur_line[1], cur_line[2], xcorr[file_name])
    t_io += t.msecs
    Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    Y_var = Variable(np.abs(Y).astype(np.float32))
    if args.gpu >= 0:
        Y_var.to_gpu(args.gpu)
    with Timer() as t:
        with using_config('train', False): 
            N_masks, X_masks = model.calc_masks(Y_var)
        N_masks.to_cpu()
        X_masks.to_cpu()
    t_net += t.msecs

    with Timer() as t:
        data_tmp=X_masks.data
        N_mask = np.median(N_masks.data, axis=1)
        X_mask = np.median(X_masks.data, axis=1)
        Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask, output_setup, corr=corr_info)
    t_beamform += t.msecs

    # the spliter in Win '\' and Linux '/'
    if scenario == 'simu':
        wsj_name = cur_line.split('/')[-1].split('_')[1]
        spk = cur_line.split('/')[-1].split('_')[0]
        env = cur_line.split('/')[-1].split('_')[-1]
    elif scenario == 'real':
        wsj_name = cur_line[3]
        spk = cur_line[0].split('/')[-1].split('_')[0]
        env = cur_line[0].split('/')[-1].split('_')[-1]

    filename = os.path.join(
            args.output_dir,
            '{}05_{}_{}'.format(stage, env.lower(), scenario),
            '{}_{}_{}.wav'.format(spk, wsj_name, env.upper())
    )

    audiowrite(istft(Y_hat, audio_data.shape[1])[context_samples:], filename[:-4]+outfile_postfix+'.wav', 16000, True, True)
    '''
    # direct apply the mask on the DS
    Y_ds_hat = np.sum(Y, axis=1) * X_mask
    audiowrite(istft(Y_ds_hat, audio_data.shape[1])[context_samples:], filename[:-4]+'_X.wav', 16000, True, True)
    Y_ds_hat = np.sum(Y, axis=1) * X_mask/(X_mask+N_mask)
    audiowrite(istft(Y_ds_hat, audio_data.shape[1])[context_samples:], filename[:-4]+'_W.wav', 16000, True, True)
    '''
print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))
