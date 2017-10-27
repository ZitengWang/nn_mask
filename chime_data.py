import json
import os
import pickle

import numpy as np
import tqdm
import random

from fgnt.mask_estimation import estimate_IBM
from fgnt.signal_processing import audioread
from fgnt.signal_processing import stft
from fgnt.utils import mkdir_p


def gen_flist_simu(chime_data_dir, stage, ext=False):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'simu'))) as fid:
        annotations = json.load(fid)
    if ext:
        isolated_dir = 'isolated_ext'
    else:
        isolated_dir = 'isolated'
    flist = [os.path.join(
            chime_data_dir, 'audio', '16kHz', isolated_dir,
            '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'),
            '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations]
    return flist

def gen_part_flist_simu(chime_data_dir, stage, env, ext=False):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'simu'))) as fid:
        annotations = json.load(fid)
    if ext:
        isolated_dir = 'isolated_ext'
    else:
        isolated_dir = 'isolated'
    flist=list()
    for a in annotations:
        if a['environment'].lower() == env.lower():        
            flist.append(os.path.join(
            chime_data_dir, 'audio', '16kHz', isolated_dir,
            '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'),
            '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment'])))
    return flist
    
def gen_part_flist_real(chime_data_dir, stage, env):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'real'))) as fid:
        annotations = json.load(fid)
    flist_tuples=list()
    for a in annotations:
        if a['environment'].lower() == env.lower():
            flist_tuples.append((os.path.join(
            chime_data_dir, 'audio', '16kHz', 'embedded', a['wavfile']),
                     a['start'], a['end'], a['wsj_name']))
    return flist_tuples
    
def gen_flist_real(chime_data_dir, stage):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'real'))) as fid:
        annotations = json.load(fid)
    flist_tuples = [(os.path.join(
            chime_data_dir, 'audio', '16kHz', 'embedded', a['wavfile']),
                     a['start'], a['end'], a['wsj_name']) for a in annotations]
    return flist_tuples


def get_audio_data(file_template, postfix='', ch_range=range(1, 7)):
    audio_data = list()
    if ch_range==100:
        audio_data.append(audioread(
                file_template + '.wav')[None, :])
        audio_data = np.concatenate(audio_data, axis=0)
        audio_data = audio_data.astype(np.float32)
        return audio_data
    for ch in ch_range:
        if os.path.exists(file_template+'.CH{}{}.wav'.format(ch, postfix)):
            audio_data.append(audioread(
                file_template + '.CH{}{}.wav'.format(ch, postfix))[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data

def get_audio_data_2ch(file_template, postfix=''):
    audio_data = list()
    chs = [1,3,4,5,6]
    ch_range=[1,3]
    ch_range[0] = chs.pop(random.randint(0,len(chs)-1))
    ch_range[1] = chs.pop(random.randint(0,len(chs)-1))
    for ch in ch_range:
        audio_data.append(audioread(
                file_template + '.CH{}{}.wav'.format(ch, postfix))[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data

def get_audio_data_with_context_2ch(embedded_template, t_start, t_end, wsj_name, cato,
                           ch_range=range(1, 7) ):
    start_context = max((t_start - 5), 0)
    context_samples = (t_start - start_context) * 16000
    audio_data = list()
#    chs = [1,3,4,5,6]
#    ch_range=[-1,-1]
#    ch_range[0] = chs.pop(random.randint(0,len(chs)-1))
#    ch_range[1] = chs.pop(random.randint(0,len(chs)-1))
    embedded_name = embedded_template.split('\\')[-1]
    file_name=embedded_name[0:4]+wsj_name+embedded_name[-4:]
    file_template='/CHiME3/data/audio/16kHz/isolated_2ch_track/'+cato+'/'+file_name
    for ch in ch_range:
    # check if the file exists in dir isolated_2ch_track
        if os.path.exists(file_template+'.CH{}.wav'.format(ch)):
            audio_data.append(audioread(
                embedded_template + '.CH{}.wav'.format(ch),
                offset=start_context, duration=t_end - start_context)[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data, context_samples
    
def get_audio_data_with_context(embedded_template, t_start, t_end,
                                ch_range=range(1, 7)):
    start_context = max((t_start - 5), 0)
    context_samples = (t_start - start_context) * 16000
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
                embedded_template + '.CH{}.wav'.format(ch),
                offset=start_context, duration=t_end - start_context)[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data, context_samples

def get_audio_data_after_corr_check(embedded_template, t_start, t_end, xcorr):
    threshold = 0.2
    ch_range=range(1,7)
    for ind in range(1,7):
        if xcorr[ind-1] < threshold:
            ch_range.remove(ind)
    # check failure, return ch_range
    start_context = max((t_start - 5), 0)
    context_samples = (t_start - start_context) * 16000
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
                embedded_template + '.CH{}.wav'.format(ch),
                offset=start_context, duration=t_end - start_context)[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data, context_samples

def prepare_training_data(chime_data_dir, dest_dir):
    for stage in ['tr', 'dt']:
        flist = gen_flist_simu(chime_data_dir, stage, ext=True)
        export_flist = list()
        #mkdir_p(os.path.join(dest_dir, stage))
        for f in tqdm.tqdm(flist, desc='Generating data for {}'.format(stage)):
            clean_audio = get_audio_data(f, '.Clean')
            noise_audio = get_audio_data(f, '.Noise')
            X = stft(clean_audio, time_dim=1).transpose((1, 0, 2))
            N = stft(noise_audio, time_dim=1).transpose((1, 0, 2))
            IBM_X, IBM_N = estimate_IBM(X, N)
            Y_abs = np.abs(X + N)
            X_abs = np.abs(X)
            N_abs = np.abs(N)
            PSX_abs = np.real(X/(X+N))
            PSN_abs = np.real(N/(X+N))         
  # add the clean speech spectrum and phase sensitive spectrum
            export_dict = {
                'IBM_X': IBM_X.astype(np.float32),
                'IBM_N': IBM_N.astype(np.float32),
                'X_abs': X_abs.astype(np.float32),
                'N_abs': N_abs.astype(np.float32),
                'PSX_abs' : PSX_abs.astype(np.float32),
                'PSN_abs' : PSN_abs.astype(np.float32),
                'Y_abs': Y_abs.astype(np.float32)
            }
            export_name = os.path.join(dest_dir, stage, f.split('/')[-1])
            with open(export_name, 'wb') as fid:
                pickle.dump(export_dict, fid)
            export_flist.append(os.path.join(stage, f.split('/')[-1]))
        with open(os.path.join(dest_dir, 'flist_{}.json'.format(stage)),
                  'w') as fid:
            json.dump(export_flist, fid, indent=4)



