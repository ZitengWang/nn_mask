forked from https://github.com/fgnt/nn-gev , follow the instructions therein for installation 

#

extentions: beamformers (MVDR, GEV, GEV-BAN, Variable Span, SDW-MWF, rank-1 MWF with different constraints ) used in [Rank-1 Constrained Multichannel Wiener Filter for Speech Recognition in Noisy Environments](https://arxiv.org/abs/1707.00201)

#
#

Recognition results of the proposed r1MWF-\mu_G-gevd, see [RESULTS](https://github.com/ZitengWang/nn_mask) for all

# 
 
./local/chime4_calc_wers_smbr.sh exp/tri4a_dnn_tr05_multi_noisy_smbr_i1lats r1mwf_rnn_gevd exp/tri4a_dnn_tr05_multi_noisy/graph_tgpr_5k
compute WER for each location

best overall dt05 WER 6.02% (language model weight = 11)
 (Number of iterations = 4)

dt05_simu WER: 6.01% (Average), 5.49% (BUS), 7.04% (CAFE), 5.58% (PEDESTRIAN), 5.93% (STREET)

dt05_real WER: 6.03% (Average), 7.15% (BUS), 6.18% (CAFE), 5.15% (PEDESTRIAN), 5.65% (STREET)

et05_simu WER: 6.84% (Average), 5.85% (BUS), 7.81% (CAFE), 7.08% (PEDESTRIAN), 6.61% (STREET)

et05_real WER: 8.74% (Average), 11.59% (BUS), 8.76% (CAFE), 7.42% (PEDESTRIAN), 7.21% (STREET)

#

LM rescore

local/chime4_calc_wers.sh exp/tri4a_dnn_tr05_multi_noisy_smbr_lmrescore r1mwf_rnn_gevd_rnnlm_5k_h300_w0.5_n100 exp/tri4a_dnn_tr05_multi_noisy_smbr_lmrescore/graph_tgpr_5k
compute dt05 WER for each location

best overall dt05 WER 4.02% (language model weight = 12)

dt05_simu WER: 4.10% (Average), 3.72% (BUS), 4.81% (CAFE), 3.66% (PEDESTRIAN), 4.23% (STREET)

dt05_real WER: 3.93% (Average), 4.59% (BUS), 3.82% (CAFE), 3.56% (PEDESTRIAN), 3.75% (STREET)

et05_simu WER: 4.68% (Average), 3.75% (BUS), 5.29% (CAFE), 4.76% (PEDESTRIAN), 4.91% (STREET)

et05_real WER: 6.03% (Average), 8.00% (BUS), 5.66% (CAFE), 5.29% (PEDESTRIAN), 5.16% (STREET)

#
#

On AM trained with all 6 channels

./local/chime4_calc_wers_smbr.sh exp/tri4a_dnn_tr05_multi_noisy_6ch_smbr_i1lats r1mwf_rnn_gevd exp/tri4a_dnn_tr05_multi_noisy_6ch/graph_tgpr_5k
compute WER for each location

best overall dt05 WER 5.41% (language model weight = 11)
 (Number of iterations = 2)

dt05_simu WER: 5.29% (Average), 4.85% (BUS), 5.94% (CAFE), 5.01% (PEDESTRIAN), 5.34% (STREET)

dt05_real WER: 5.53% (Average), 7.14% (BUS), 5.24% (CAFE), 4.37% (PEDESTRIAN), 5.37% (STREET)

et05_simu WER: 5.83% (Average), 5.17% (BUS), 6.50% (CAFE), 5.98% (PEDESTRIAN), 5.66% (STREET)

et05_real WER: 7.71% (Average), 9.95% (BUS), 6.87% (CAFE), 7.14% (PEDESTRIAN), 6.87% (STREET)

#

LM rescore

local/chime4_calc_wers.sh exp/tri4a_dnn_tr05_multi_noisy_6ch_smbr_lmrescore r1mwf_rnn_gevd_rnnlm_5k_h300_w0.5_n100 exp/tri4a_dnn_tr05_multi_noisy_6ch_smbr_lmrescore/graph_tgpr_5k
compute dt05 WER for each location

best overall dt05 WER 3.58% (language model weight = 14)

dt05_simu WER: 3.60% (Average), 3.39% (BUS), 4.16% (CAFE), 3.26% (PEDESTRIAN), 3.60% (STREET)

dt05_real WER: 3.56% (Average), 4.44% (BUS), 3.33% (CAFE), 3.14% (PEDESTRIAN), 3.32% (STREET)

et05_simu WER: 3.85% (Average), 3.23% (BUS), 4.22% (CAFE), 3.98% (PEDESTRIAN), 3.96% (STREET)

et05_real WER: 5.05% (Average), 6.51% (BUS), 4.45% (CAFE), 4.45% (PEDESTRIAN), 4.82% (STREET)
 
