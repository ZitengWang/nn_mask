forked from https://github.com/fgnt/nn-gev , follow the instructions therein for installation 

#

extentions: beamformers (MVDR, GEV, GEV-BAN, Variable Span, SDW-MWF, rank-1 MWF with different constraints ) used in [Rank-1 Constrained Multichannel Wiener Filter for Speech Recognition in Noisy Environments](https://www.sciencedirect.com/science/article/pii/S0885230817301584) and [ON SDW-MWF AND VARIABLE SPAN LINEAR FILTER WITH APPLICATION TO SPEECH RECOGNITION IN NOISY ENVIRONMENTS](https://2018.ieeeicassp.org/Papers/ViewPapers.asp?PaperNum=2278) 

#
#

Recognition results, see [RESULTS](https://github.com/ZitengWang/nn_mask) for all

# 
 

<table>
  <tr>
    <td></td>
    <td colspan="2">CHiME4 official backend</td>
    <td colspan="2">without LM rescore</td>
  </tr>
  <tr>
    <td></td>
    <td>et-simu</td>
    <td>et-real</td>
    <td>et-simu</td>
    <td>et-real</td>
  </tr>
  <tr>
    <td>Noisy</td>  <td>-</td> <td>-</td> <td>14.15</td> <td>23.52</td>
  </tr>
  <tr>
    <td>WDAS</td>  <td>10.91</td> <td>11.47</td> <td>14.20</td> <td>15.04</td>
  </tr>
  <tr>
    <td>MVDR</td>  <td>5.93</td> <td>7.30</td> <td>8.70</td> <td>10.31</td>
  </tr>
  <tr>
    <td>GEV-BAN</td>  <td>6.29</td> <td>7.25</td> <td>9.17</td> <td>10.48</td>
  </tr>
  <tr>
    <td>GEV</td>  <td>6.98</td> <td>7.14</td> <td>10.01</td> <td>10.53</td>
  </tr>
  <tr>
    <td>MWF</td>  <td>10.81</td> <td>12.92</td> <td>12.54</td> <td>16.16</td>
  </tr>
  <tr>
    <td>VS-rank1</td>  <td>4.15</td> <td>7.35</td> <td>6.37</td> <td>10.22</td>
  </tr>
  <tr>
    <td>r1MWF-0</td>  <td>4.56</td> <td>8.36</td> <td>7.03</td> <td>11.40</td>
  </tr>
  <tr>
    <td>r1MWF-1</td>  <td>4.59</td> <td>8.34</td> <td>7.07</td> <td>11.44</td>
  </tr>
  <tr>
    <td>r1MWF-\mu_G</td> <td>5.34</td> <td>7.27</td> <td>8.00</td> <td>10.33</td>
  </tr>
  <tr>
    <td>r1MWF-\mu_G-evd</td>  <td>5.17</td> <td>6.51</td> <td>7.63</td> <td>9.25</td>
  </tr>
  <tr>
    <td>r1MWF-\mu_G-gevd</td>  <td>4.68</td> <td>6.03</td> <td>6.87</td> <td>8.74</td>
  </tr>
  <tr>
  </tr>
   <tr>
    <td>gevd-SDW-MWF</td>  <td>4.39</td> <td>7.38</td> <td>-</td> <td>-</td>
  </tr>
   <tr>
    <td>gevd-GEV-BAN</td>  <td>4.21</td> <td>6.17</td> <td>-</td> <td>-</td>
  </tr>
    <tr>
    <td>gevd-GEV</td>  <td>4.65</td> <td>5.93</td> <td>-</td> <td>-</td>
  </tr>
</table>

