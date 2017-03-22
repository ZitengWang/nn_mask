#!/usr/bin/env python
""" extract.py

Extract CHiME 3 audio segments from continuous audio

Usage:
  extract.py [-f] [-p pad] [-c channel] <segfilenm> <inwavroot> <outwavroot>
  extract.py --help

Options:
  <segfilenm>    Name of the segmentation file
  <inwavroot>    Name of the root dir for the input audio file
  <outwavroot>   Name of the root dir for the output segments
  -p <pad>, --padding=<pad> padding at start and end in seconds  [default: 0]
  -f, --fullname  Use fullname for outfile
  -c <chan>, --channel=<chan>  Recording channel (defaults to all)
  --help         print this help screen

"""

from __future__ import print_function
import json
import os
import subprocess
import argparse
import sys


def extract(segment, in_root, out_root, padding=0.0, channel=0, fullname=False):
    """use sox to extract segment from wav file

       in_root - root directory for unsegmented audio files
       out_root - root directory for output audio segments
    """
    infilenm = '{}/{}.CH{}.wav'.format(in_root, segment['wavfile'], channel)

    if fullname:
        outtemplate = '{}/{}.{}.{}.{}.{:02d}.{:03d}.ch{}.wav'
        outfilenm = outtemplate.format(out_root,
                                       segment['wavfile'],
                                       segment['wsj_name'],
                                       segment['environment'],
                                       segment['speaker'],
                                       segment['repeat'],
                                       segment['index'],
                                       channel)
    else:
        outfilenm = '{}/{}_{}_{}.CH{}.wav'.format(out_root,
                                                  segment['speaker'],
                                                  segment['wsj_name'],
                                                  segment['environment'],
                                                  channel)

    subprocess.call(['sox', infilenm, outfilenm,
                     'trim',
                     str(segment['start'] - padding),
                     '=' + str(segment['end'] + padding)])


def to_string(segment):
    return "{}:{}-{}:{:03d}({:03d})".format(segment['wavfile'],
                                            segment['start'],
                                            segment['end'],
                                            segment['index'],
                                            segment['repeat'])


def do_extract(seg_filenm, in_root, out_root,
               padding=0.0, channel=0, fullname=False):
    """
    Extract segments listed in seg file from recording channel, 'channel'
    """

    with open(seg_filenm, 'r') as infile:
        json_string = infile.read()
    segments = json.loads(json_string)

    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    print('Extracting audio in channel {}...'.format(channel))

    for i, segment in enumerate(segments):
        sys.stdout.write(' Processing segment {: 5}/{: <5}\r'.format(i+1, len(segments)))
        sys.stdout.flush()

        extract(segment, in_root, out_root, padding=padding,
                channel=channel, fullname=fullname)
    sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    """Main method called from commandline."""
    parser = argparse.ArgumentParser(description='Extract CHiME 3 audio segments from continuous audio.')
    parser.add_argument('segfilenm', metavar='<segfilenm>',
                        help='Name of the segmentation file', type=str)
    parser.add_argument('inwavroot', metavar='<inwavroot>',
                        help='Name of the root dir for the input audio file', type=str)
    parser.add_argument('outwavroot', metavar='<outwavroot>',
                        help='Name of the root dir for the output segments', type=str)
    parser.add_argument('-p', '--padding', metavar='pad',
                        help='Padding at start and end in seconds [default: 0]', type=float, default=0)
    parser.add_argument('-f', '--fullname',
                        help='Use fullname for outfile', action='store_true')
    parser.add_argument('-c', '--channel', metavar='channel',
                        help='Recording channel (defaults to all).', action='append', type=int, default=[])

    args = parser.parse_args()

    segfilenm = args.segfilenm
    in_root = args.inwavroot
    out_root = args.outwavroot
    padding = args.padding
    fullname = args.fullname
    channels = args.channel

    if len(channels) == 0:
        channels = [0, 1, 2, 3, 4, 5, 6]

    for channel in channels:
        do_extract(segfilenm, in_root, out_root, padding, channel, fullname)


if __name__ == '__main__':
    main()

# ./extract.py ../../data/annotations/utterance/LR_141103_01.json ../../data/16khz16bit xxx
