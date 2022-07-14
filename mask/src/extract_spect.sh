#!/bin/bash
# Copyright (C) 2020 Shahin Amiriparian, Michael Freitag, Maurice Gerczuk, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.


verbose_option=""

# Uncomment for debugging of auDeep
# verbose_option=" --verbose --debug"

# Uncomment for debugging of shell script
# set -x;

export PYTHONUNBUFFERED=1

taskName="compare20-Mask"
workspace="../"

# base directory for audio files
audio_base="../"

##########################################################
# 1. Spectrogram Extraction
##########################################################

# We use 80 ms Hann windows to compute spectrograms
window_width="0.08"

# We use 40 ms overlap between windows to compute spectrograms
window_overlap="0.04"

# Mel-scale spectrograms with 128 frequency bands are extracted
mel_bands="128"

# The ComParE 2020 Mask (M) audio files differ in length. By setting the --fixed-length option, we make sure that all
# audio files are exactly 1 second long. This is achieved by cutting or zero-padding audio files as required.
fixed_length="1"

# We filter low amplitudes in the spectrograms, which eliminates some background noise. Our system normalises
# spectrograms so that the maximum amplitude is 0 dB, and we filter amplitudes below -30 dB, -45 dB, -60 dB and -75 dB.
clip_below_values="-60" #"-30 -45 -60 -75"

# Parser for the data set
parser="audeep.backend.parsers.compare20_mask.Compare20MaskParser"

# Base path for spectrogram files. auDeep automatically creates the required directories for us.
spectrogram_base="${workspace}/spectrograms_new60"

for clip_below_value in ${clip_below_values}; do
    spectrogram_file="${spectrogram_base}/${taskName}-${window_width}-${window_overlap}-${mel_bands}_no_clip.nc"

    if [ ! -f ${spectrogram_file} ]; then
        echo audeep preprocess${verbose_option} --parser ${parser} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --center-fixed --mel-spectrum ${mel_bands}
        echo
        start=`date +%s`
        audeep preprocess${verbose_option} --parser ${parser} --basedir ${audio_base} --output ${spectrogram_file} --window-width ${window_width} --window-overlap ${window_overlap} --fixed-length ${fixed_length} --center-fixed  --mel-spectrum ${mel_bands}
        end=`date +%s`
        echo "Runtime: $((end-start))"
    fi
done
