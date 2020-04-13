# Video-Summarisation

## This repo is part of a masters submission to the Unversity of the Witwatersrand, Johannesburg.

We extract I3D features as feature respresentations for each frame which has temporal infomation as well. 
The features include a RGB stream and a Optical flow stream

These features are fed into two auto-encoders

The latent features from the auto encoder is taken and fed into a MLP to make frame level predictions

The files 'AE_ConvRNN_I3D_features.py' in the 'Summe' and 'Tvsum50' folders perform the above process


