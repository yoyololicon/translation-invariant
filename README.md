# Translation Invariant Network

This is a pytorch implementation of the baseline model described in 
[Invariances and Data Augmentation for Supervised Music Transcription](https://arxiv.org/pdf/1711.04845.pdf).
It is the current state-of-the-art Multi pitch Estimation Model evaluated on MusicNet dataset. 

The Implementation details are based on original [repository](https://github.com/jthickstun/thickstun2018invariances).

## Quick Start

1. Download [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html), the raw format.
2. Train your model.


    python train.py --root where/your/data/is \
                --outfile your_model_name.pth
                --preprocess #set this when execute the first time
                --steps 100000

You can use ctrl+C to stop the process, and the model will always be saved.

3. Test the model on test data same as in original paper.