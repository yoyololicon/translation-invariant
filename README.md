# Translation Invariant Network

This is a pytorch implementation of the baseline model described in 
[Invariances and Data Augmentation for Supervised Music Transcription](https://arxiv.org/pdf/1711.04845.pdf).
It is the current state-of-the-art Multi pitch Estimation Model evaluated on MusicNet dataset. 

The Implementation details are based on original [repository](https://github.com/jthickstun/thickstun2018invariances).

## Quick Start

1. Download [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html), the raw format.
2. Train your model.

```
python train.py --root where/your/data/is \
                --outfile your_model_name.pth
                --preprocess #set this when execute the first time
                --steps 100000
```
    

You can use ctrl+C to stop the process, and the model will always be saved.

3. Test the model on test data same as in original paper.

    
```
python test.py --infile your_model.pth

==> Loading ID 2303
==> Loading ID 1819
==> Loading ID 2382
average precision on testset: 0.6682095457772823
threshold is 0.6862663660310201
(0.6325938046425077, 0.6836922223378065, 0.6571511894346539)
```