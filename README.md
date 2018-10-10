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

==> Loading Data...
==> Building model..
7159808 of parameters.
Start Training.
steps / mse / avp_train / avp_test
1000 1.1489939180016517 0.35331320842371294 0.6423918783844346
2000 0.8266454307436943 0.6419942976527393 0.6908275697088563
3000 0.7581750468611718 0.6841538815755338 0.7231194980728902
...
99000 0.5766582242846489 0.8026213566978918 0.779892872485184
100000 0.5726349068582058 0.8063233963916538 0.7788928809627774
Finished

```

You can use ctrl+C to stop the process, and the model will always be saved.

3. Test the model on test data same as in original paper (a pre-trained model is also included in the repository).
    
```
python test.py --infile your_model.pth

==> Loading ID 2303
==> Loading ID 1819
==> Loading ID 2382
average precision on testset: 0.7691312928576854
```