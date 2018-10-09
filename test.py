import torch
from torch.utils.data import DataLoader
import argparse

from scipy.optimize import minimize_scalar
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from musicnet import MusicNet_song

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/media/ycy/Shared/Datasets/musicnet')
parser.add_argument('--infile', type=str, default='pre-trained.pth')

if __name__ == '__main__':
    args = parser.parse_args()

    test_ids = [2303, 1819, 2382]

    net = torch.load(args.infile)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    net.eval()

    y_true = []
    y_score = []

    with torch.no_grad():
        for id in test_ids:
            print('==> Loading ID', id)
            test_song = MusicNet_song(args.root, id, 44100)
            test_loader = DataLoader(test_song, batch_size=10, num_workers=1)

            for _, (inputs, targets) in enumerate(test_loader):
                y_true += [targets.detach().numpy()]
                inputs = inputs.to(device)

                outputs = net(inputs)
                y_score += [outputs.detach().cpu().numpy()]

        y_score = np.vstack(y_score).flatten()
        y_true = np.vstack(y_true).flatten()
        print("average precision on testset:", average_precision_score(y_true, y_score))


        def threshold(x):
            y2 = y_score > x
            return 1 - f1_score(y_true, y2)


        res = minimize_scalar(threshold, bounds=(0, 1), method='bounded')

        # need to change to mir_eval in the future
        print('threshold is', res.x)
        print(precision_recall_fscore_support(y_true, y_score > res.x, average='binary')[:3])
