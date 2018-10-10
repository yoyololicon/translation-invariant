import torch
from torch import optim
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse

from baseline_model import spectrograms, EMA
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from musicnet import MusicNet, MusicNet_song

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/host/data_dsk1/dataset/musicnet')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--outfile', type=str, default='pre-trained.pth')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--batch', type=int, default=150)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        m.weight.data.normal_(0, 1e-4)


if __name__ == '__main__':
    args = parser.parse_args()

    preprocess = args.preprocess
    esize = args.steps * args.batch
    batch_size = args.batch

    window = 16384
    decay = 2e-4

    print('==> Loading Data...')
    train_set = MusicNet(args.root, preprocess=args.preprocess, normalize=True, window=window, epoch_size=esize)
    valid_set = MusicNet(args.root, train=False, normalize=True, window=window, epoch_size=batch_size * 10)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4)

    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = spectrograms().to(device)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    if device == 'cuda':
        cudnn.benchmark = True

    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    net.apply(init_weights)
    ema = EMA(1 - decay)
    for name, p in net.named_parameters():
        if p.requires_grad:
            ema.register(name, p.data)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.95)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=1/3)

    print("Start Training.")
    print("steps / mse / avp_train / avp_test")
    global_step = 0
    average_loss = []
    avp_train = []
    try:
        with train_set, valid_set:
            net.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                scheduler.step()
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, targets) * 64
                loss.backward()
                optimizer.step()
                for name, p in net.named_parameters():
                    if p.requires_grad:
                        p.data = ema(name, p.data)

                global_step += 1
                average_loss.append(loss.item())

                y_score = outputs.detach().cpu().numpy().flatten()
                y_true = targets.detach().cpu().numpy().flatten()
                avp_train.append(average_precision_score(y_true, y_score))

                if global_step % 1000 == 0:
                    net.eval()
                    with torch.no_grad():
                        y_true = []
                        y_score = []
                        for _, (inputs, targets) in enumerate(valid_loader):
                            y_true += [targets.detach().numpy()]
                            inputs = inputs.to(device)

                            outputs = net(inputs)
                            y_score += [outputs.detach().cpu().numpy()]

                        y_score = np.vstack(y_score).flatten()
                        y_true = np.vstack(y_true).flatten()
                        print(global_step, np.mean(average_loss), np.mean(avp_train),
                              average_precision_score(y_true, y_score))
                        average_loss.clear()
                        avp_train.clear()
                    net.train()

    except KeyboardInterrupt:
        print('Graceful Exit')
    else:
        print('Finished')
    net.eval()
    net.cpu()
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save(net, args.outfile)
