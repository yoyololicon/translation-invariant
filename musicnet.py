from torch.utils.data import Dataset
from intervaltree import IntervalTree
import os
import pickle
import mmap
import numpy as np
from csv import DictReader
from scipy.io import wavfile

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


# modified from the origin pytorch musicnet
# remove download action (you have to download the data outside), and make it can run on python3
class MusicNet(Dataset):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"

        preprocess: if first time use this class, assign it as true
    """
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]

    def __init__(self, root, train=True, preprocess=False, mmap=False, normalize=True, window=16384, pitch_shift=0,
                 jitter=0., epoch_size=100000):
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.size = epoch_size
        self.m = 128

        self.root = os.path.expanduser(root)

        if preprocess:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.npy'): continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff) / sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (
                    os.path.join(self.data_path, record), os.fstat(f.fileno()).st_size / sz_float)
                f.close()

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self, rec_id, s, shift=0, jitter=0):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        scale = 2. ** ((shift + jitter) / 12.)

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s * sz_float:int(s + scale * self.window) * sz_float],
                              dtype=np.float32).copy()
        else:
            fid, _ = self.records[rec_id]
            with open(fid, 'rb') as f:
                f.seek(s * sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale * self.window))

        if self.normalize: x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window, dtype=np.float32)
        x = np.interp(scale * xp, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

        y = np.zeros(self.m, dtype=np.float32)
        for label in self.labels[rec_id][s + scale * self.window / 2]:
            y[label.data[1] + shift] = 1

        return x, y

    def __getitem__(self, index):
        """
        Args:
            index (int): (ignored by this dataset; a random data point is returned)
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        shift = 0
        if self.pitch_shift > 0:
            shift = np.random.randint(-self.pitch_shift, self.pitch_shift)

        jitter = 0.
        if self.jitter > 0:
            jitter_amount = np.random.uniform(-self.jitter, self.jitter)

        rec_id = self.rec_ids[np.random.randint(0, len(self.rec_ids))]
        s = np.random.randint(0, self.records[rec_id][1] - (2. ** ((shift + jitter) / 12.)) * self.window)
        return self.access(rec_id, s, shift, jitter)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
               os.path.exists(os.path.join(self.root, self.test_data)) and \
               os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
               os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree))

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""

        # process and save as torch files
        print('Processing...')

        self.process_data(self.test_data)

        trees = self.process_labels(self.test_labels)
        with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.process_data(self.train_data)

        trees = self.process_labels(self.train_labels)
        with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        print('Download Complete')

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'): continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root, path, item))
            np.save(os.path.join(self.root, path, item[:-4]), data)

    # wite out labels in intervaltrees for fast access
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'r') as f:
                reader = DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument, note, start_beat, end_beat, note_value)
            trees[uid] = tree
        return trees


class MusicNet_song(Dataset):
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]
    metafile = 'musicnet_metadata.csv'

    def __init__(self, root, id, sr, normalize=True, window=16384, hopsize=512, duration=90):
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.hopsize = hopsize
        self.id = id
        self.m = 128
        self.dur = duration
        self.sr = sr

        self.root = os.path.expanduser(root)
        filename = str(id) + '.npy'

        labels_path = None
        for root, _, files in os.walk(self.root):
            for file in files:
                if file == filename:
                    self.data = np.load(os.path.join(root, file)).astype(np.float32)
                    if self.train_data in root:
                        labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
                    else:
                        labels_path = os.path.join(self.root, self.test_labels, self.test_tree)
                    break

        if labels_path:
            with open(labels_path, 'rb') as f:
                trees = pickle.load(f)
                self.label = trees[id]
        else:
            raise RuntimeError("Can't find the file.")

        self.size = (len(self.data) - window) // hopsize + 1
        if duration:
            self.data = self.data[:sr * duration + window]
            self.size = int(sr * duration / hopsize) + 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        pos = index * self.hopsize
        x = self.data[pos:pos + self.window]
        if self.normalize: x = x / (np.linalg.norm(x) + epsilon)
        y = np.zeros(self.m, dtype=np.float32)
        for label in self.label[pos + self.window // 2]:
            y[label.data[1]] = 1
        return x, y
