from torch.utils.data import Dataset
from meghair.utils.imgproc import imdecode
import nori2 as nori


class ImagenetDataset(Dataset):
    """
    Args:
        nori_file(str): nori file contains nori ids and labels, 
            like /data/datasets/ILSVRC2012/imagenet.train.nori.list
        input_size(tuple):
        train(bool): True if trainning else False
        transform(callable, optional): Optional transform to be applied
            on an image
    """

    def __init__(self, nori_file, input_size=(224, 224), train=True, transform=None):
        self.nori_file = nori_file
        self.input_size = input_size
        self.train = train
        self.samples = self._read_nori_file()
        self.transform = transform
        self.fetcher = nori.Fetcher()

    def _read_nori_file(self):
        samples = []
        with open(self.nori_file) as f:
            for line in f:
                nid, label, _ = line.strip().split('\t')
                samples.append({'nid': nid, 'label': int(label)})
        return samples

    def _get_img(self, path):
        data = self.fetcher.get(path)
        image = imdecode(data)[:, :, :3]  # numpy.ndarray
        return image

    def __getitem__(self, idx):
        image = self._get_img(self.samples[idx]['nid'])
        image = image[:, :, ::-1]  #BGR to RGB used by PyTorch
        label = self.samples[idx]['label']  # TODO
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)
