import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import gzip
import struct


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)

        h, w, c = img.shape
        pd = self.padding
        padded_img = np.pad(img, ((pd, pd), (pd, pd), (0, 0)), mode='constant')
        cropped_img = padded_img[shift_x + pd:shift_x + h + pd,
                                 shift_y + pd:shift_y + w + pd, :]
        assert cropped_img.shape == (h, w, c), 'return shape should be hxwxc'
        return cropped_img



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sample, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
     
    '''
    Members:
        ordering: the sublists of the index list of dataset
    '''
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        # split the index list with the value in range(batch_size, len(dataset), batch_size)
        if not self.shuffle:
            index_list = np.arange(len(dataset))
            split_points = range(batch_size, len(dataset), batch_size)
            self.ordering = np.array_split(index_list, split_points)

    def __iter__(self):
        '''
        We firstlt get the index list of dataset, i.e. index_list,
        then shuffle it randomly,
        and finally split it into sublists.
        '''
        self.index = 0
        if self.shuffle:
            index_list = np.arange(len(self.dataset))
            np.random.shuffle(index_list)
            split_points = range(self.batch_size, len(index_list), self.batch_size)
            self.ordering = np.array_split(index_list, split_points)
        return self

    def __next__(self):
        if self.index >= len(self.ordering):
            raise StopIteration
        batch_index = self.ordering[self.index]
        # mini_batch has 2 element: sample and label
        mini_batch = [Tensor(data) for data in self.dataset[batch_index]]
        self.index += 1
        return mini_batch

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:

        # MNIST stores data in big-endian format
        # load header variables
        _ = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        # load images pixels
        images_data = f.read(num_images * num_rows * num_cols)
        images = struct.unpack('>' + 'B' * len(images_data), images_data)

        # reshape images
        images = np.array(images).reshape(num_images, num_rows * num_cols)

        # normalize the dataset
        images = images / 255

        # convert the data to fp32
        images = images.astype(np.float32)

    with gzip.open(label_filename, 'rb') as f:

        # load header variables
        _ = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        # load images pixels
        labels_data = f.read(num_labels)
        labels = struct.unpack('>' + 'B' * len(labels_data), labels_data)

        # reshape images
        labels = np.array(labels)

        # convert the data to uint8
        labels = labels.astype(np.uint8)

    return (images, labels)
    ### END YOUR CODE

class MNISTDataset(Dataset):
    '''
    Notes:
        1. before doing transforms, need to reshape sample first
        2. when call __getitem_, index can be a slice
    '''
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.samples, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        batch_samples = self.samples[index]
        batch_labels = self.labels[index]
        if self.transforms == None:
            return batch_samples, batch_labels
        return self.apply_transforms(batch_samples.reshape(28, 28, 1)), batch_labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.samples.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
