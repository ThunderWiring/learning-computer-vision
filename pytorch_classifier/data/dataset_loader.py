import abc
import csv
import cv2 as cv
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
import torch
from PIL import Image
from features.image_features import get_numpy_image

from features.image_transformers import BlurTransform, WarpTransform, NoiseTransform

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 0  # multiprocessing doesnt work well on Windows if need to use cuda tensors
          }

CSV_PATH = 'C:/MyProjects/ML/datasets/clothing/images.csv'
DATA_PATH = 'C:/MyProjects/ML/datasets/clothing/images_original/'
IMAGE_FORMAT = 'jpg'


class LocalDataSet(data.Dataset):
    def __init__(self, list_ids, labels, path_to_data, data_suffix):
        self.labels = labels
        self.list_IDs = list_ids
        self.data_path = path_to_data
        self.data_suffix = data_suffix

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        data_path = self.data_path + ID + '.' + self.data_suffix
        datasets.ImageFolder(
            root=data_path, transform=transforms.ToTensor())
        X = torch.load(data_path)
        y = self.labels[ID]

        return X, y


class ImageDataSet(LocalDataSet):
    def __init__(self, list_ids, labels, path_to_data, data_suffix, transform=None):
        super().__init__(list_ids, labels, path_to_data, data_suffix)
        self.transform = transform

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        image_path = self.data_path + ID + '.' + self.data_suffix
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, self.labels[ID]


class DataSetLoaderBase:
    def __init__(
            self, path_to_data, data_suffix, trainig_size=0.5, testing_size=0.1, transform=None):
        assert (trainig_size > 0 and testing_size >
                0 and (trainig_size + testing_size) <= 1)
        self.training_pct = trainig_size
        self.testing_pct = testing_size
        self.validation_pct = 1 - trainig_size - testing_size

        self.list_IDs, self.labels = self.get_data_labels()
        self.unique_lables = self.get_unique_labels(self.labels)

        # Commented out in favor of _get_balance_sets() - see the eda notebook

        # shuffled = np.random.randint(
        #     low=0, high=self.list_IDs.shape[0] - 1, size=(self.list_IDs.shape[0],))
        # shuffled = np.array(shuffled)

        # training_idxs = shuffled[: int(trainig_size * shuffled.shape[0]) - 1]

        # test_start_idx = len(training_idxs)
        # testing_idxs = shuffled[test_start_idx: test_start_idx +
        #                         int(testing_size * shuffled.shape[0]) - 1]
        # validation_idxs = [] if self.validation_pct == 0 else shuffled[len(
        #     training_idxs) + len(testing_idxs)-1:]

        # self.training_IDs = self.list_IDs[training_idxs]
        # self.testing_IDs = self.list_IDs[testing_idxs]
        # self.validation_IDs = self.list_IDs[validation_idxs]

        self.training_IDs, self.validation_IDs, self.testing_IDs = self._get_balance_sets()
        self.data_path = path_to_data
        self.data_suffix = data_suffix

        self.transform = transform

    @abc.abstractclassmethod
    def get_data_labels(self):
        pass

    def get_unique_labels(self, labels):
        dedupped = set()
        all_labels = list(labels.values())
        return [
            x for x in all_labels if x not in dedupped and (dedupped.add(x) or True)]

    def _get_balance_sets(self):
        '''
        The data set might not be balanced (i.e. one class have the majority of the entries)
        Purpose of this function is to assign each category (train, validate and test) a balanced portion.
        '''
        labels_map = {}
        for data_id in self.list_IDs:
            label = self.labels[data_id]
            if label in labels_map:
                labels_map[label].append(data_id)
            else:
                labels_map[label] = [data_id]

        training, validation, testing = [], [], []
        for label in self.unique_lables:
            ids = labels_map[label]
            ids_count = len(ids)
            train_count = int(self.training_pct * ids_count)
            validation_count = int(self.validation_pct * ids_count)

            training += ids[:train_count]
            validation += ids[train_count: train_count + validation_count]
            testing += ids[validation_count + train_count:]

        return training, validation, testing

    def get_training_generator(self):
        training_lables = {k: self.labels[k] for k in self.training_IDs}
        training_set = ImageDataSet(
            self.training_IDs, training_lables, self.data_path, self.data_suffix, transform=self.transform)
        return torch.utils.data.DataLoader(training_set, **params)

    def get_validation_generator(self):
        validation_lables = {k: self.labels[k] for k in self.validation_IDs}
        validation_set = ImageDataSet(
            self.validation_IDs, validation_lables, self.data_path, self.data_suffix, transform=self.transform)
        return torch.utils.data.DataLoader(validation_set, **params)

    def get_test_generator(self):
        test_lables = {k: self.labels[k] for k in self.testing_IDs}
        test_set = ImageDataSet(
            self.testing_IDs, test_lables, self.data_path, self.data_suffix, transform=self.transform)
        return torch.utils.data.DataLoader(test_set, **params)


class GarmentDataSetLoader(DataSetLoaderBase):
    # downloaded set from https://www.kaggle.com/agrigorev/clothing-dataset-full

    def __init__(self, trainig_size=0.5, testing_size=0.1, transform=None):
        super().__init__(
            DATA_PATH, IMAGE_FORMAT, transform=transform)

    def get_data_labels(self):
        list_IDs = []
        labels = {}
        with open(CSV_PATH, mode='r') as garment_dataset:
            reader = csv.reader(garment_dataset)
            next(reader)  # skipping the 1st row as it has only the columns titles
            for row in reader:
                list_IDs.append(row[0])
                labels[row[0]] = row[2]
        return np.array(list_IDs), labels

    def get_unique_lables(self):
        return self.unique_lables


def augment_cloth_images():
    warp_t = WarpTransform()
    blur_t = BlurTransform()
    noise_t = NoiseTransform()
    new_entries = {}
    with open(CSV_PATH, mode='r') as garment_dataset:
        reader = csv.reader(garment_dataset)
        next(reader)  # skipping the 1st row as it has only the columns titles
        for row in reader:
            label = row[2]
            if label == 'Unknown':
                # i looked into this data set and saw this label.
                # to make this work for any dataset, this 'if' is redundant/wrong
                continue
            img = cv.imread(DATA_PATH + row[0] + '.' + IMAGE_FORMAT)
            warp = get_numpy_image(warp_t.transform(img))
            blur = get_numpy_image(blur_t.transform(img))
            noise = get_numpy_image(noise_t.transform(img))
            new_entries[row[0] + '_warp'] = label
            new_entries[row[0] + '_blur'] = label
            new_entries[row[0] + '_noise'] = label
            cv.imwrite(DATA_PATH + row[0] + '_warp.' + IMAGE_FORMAT, warp)
            cv.imwrite(DATA_PATH + row[0] + '_blur.' + IMAGE_FORMAT, blur)
            cv.imwrite(DATA_PATH + row[0] + '_noise.' + IMAGE_FORMAT, noise)

    with open(CSV_PATH, mode='w') as garment_dataset:
        writer = csv.writer(garment_dataset, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for new_img in list(new_entries.keys()):
            writer.writerow([new_img, new_entries[new_img]])
