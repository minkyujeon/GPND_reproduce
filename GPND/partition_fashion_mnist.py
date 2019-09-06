from utils import mnist_reader
import pickle

import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_fmnist(transform=transforms.ToTensor()):
    train_set = dsets.FashionMNIST(root='fmnist/',
                                   train=True,
                                   transform=transform,
                                   download=True
                                   )
    valid_set = dsets.FashionMNIST(root='fmnist/',
                                   train=False,
                                   transform=transform,
                                   download=True
                                   )

    train_data = train_set.train_data.float() / 255.
    train_labels = train_set.train_labels
    test_data = valid_set.test_data.float() / 255.
    test_labels = valid_set.test_labels

    return train_data, train_labels, test_data, test_labels

_ = load_fmnist()

folds = 5

#Split mnist into 5 folds:
mnist = items_train = mnist_reader.Reader('fmnist/raw', train=True, test=True).items
class_bins = {}

random.shuffle(mnist)

for x in mnist:
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)

mnist_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(mnist_folds)):
    print(len(mnist_folds[i]))

    output = open('f_data_fold_%d.pkl' % i, 'wb')
    pickle.dump(mnist_folds[i], output)
    output.close()
