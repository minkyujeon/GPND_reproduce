from utils import cifar_reader
import pickle

import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_cifar10(transform=transforms.ToTensor(),):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_set = dsets.CIFAR10(root='cifar10/',
                                   train=True,
                                   transform=transform,
                                   download=True
                                   )
    valid_set = dsets.CIFAR10(root='cifar10/',
                                   train=False,
                                   transform=transform,
                                   download=True
                                   )

    # train_data = train_set.data.float() / 255.
    # train_labels = train_set.targets
    # test_data = valid_set.data.float() / 255.
    # test_labels = valid_set.targets
    
    train_data = train_set.data.astype('float64') / 255.
    train_labels = train_set.targets
    test_data = valid_set.data.astype('float64') / 255.
    test_labels = valid_set.targets

    return train_data, train_labels, test_data, test_labels

_ = load_cifar10()

folds = 5

#Split cifar10 into 5 folds:
cifar10 = items_train = cifar_reader.Reader('cifar-10-batches-bin', train=True, test=True).items
class_bins = {}

random.shuffle(cifar10)

for x in cifar10:

    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)

cifar10_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        cifar10_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(cifar10_folds)):
    print(len(cifar10_folds[i]))

    output = open('cifar10_data_fold_%d.pkl' % i, 'wb')
    pickle.dump(cifar10_folds[i], output)
    output.close()
