# Copyright 2018 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import time
import random

import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# If zd_merge true, will use zd discriminator that looks at entire batch.
zd_merge = False
device = None
use_cuda = None


def validate(
    model_and_opt_dict, mnist_valid_x, batch_size, BCE_loss,
    y_real_, y_fake_, y_real_z, y_fake_z, zsize,
    device, zd_merge
):
    loss_dict = {"G": 0, "D": 0, "E": 0, "GE": 0, "ZD": 0,}

    D, _ = model_and_opt_dict['D']
    G, _ = model_and_opt_dict['G']
    E, _ = model_and_opt_dict['E']
    ZD, _ = model_and_opt_dict['ZD']

    with torch.no_grad():
        D.eval()
        G.eval()
        E.eval()
        ZD.eval()

        for batch_it in range(len(mnist_valid_x) // batch_size):
            # print('len(valid):',len(mnist_valid_x))
            x = extract_batch(mnist_valid_x, batch_it, batch_size).view(-1, 1, 32, 32)

            #############################################
            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((batch_size, zsize), device=device).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss

            #############################################
            z = torch.randn((batch_size, zsize), device=device).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            #############################################
            z = torch.randn((batch_size, zsize), device=device).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss

            #############################################
            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

            Recon_loss = F.binary_cross_entropy(x_d, x)

            #############################################
            loss_dict['D'] += D_train_loss.item()
            loss_dict['G'] += G_train_loss.item()
            loss_dict['ZD'] += ZD_train_loss.item()
            loss_dict['GE'] += Recon_loss.item()
            loss_dict['E'] += E_loss.item()

            #############################################

        num_valid = len(mnist_valid_x)
        avg_loss = sum(loss_dict.values()) / num_valid

        D.train()
        G.train()
        E.train()
        ZD.train()

        print("Valid Recon avg: {}".format(loss_dict['GE'] / num_valid))
        return avg_loss


def train_an_epoch(
    model_and_opt_dict, mnist_train_x, batch_size, BCE_loss,
    y_real_, y_fake_, y_real_z, y_fake_z, zsize,
    device, zd_merge, subject_folder, inliner_classes, epoch,
):
    loss_dict = {"G": 0, "D": 0, "E": 0, "GE": 0, "ZD": 0,}
    epoch_start_time = time.time()

    D, D_optimizer = model_and_opt_dict['D']
    G, G_optimizer = model_and_opt_dict['G']
    E, E_optimizer = model_and_opt_dict['E']
    _, GE_optimizer = model_and_opt_dict['GE']
    ZD, ZD_optimizer = model_and_opt_dict['ZD']

    for batch_it in range(len(mnist_train_x) // batch_size):
        x = extract_batch(mnist_train_x, batch_it, batch_size).view(-1, 1, 32, 32)
        #print('x:',x[0].shape)
        #############################################
        D.zero_grad()

        D_result = D(x).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z = torch.randn((batch_size, zsize), device=device).view(-1, zsize, 1, 1)
        z = Variable(z)

        x_fake = G(z).detach()
        D_result = D(x_fake).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()

        D_optimizer.step()

        #############################################
        G.zero_grad()

        z = torch.randn((batch_size, zsize), device=device).view(-1, zsize, 1, 1)
        z = Variable(z)

        x_fake = G(z)
        D_result = D(x_fake).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)

        G_train_loss.backward()
        G_optimizer.step()

        #############################################
        ZD.zero_grad()

        z = torch.randn((batch_size, zsize), device=device).view(-1, zsize)
        z = Variable(z)

        ZD_result = ZD(z).squeeze()
        ZD_real_loss = BCE_loss(ZD_result, y_real_z)

        z = E(x).squeeze().detach()

        ZD_result = ZD(z).squeeze()
        ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

        ZD_train_loss = ZD_real_loss + ZD_fake_loss
        ZD_train_loss.backward()

        ZD_optimizer.step()

        #############################################
        E.zero_grad()
        G.zero_grad()

        z = E(x)
        x_d = G(z)

        ZD_result = ZD(z.squeeze()).squeeze()

        E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

        Recon_loss = F.binary_cross_entropy(x_d, x)

        (Recon_loss + E_loss).backward()

        GE_optimizer.step()

        #############################################

        loss_dict['D'] += D_train_loss.item()
        loss_dict['G'] += G_train_loss.item()
        loss_dict['ZD'] += ZD_train_loss.item()
        loss_dict['GE'] += Recon_loss.item()
        loss_dict['E'] += E_loss.item()

        #############################################

        if batch_it == 0:
            directory = os.path.join(
                subject_folder,
                'results'+str(inliner_classes[0])
            )
            if not os.path.exists(directory):
                os.makedirs(directory)

            comparison = torch.cat([x[:64], x_d[:64]])
            save_image(
                comparison.cpu(),
                os.path.join(
                    directory,
                    'reconstruction_' + str(epoch) + '.png',
                ),
                nrow=64
            )

    num_train = len(mnist_train_x)
    for k, _ in loss_dict.items():
        loss_dict[k] /= num_train
    avg_loss = sum(loss_dict.values())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' % (
        (epoch + 1), per_epoch_ptime, loss_dict['G'], loss_dict['D'], loss_dict['ZD'], loss_dict['GE'], loss_dict['E'])
    )
    print("Train Recon avg: {}".format(loss_dict['GE'] / num_train))
    return avg_loss


def setup(x):
    global device
    global use_cuda
    if use_cuda:
        return x.cuda(device)
    else:
        return x.cpu()

def numpy2torch(x):
    return setup(torch.from_numpy(x))

def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)

def list_of_pairs_to_numpy(l):
    return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)


def init_models(zsize, zd_merge, batch_size):
    G = Generator(zsize)
    setup(G)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator()
    setup(D)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(zsize)
    setup(E)
    E.weight_init(mean=0, std=0.02)

    if zd_merge:
        ZD = ZDiscriminator_mergebatch(zsize, batch_size).to(device)
    else:
        ZD = ZDiscriminator(zsize, batch_size).to(device)

    setup(ZD)
    ZD.weight_init(mean=0, std=0.02)

    return G, D, E, ZD

def main(test_fold_id, valid_fold_id, subject_folder, gpu_num, num_epoch, is_fashion, inliner_classes, total_classes, batch_size, zsize, folds=5):
    global use_cuda
    use_cuda = torch.cuda.is_available()
    torch.set_default_tensor_type('torch.FloatTensor')
    

    if use_cuda:
        global device
        device = torch.device("cuda:{}".format(gpu_num))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        np.random.seed(2019)
        random.seed(2019)
        torch.manual_seed(2019)
        torch.cuda.manual_seed_all(2019)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("Running on ", torch.cuda.get_device_name(device))

    data_file_name_format = 'data_fold_%d.pkl'
    if is_fashion:
        data_file_name_format = "f_" + data_file_name_format

    mnist_train = []
    for i in range(folds):
        if i != test_fold_id and i != valid_fold_id:
            with open(data_file_name_format % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            mnist_train += fold

    with open(data_file_name_format % valid_fold_id, 'rb') as pkl:
        mnist_valid = pickle.load(pkl)
    
    
    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inliner_classes]
    # print('mnist_train:',mnist_train[0][1].shape)
    random.shuffle(mnist_train)
    # print('mnist_valid:',len(mnist_valid)) # 13996
    # print("Train set size:", len(mnist_train)) # 4140
    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(mnist_valid)
    
    outlier_classes = [i for i in range(total_classes) if i not in inliner_classes]

    lr = 0.002

    G, D, E, ZD = init_models(zsize, zd_merge, batch_size)
    model_and_opt_dict = {
        "G": [G, optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))],
        "D": [D, optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))],
        "E": [E, optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))],
        "GE": [None, optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))],
        "ZD": [ZD, optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))],
    }
    sample = torch.randn(64, zsize, device=device).view(-1, zsize, 1, 1)

    best_model_info_dict = None
    lowest_valid_loss = np.inf
    loss_going_up_again_cnt = 0
    epoch = 0
    while True:
        for key, (model, opt) in model_and_opt_dict.items():
            if model:
                model.train()

        def shuffle(X):
            np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)
        shuffle(mnist_train_x)

        if (epoch + 1) % 30 == 0:
            for key, (model, opt) in model_and_opt_dict.items():
                opt.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        BCE_loss = nn.BCELoss()

        y_real_ = torch.ones(batch_size, device=device)
        y_fake_ = torch.zeros(batch_size, device=device)

        y_real_z = torch.ones(1 if zd_merge else batch_size, device=device)
        y_fake_z = torch.zeros(1 if zd_merge else batch_size, device=device)

        avg_train_loss = train_an_epoch(
            model_and_opt_dict, mnist_train_x, batch_size, BCE_loss,
            y_real_, y_fake_, y_real_z, y_fake_z, zsize,
            device, zd_merge, subject_folder, inliner_classes, epoch
        )
        avg_valid_loss = validate(
            model_and_opt_dict, mnist_valid_x, batch_size, BCE_loss,
            y_real_, y_fake_, y_real_z, y_fake_z, zsize,
            device, zd_merge
        )
        print("\t- avg train loss({}) / avg valid loss({})".format(avg_train_loss, avg_valid_loss))
        if avg_valid_loss < lowest_valid_loss:
            lowest_valid_loss = avg_valid_loss
            best_model_info_dict = {
                'Gmodel': model_and_opt_dict['G'][0],
                'Emodel': model_and_opt_dict['E'][0],
                'Dmodel': model_and_opt_dict['D'][0],
                'ZDmodel': model_and_opt_dict['ZD'][0],
                'epoch': epoch,
                'lowest_loss': lowest_valid_loss
            }
            loss_going_up_again_cnt = 0
        else:
            loss_going_up_again_cnt += 1
            if num_epoch is None and loss_going_up_again_cnt == 30:
                break

        with torch.no_grad():
            resultsample = model_and_opt_dict['G'][0](sample).cpu()
            directory = os.path.join(
                subject_folder, 'results'+str(inliner_classes[0])
            )
            os.makedirs(directory, exist_ok = True)
            save_image(
                resultsample.view(64, 1, 32, 32),
                os.path.join(
                    directory,
                    'sample_' + str(epoch) + '.png'
                ),
            )

        epoch += 1
        if epoch == num_epoch:
            break

    print("Training finish!... save training results")
    torch.save(G.state_dict(), os.path.join(subject_folder, "Gmodel.pkl"))
    torch.save(E.state_dict(), os.path.join(subject_folder, "Emodel.pkl"))
    torch.save(D.state_dict(), os.path.join(subject_folder, "Dmodel.pkl"))
    torch.save(ZD.state_dict(), os.path.join(subject_folder, "ZDmodel.pkl"))

    if num_epoch:
        torch.save(model_and_opt_dict['G'][0].state_dict(), os.path.join(subject_folder, "Gmodel.pkl"))
        torch.save(model_and_opt_dict['E'][0].state_dict(), os.path.join(subject_folder, "Emodel.pkl"))
        torch.save(model_and_opt_dict['D'][0].state_dict(), os.path.join(subject_folder, "Dmodel.pkl"))
        torch.save(model_and_opt_dict['ZD'][0].state_dict(), os.path.join(subject_folder, "ZDmodel.pkl"))
    else:
        torch.save(best_model_info_dict['Gmodel'].state_dict(), os.path.join(subject_folder, "Gmodel.pkl"))
        torch.save(best_model_info_dict['Emodel'].state_dict(), os.path.join(subject_folder, "Emodel.pkl"))
        torch.save(best_model_info_dict['Dmodel'].state_dict(), os.path.join(subject_folder, "Dmodel.pkl"))
        torch.save(best_model_info_dict['ZDmodel'].state_dict(), os.path.join(subject_folder, "ZDmodel.pkl"))

    with open(os.path.join(subject_folder, "best_epoch_num_is_{}".format(str(epoch))), "w") as f:
        pass

if __name__ == '__main__':
    main(
        test_fold_id=0, valid_fold_id=1, subject_folder="newly_test", gpu_num=0, num_epoch=10,
        is_fashion=False, inliner_classes=[0], total_classes=10, batch_size=128, zsize=32,
    )
