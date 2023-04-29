import os
import numpy as np
import pickle
import torch
from torch import nn
from datetime import date
import torch.nn.functional as F
import torch.optim as optim
from RNALifeTime.Callbacks import EarlyStopping
from RNALifeTime.Models import DegradationModel, MLPModel
from RNALifeTime.Preprocessing import get_data, preprocess_data
from RNALifeTime.Utils import test, batch_generator

from argparse import ArgumentParser
import logging

logger = logging.getLogger()


def run(arg, seqs_per_traj=501):
    if arg.dtype == 'float16':
        dtype = np.float16
    elif arg.dtype == 'float32':
        dtype = np.float32
    else:
        dtype = np.float64

    base_path = os.path.normpath(arg.path)
    with open(os.path.join(base_path, 'residues.pkl'), 'rb') as file:
        residues = pickle.load(file)
    with open(os.path.join(base_path, 'temperature_data.pkl'), 'rb') as file:
        temperature_data = pickle.load(file)
    with open(os.path.join(base_path, 'Mg2_data.pkl'), 'rb') as file:
        Mg2_data = pickle.load(file)
    with open(os.path.join(base_path, 'location.pkl'), 'rb') as file:
        locations = pickle.load(file)
    with open(os.path.join(base_path, 'contact_matrix.pkl'), 'rb') as file:
        contact_matrix = pickle.load(file)
    with open(os.path.join(base_path, 'contacts.pkl'), 'rb') as file:
        contacts = pickle.load(file)
    with open(os.path.join(base_path, 'seq_code.pkl'), 'rb') as file:
        seq_code = pickle.load(file)
    path_to_data = os.path.join(base_path, 'prepared')
    if arg.preprocessed:
        with open(os.path.join(path_to_data, 'contact_mask.pkl'), 'rb') as file:
            contact_mask = pickle.load(file)
        with open(os.path.join(path_to_data, 'temperatures.pkl'), 'rb') as file:
            temperatures = pickle.load(file)
        with open(os.path.join(path_to_data, 'mgs.pkl'), 'rb') as file:
            mgs = pickle.load(file)
        with open(os.path.join(path_to_data, 'distances.pkl'), 'rb') as file:
            distances = pickle.load(file)
        with open(os.path.join(path_to_data, 'targets.pkl'), 'rb') as file:
            targets = pickle.load(file)
    else:
        contact_mask, temperatures, mgs, distances, targets = \
            get_data(locations, seq_code, temperature_data, Mg2_data, contacts, path_to_data)

    max_len = max(max(map(lambda x: len(x), list(residues))), arg.max_length)

    distance_torch, condition_data, contacts_mask, frames, bonds, Y = \
        preprocess_data(distances, temperatures, mgs, targets, contact_mask, contact_matrix, max_len,
                        seqs_per_traj, dtype=dtype)
    test_start2 = len(distances) // 3
    test_start = test_start2 * 500

    distance_torch_train, distance_torch_test = distance_torch[test_start:], distance_torch[:test_start]
    condition_data_train, condition_data_test = condition_data[test_start:], condition_data[:test_start]
    contacts_mask_train, contacts_mask_test = contacts_mask[test_start:], contacts_mask[:test_start]
    frames_train, frames_test = frames[test_start:], frames[:test_start]
    Y_train, Y_test = Y[test_start:], Y[:test_start]
    bonds_train, bonds_test = bonds[test_start2:], bonds[:test_start2]

    del contact_matrix

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(arg.seed)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if arg.model_type == 'RNA-LifeTime':
        model = DegradationModel(ftr_dim=2,
                                 emb=max_len,
                                 num_gaussians=arg.num_gaussians,
                                 mask=None,
                                 contact_type_dim=10,
                                 truncated=arg.truncated).to(device)
    else:
        model = MLPModel(ftr_dim=2, emb=max_len, contact_type_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    early_stopping = EarlyStopping(
        patience=arg.patience, verbose=True, path=
        os.path.join(
            arg.cp_dir,
            'model_checkpoint-gaussian{0}-model-{1}-date-{2}.pt'.format(
                arg.num_gaussians,
                arg.model_type,
                str(date.today().today()),
            )
        )
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=arg.gamma)

    loss_values = []
    test_loss = []
    distance_torch_test, condition_data_test, frames_test, contacts_mask_test, Y_test = \
        torch.as_tensor(distance_torch_test).float(), \
        torch.as_tensor(condition_data_test).float(), \
        torch.as_tensor(frames_test).float(), \
        torch.as_tensor(contacts_mask_test).float(), \
        Y_test

    for epoch in range(1, arg.num_epochs + 1):
        model.train()
        for batch_idx, (dis, cond, frame, mask, bond, target) in enumerate(batch_generator(distance_torch_train,
                                                                                           condition_data_train,
                                                                                           frames_train,
                                                                                           contacts_mask_train,
                                                                                           bonds_train,
                                                                                           Y_train,
                                                                                           arg.batch_size)):
            if len(dis) == 0:
                break
            dis, feature, frame, mask, bond = dis.to(device), cond.to(device), frame.to(device), mask.to(
                device), bond.to(device)  # , target.to(device)
            optimizer.zero_grad()
            output = model(frame, dis, bond, feature, mask)
            loss = F.l1_loss(output, torch.cat(list(target)))
            loss.backward()
            optimizer.step()
            if batch_idx % arg.logging_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Mean Pred: {:.3f}\t Mean True: {:.3f}'.format(
                    epoch, batch_idx * len(dis), len(distance_torch_train),
                           100. * batch_idx / (len(distance_torch_train) // arg.batch_size), loss.item(),
                    output.mean().item(), torch.cat(list(target)).mean().item()))
            loss_values.append(loss.item())

        # model validation
        test_l1_loss, test_mse_loss = test(model, device, distance_torch_test, condition_data_test, frames_test, contacts_mask_test,
                            bonds_test, Y_test)
        test_loss.append(test_l1_loss)
        lr_scheduler.step()
        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(test_l1_loss, model)

        if early_stopping.early_stop:
            print("Early stopping...")
            break


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-lr", "--learning-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.003, type=float)

    parser.add_argument("-c", "--checkpoint_dir", dest="cp_dir",
                        help="Checkpoint directory",
                        default='./MD-simulation/models/')

    parser.add_argument("-m", "--model", dest="model_type",
                        help="Choose model to run",
                        default='RNA-LifeTime')

    parser.add_argument("-p", "--data_dir", dest="path",
                        help="Data loading path",
                        default='./MD-simulation/')

    parser.add_argument("-f", "--preprocessed", dest="preprocessed",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        default=True, type=bool)

    parser.add_argument("-t", "--truncated", dest="truncated",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        default=True, type=bool)

    parser.add_argument("-g", "--num_gau", dest="num_gaussians",
                        help="Number of Gaussian used for modeling solvent-mediated interaction.",
                        default=5, type=int)

    parser.add_argument("-l", "--max_len", dest="max_length",
                        help="Max sequence length. Shorter sequences are padded.",
                        default=72, type=int)

    parser.add_argument("-fh", "--ff_hidden_mult", dest="feedforward_hidden_multiplier",
                        help="Scaling factor that determines the size of the hidden layers relative to the input layers.",
                        default=4, type=int)

    parser.add_argument("-d", "--dropout", dest="dropout",
                        help="Dropout rate.",
                        default=0.2, type=float)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--log_interval",
                        dest="logging_interval",
                        help="Logging interval.",
                        default=10, type=int)

    parser.add_argument("--patience",
                        dest="patience",
                        help="Patience in early stopping.",
                        default=5, type=int)

    parser.add_argument("--step",
                        dest="step_size",
                        help="Period of learning rate decay.",
                        default=2, type=int)

    parser.add_argument("--gamma",
                        dest="gamma",
                        help="Multiplicative factor of learning rate decay.",
                        default=0.3, type=float)

    parser.add_argument("--dtype",
                        dest="dtype",
                        help="Data type.",
                        default='float32', type=str)
    options = parser.parse_args()

    print('OPTIONS ', options)

    run(options)
