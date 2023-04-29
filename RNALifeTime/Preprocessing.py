import os
import pandas as pd
import numpy as np
import torch
import pickle


def soft_cut_q(dis_matrix, dis_matrix0, contact):
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.2
    indices = pd.DataFrame(contact).drop_duplicates().to_numpy()
    boolean_ind = torch.zeros_like(dis_matrix, dtype=bool)
    for i, j in indices:
        if abs(i - j) > 1:
            boolean_ind[i, j] = True
    r = dis_matrix[boolean_ind]
    r0 = dis_matrix0[boolean_ind]
    Q = 1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0)))
    return Q


def get_unique_contact_mask(contact, max_len):
    indices = pd.DataFrame(contact).drop_duplicates().to_numpy()
    boolean_ind = torch.zeros((max_len, max_len), dtype=bool)
    for i, j in indices:
        if abs(i - j) > 1:
            boolean_ind[i, j] = True
    return boolean_ind


def get_data(locations, seq_code, temperature_data, Mg2_data, contacts, path="./MD-simulation/prepared"):
    seqs_per_traj = 501
    max_len = 72
    targets = []
    temperatures = []
    mgs = []
    distances = []
    contact_mask = []
    for i in range(len(locations)):
        if locations[i].shape[0] < seqs_per_traj or seq_code[i] == '1OSU':
            continue
        target_loc = torch.as_tensor(locations[i]).float()
        target_x = torch.cdist(target_loc, target_loc, p=2.0)
        temperatures.append([temperature_data[i]] * (seqs_per_traj - 1))
        mgs.append([Mg2_data[i]] * (seqs_per_traj - 1))
        tem = np.zeros((seqs_per_traj - 1, target_x.shape[1], target_x.shape[2]))
        tem[:] = target_x[0]
        distances.append(tem)
        contact_mask.append(get_unique_contact_mask(contacts[i], max_len))
        targets.append([soft_cut_q(target_x[j], target_x[0], contacts[i]) for j in range(1, seqs_per_traj)])

    with open(os.path.join(path, 'contact_mask.pkl'), 'wb') as file:
        pickle.dump(contact_mask, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'temperatures.pkl'), 'wb') as file:
        pickle.dump(temperatures, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'mgs.pkl'), 'wb') as file:
        pickle.dump(mgs, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'distances.pkl'), 'wb') as file:
        pickle.dump(distances, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'targets.pkl'), 'wb') as file:
        pickle.dump(targets, file, pickle.HIGHEST_PROTOCOL)
    return contact_mask, temperatures, mgs, distances, targets


def preprocess_data(distances, temperatures, mgs, targets, contact_mask, contact_matrix, max_len, seqs_per_traj,
                    dtype=np.float64):
    padded_distance = []
    padded_temp = []
    padded_con = []
    padded_bonds = []
    padded_y = []
    Y = []
    frames = []
    seqs_per_traj_ = seqs_per_traj - 1
    for i in range(len(distances)):
        padded_seq = np.zeros((seqs_per_traj_, max_len, max_len), dtype=dtype)
        padded_seq[:distances[i].shape[0], :distances[i].shape[1], :distances[i].shape[2]] = distances[i]
        padded_distance.append(padded_seq)

        padded_bond = np.zeros((1, max_len, max_len, 10), dtype=dtype)
        padded_bond[0, :contact_matrix[i].shape[0], :contact_matrix[i].shape[1], :] = contact_matrix[i]
        padded_bonds.append(padded_bond)

        padded_temp.append(np.array([temperatures[i], mgs[i]]).T)

        padded_contact = np.zeros((seqs_per_traj_, max_len, max_len), dtype=dtype)
        padded_contact[:, :contact_mask[i].shape[0], :contact_mask[i].shape[1]] = contact_mask[i]
        padded_con.append(padded_contact)
        frames.append(list(range(1001))[2:1001:2])
        padded_y.extend(targets[i])

    distance = np.vstack(padded_distance)  # (#, 50, 50)
    bonds = np.vstack(padded_bonds)
    condition_data = np.vstack(padded_temp)
    contacts_mask = np.vstack(padded_con)
    frames = np.concatenate(frames)
    Y = np.array(padded_y)
    return distance, condition_data, contacts_mask, frames, bonds, Y