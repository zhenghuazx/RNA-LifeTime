import torch
import numpy as np
import torch.nn.functional as F


def get_triangle_index(start, length, factor):
    duplicated_idx = []
    for i in range(length):
        duplicated_idx.append([start + i] * ((length - i) * factor))
    return np.concatenate(duplicated_idx)


def batch_generator(distance_torch, condition_data, frames, contacts_mask, bonds, Y, batch_size=64, shuffle=True,
                    upsample=True):
    indices = np.arange(len(distance_torch))
    if upsample:
        indices = np.arange(len(distance_torch))[::2]  # np.arange(len(distance_torch))
        upsample_indices = np.concatenate(
            [get_triangle_index(i * 500, 25, 1) for i in range(len(distance_torch) // 500)])
        indices = np.concatenate([indices, upsample_indices])
    batch = []
    while True:
        # it might be a good idea to shuffle your data before each epoch
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                batch_bonds = np.array([bonds[i // 500] for i in batch])
                yield torch.as_tensor(distance_torch[batch]).float(), torch.as_tensor(
                    condition_data[batch]).float(), torch.as_tensor(frames[batch]).float(), torch.as_tensor(
                    contacts_mask[batch]).float(), torch.as_tensor(batch_bonds).float(), Y[
                          batch]  # torch.as_tensor(Y[batch]).float()
                batch = []
        return [], [], [], [], [], []


def test(model, device, distance_torch, condition_data, frames_data, contacts_mask, bonds, Y, BATCH_SIZE=512 * 4):
    model.eval()
    test_loss, test_mse_loss = 0, 0
    test_loss_list, test_mse_loss_list = [], []
    with torch.no_grad():
        for batch_idx, (dis, cond, frame, mask, bond, target) in enumerate(batch_generator(distance_torch,
                                                                                           condition_data,
                                                                                           frames_data,
                                                                                           contacts_mask,
                                                                                           bonds,
                                                                                           Y,
                                                                                           BATCH_SIZE)):
            dis, feature, frame, mask, bond = dis.to(device), cond.to(device), frame.to(device), mask.to(
                device), bond.to(device)
            output = model(frame, dis, bond, feature, mask)
            l1_loss = F.l1_loss(output, torch.cat(list(target)))
            mse_loss = F.mse_loss(output, torch.cat(list(target)))
            test_mse_loss += mse_loss.item() * BATCH_SIZE
            test_mse_loss_list.append(mse_loss.item() * BATCH_SIZE)
            test_loss += l1_loss.item() * BATCH_SIZE
            test_loss_list.append(l1_loss.item() * BATCH_SIZE)
    print('\nTest set: MAE loss: {:.4f}; MSE loss: {:.4f}\n'.format(
        test_loss / len(distance_torch), test_mse_loss / len(distance_torch)))
    return test_loss / len(distance_torch), test_mse_loss / len(distance_torch)
