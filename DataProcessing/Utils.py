from bs4 import BeautifulSoup
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import mdtraj as md
import copy
import os

CODE = ['AU', 'AC', 'AG', 'CU', 'CG', 'GU', 'AA', 'UU', 'CC', 'GG']
CODE = np.array(CODE)


def get_contacts(Mg2, RNA_id, path):
    with open(path, 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "xml")
    atom_contacts = [[int(c['i']) - 1, int(c['j']) - 1] for c in Bs_data.find('contacts').find_all('interaction')]
    return atom_contacts


def get_atom_residue(t_RNA, ailun_simulation=False):
    atom_dict = md.geometry.dihedral._construct_atom_dict(t_RNA.topology)[0]
    if ailun_simulation:
        atom_keys = copy.deepcopy(set(atom_dict.keys()))
        for i in atom_keys:
            if i > 70:
                del atom_dict[i]
                continue
            for name in [i for i in list(atom_dict[i].keys())]:
                j = atom_dict[i][name]
                if "*" in name:
                    del atom_dict[i][name]
                    atom_dict[i][name.replace("*", "\'")] = j
    else:
        for i in atom_dict.keys():
            for name in [i for i in list(atom_dict[i].keys())]:
                j = atom_dict[i][name]
                if "*" in name:
                    del atom_dict[i][name]
                    atom_dict[i][name.replace("*", "\'")] = j
    return atom_dict


def get_atom2residue2(t_RNA, RNA_id, Mg2, rep_id, ailun_simulation=False, path='/scratch/zheng.hua1/processed/'):
    temperature = 0.5
    if ailun_simulation:
        dcd_path = '/scratch/whitford/A-riboswitch/K50Mg{}/temp_{}/REP{}/A-riboswitch_opensmog_pbc_trajectory.dcd'.format(
            "0.10", temperature, rep_id)
        path_gro = '/scratch/whitford/A-riboswitch/K50Mg{0}/temp_{1}/system/A-ribo_opensmog.Mg{0}.mgkcl.gro'.format(
            "0.10", temperature)
        path_xml = '/scratch/whitford/A-riboswitch/K50Mg{0}/temp_{1}/system/A-ribo_opensmog.Mg{0}.mgkcl.xml'.format(
            "0.10", temperature)

    else:
        path_gro = '/scratch/zheng.hua1/processed/processed-MG{}-CL/{}.OpenSMOG.AA+custom+ions.MGKCL.gro'.format(Mg2,
                                                                                                                 RNA_id)
        path_xml = '/scratch/zheng.hua1/processed/processed-MG{}-CL/{}.OpenSMOG.AA+custom+ions.MGKCL.xml'.format(Mg2,
                                                                                                                 RNA_id)

    atom_contacts = get_contacts(Mg2, RNA_id, path_xml)
    atom_dict = get_atom_residue(t_RNA, ailun_simulation)
    indices = []
    for i, j in atom_dict.items():
        for k, v in j.items():
            indices.append([v, i])
    indices_dict = dict(indices)
    if ailun_simulation:
        contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts if
                    i in indices_dict and j in indices_dict and indices_dict[i] <= 70 and indices_dict[j] <= 70]
        # contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts if i in indices_dict and j in indices_dict]
    else:
        contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts]
    contact_matrix = contact_format(t_RNA, contacts, ailun_simulation)
    return contacts, contact_matrix


def get_atom2residue(RNA_id, Mg2, rep_id, run_group, ailun_simulation=False, path='/scratch/zheng.hua1/processed/'):
    temperature = 0.5
    if ailun_simulation:
        dcd_path = '/scratch/whitford/A-riboswitch/K50Mg{}/temp_{}/REP{}/A-riboswitch_opensmog_pbc_trajectory.dcd'.format(
            "0.10", temperature, rep_id)
        path_gro = '/scratch/whitford/A-riboswitch/K50Mg{0}/temp_{1}/system/A-ribo_opensmog.Mg{0}.mgkcl.gro'.format(
            "0.10", temperature)
        path_xml = '/scratch/whitford/A-riboswitch/K50Mg{0}/temp_{1}/system/A-ribo_opensmog.Mg{0}.mgkcl.xml'.format(
            "0.10", temperature)

    else:
        path_gro = '/scratch/zheng.hua1/processed/processed-MG{}-CL/{}.OpenSMOG.AA+custom+ions.MGKCL.gro'.format(Mg2,
                                                                                                                 RNA_id)
        path_xml = '/scratch/zheng.hua1/processed/processed-MG{}-CL/{}.OpenSMOG.AA+custom+ions.MGKCL.xml'.format(Mg2,
                                                                                                                 RNA_id)
        if rep_id == 0:
            rep_id = ''
            dcd_path = os.path.join(path, 'RNA{0}-Mg{1}-NEW/simulation_result'.format(run_group, Mg2),
                                    'output_{}_MG{}_temperature{}/{}_trajectory.dcd'.format(RNA_id, Mg2, temperature,
                                                                                            RNA_id))
        else:
            dcd_path = os.path.join(path, 'RNA{0}-Mg{1}-NEW/simulation_result'.format(run_group, Mg2),
                                    'output_{}_MG{}_temperature{}/{}_trajectory.dcd_{}'.format(RNA_id, Mg2, temperature,
                                                                                               RNA_id, rep_id))

    atom_contacts = get_contacts(Mg2, RNA_id, path_xml)
    t = md.load_dcd(dcd_path, top=path_gro)
    t_RNA = t.remove_solvent()
    atom_dict = get_atom_residue(t_RNA, ailun_simulation)
    indices = []
    for i, j in atom_dict.items():
        for k, v in j.items():
            indices.append([v, i])
    indices_dict = dict(indices)
    if ailun_simulation:
        contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts if
                    i in indices_dict and j in indices_dict and indices_dict[i] <= 70 and indices_dict[j] <= 70]
        # contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts if i in indices_dict and j in indices_dict]
    else:
        contacts = [[indices_dict[i], indices_dict[j]] for i, j in atom_contacts]
    contact_matrix = contact_format(t_RNA, contacts, ailun_simulation)
    return contacts, contact_matrix


def contact_format(t_RNA, contacts, ailun_simulation):
    if ailun_simulation:
        residues = [r.name[:1] for r in t_RNA.topology.residues if r.resSeq <= 71]
    else:
        residues = [r.name[:1] for r in t_RNA.topology.residues]
    contact_types = []
    for a, b in contacts:
        i, j = residues[a], residues[b]
        if i < j:
            ij = i + j
        else:
            ij = j + i
        contact_types.append(ij)

    label_encoder = LabelEncoder()
    label_encoder.fit(CODE)
    vec = label_encoder.transform(contact_types)
    contact_types = to_categorical(vec, num_classes=len(CODE))  # .reshape(len(residues), len(residues), len(code))
    contact_matrix = np.zeros((len(residues), len(residues), len(CODE)))
    for i, (a, b) in enumerate(contacts):
        contact_matrix[a][b] = contact_types[i]
    return contact_matrix
