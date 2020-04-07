import csv


import tensorflow as tf
import numpy as np


from rdkit import Chem


def read_csv(prop, s_name, l_name, seed, shuffle):
    rand_state = np.random.RandomState(seed)
    with open('./data/'+prop+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            (row[s_name], row[l_name]) for row in reader if row[l_name] != ''
        ])
        if shuffle:
            rand_state.shuffle(contents)
    return contents


def atom_feature(atom):

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: float(x == s), allowable_set))


    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()]
    )    


def convert_smiles_to_graph(smi_and_label):    
    smi = smi_and_label[0].numpy()
    label = float(smi_and_label[1].numpy())
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        feature = [atom_feature(atom) for atom in mol.GetAtoms()]
        num_atoms = len(feature)
        adj += np.eye(num_atoms)
        return [feature, adj, label]


def get_slf(prop):

    tox21_labels = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    s_name, l_name, f_name = None, None, None

    if prop in tox21_labels:
        s_name = 'smiles'
        l_name = prop
        f_name = 'tox21'

    else:
        if 'dude' in prop:
            s_name = 'smiles'
            l_name = 'activity'
            f_name = prop

        elif 'chembl' in prop:
            s_name = 'smiles'
            l_name = 'pIC50'
            f_name = prop

        else:

            smiles_dict = {
                'bace_c':'mol',
                'bace_r':'mol',
                'BBBP':'smiles',
                'HIV':'smiles',
                'egfr':'smiles',
                'egfr_chembl':'smiles',
                'logp':'smiles',
                'tpsa':'smiles',
                'sas':'smiles',
            }    
        
            label_dict = {
                'bace_c':'Class',
                'bace_r':'pIC50',
                'BBBP':'p_np',
                'HIV':'HIV_active',
                'egfr':'activity',
                'egfr_chembl':'activity',
                'logp':'prop',
                'tpsa':'prop',
                'sas':'prop',
            }

            s_name = smiles_dict[prop]
            l_name = label_dict[prop]
            f_name = prop

    return s_name, l_name, f_name
 
 
def get_dataset(prop, 
                batch_size,
                train_ratio=0.8, 
                seed=123,
                shuffle=True,
                oversampling=False,
                pos_ratio=0.5):


    s_name, l_name, f_name = get_slf(prop)

    smi_and_label = read_csv(f_name, s_name, l_name, seed, shuffle)
    """
    if oversampling:
        label = smi_and_label[1]
        bool_sample = label != 0
        pos = smi_and_label[bool_sample]
        neg = smi_and_label[~bool_sample]
        pos_ds = tf.data.Dataset.from_tensor_slices(pos)
        neg_ds = tf.data.Dataset.from_tensor_slices(neg)
        total_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[pos_ratio, 1.0-pos_ratio])
        exit(-1)
    else:    
    """
    total_ds = tf.data.Dataset.from_tensor_slices(smi_and_label)

    num_total = smi_and_label.shape[0]
    num_train = int(num_total*train_ratio)

    train_ds = total_ds.take(num_train)
    test_ds = total_ds.skip(num_train)

    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=10*batch_size)
    train_ds = train_ds.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph, 
                                 inp=[x], 
                                 Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=7
    )
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.padded_batch(batch_size, padded_shapes=([None, 58], [None,None], []))
    train_ds = train_ds.cache()

    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph, 
                                 inp=[x], 
                                 Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=7
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.padded_batch(batch_size, padded_shapes=([None, 58], [None,None], []))
    test_ds = test_ds.cache()

    return train_ds, test_ds, num_total, num_train

    
def get_test_dataset(prop, 
                     batch_size,
                     seed=123):

    s_name, l_name, f_name = get_slf(prop)

    smi_and_label = read_csv(f_name, s_name, l_name, seed)
    total_ds = tf.data.Dataset.from_tensor_slices(smi_and_label)

    num_total = smi_and_label.shape[0]
    num_train = int(num_total*train_ratio)

    train_ds = total_ds.take(num_train)
    test_ds = total_ds.skip(num_train)

    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=10*batch_size)
    train_ds = train_ds.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph, 
                                 inp=[x], 
                                 Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=7
    )
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.padded_batch(
        batch_size=batch_size, 
        padded_shapes=([None, 58], [None,None], [])
    )
    train_ds = train_ds.cache()

    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph, 
                                 inp=[x], 
                                 Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=7
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.padded_batch(
        batch_size=batch_size, 
        padded_shapes=([None, 58], [None,None], [])
    )
    test_ds = test_ds.cache()

    return train_ds, test_ds, num_total, num_train
