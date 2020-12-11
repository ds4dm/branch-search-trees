""" Converter to transform collected pickle data files to IL inputs. """

import h5py
import numpy as np
import pickle
import argparse
import os
import glob
import torch

# state dimensions
state_dims = {
    'var_dim': 25,
    'node_dim': 8,
    'mip_dim': 53
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--pkl_file_dir',
        type=str,
        help='Pathway to the directory containing all the pkl data collect files.'
    )
    parser.add_argument(
        '--dataset_mode',
        type=str,
        default='train',
        help='Denotes train or val/test data.'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-8,
        help='Numerical stability factor.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Directory to save the h5 and pkl files.'
    )
    args = parser.parse_args()

    # set the NumPy random seed
    np.random.seed(0)

    # stack all the state_vectors into a single matrix
    pkl_paths = sorted(glob.glob(args.pkl_file_dir + '/**/*_data.pkl',  recursive=True))
    state_vectors = np.zeros((0, state_dims.node_dim + state_dims.mip_dim)).astype('float32')
    num_data = 0
    for index, pkl in enumerate(pkl_paths):
        with open(pkl, 'rb') as f:
            print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
            D = pickle.load(f)
            if len(D) > 0:  # We only collect data with more than 1 candidate
                non_trivial_cands_keys = [key for key in D if D[key]['cands_state_mat'].shape[0] > 1]
                num_data += len(non_trivial_cands_keys)
    print('Processed {:d} datapoints...'.format(num_data))

    # create an h5 file
    print('Creating an h5 file...')
    f = h5py.File(os.path.join(args.out_dir, '{}.h5'.format(args.dataset_mode)), 'w')
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    dataset = f.create_dataset('dataset', (num_data,), dtype=dt)
    counter = 0

    for index, pkl in enumerate(pkl_paths):
        with open(pkl, 'rb') as f:
            print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
            D = pickle.load(f)
            for idx in D:
                if D[idx]['cands_state_mat'].shape[0] > 1:  # We only collect data with more than 1 candidate
                    # flat_vector is always [target, node, mip, grid_flattened]
                    flat_vector = np.hstack([D[idx]['varRELpos'], D[idx]['node_state'], D[idx]['mip_state'],
                                             D[idx]['cands_state_mat'].flatten()]).astype('float32')
                    dataset[counter] = flat_vector
                    counter += 1
    print('Processed {:d} datapoints.'.format(counter))
