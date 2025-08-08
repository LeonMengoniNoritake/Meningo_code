import os
import h5py
import numpy as np
import utils as ut
import sys
import glob
from collections import defaultdict

def main():
    # Load config
    args = ut.parse_args(op_type='agg_type')
    agg_type = args.op_type
    cfg = ut.load_config(args.config)

    ############### MAIN LOGIC
    # Get h5 files 
    h5_files = sorted(
        glob.glob(os.path.join(cfg.batch_datasets_dir, '*.h5')),
        key=ut.extract_number
    )
    if len(h5_files) == 0:
        print(f'No .h5 files found in {cfg.batch_datasets_dir}')
        sys.exit(1)
    elif len(h5_files) != cfg.n_batches:
        print(f'Expected {cfg.n_batches} .h5 files but found {len(h5_files)} in {cfg.batch_datasets_dir}')
        sys.exit(1)

    print(f'Found {len(h5_files)} files to aggregate.\n')

    # Aggregate them in a single dict
    dataset_dict = defaultdict(list)
    for batch_file in h5_files:
        _, batch_dataset_dict = ut.h5file_export(batch_file)
        for key, array in batch_dataset_dict.items():
            dataset_dict[key].append(array)
    dataset_dict = {key: np.concatenate(arrays, axis=0) for key, arrays in dataset_dict.items()}

    if agg_type == 'all': # first aggregation, so all attributes must be defined as if it was the first time
        # Define attributes
        n_param_sets = len(dataset_dict[list(dataset_dict.keys())[0]])

        # # Define target number of parameter sets when repopulating 
        # repop_n_param_sets = n_param_sets

        # Define all parameter names
        prior_param_names = list(cfg.priors.keys())
        # param_names = prior_param_names + ['beta']
        param_names = prior_param_names
        
        # Define all parameter bounds
        prior_bounds = np.array([cfg.priors[param_name] for param_name in prior_param_names])        
        # beta_idx = param_names.index('beta')
        # beta_array = dataset_dict['param_sets'][:,beta_idx]
        # beta_bounds = np.array([np.min(beta_array), np.max(beta_array)])
        # bounds = np.vstack([prior_bounds, beta_bounds])
        bounds = prior_bounds

        attributes_dict = {
            'n_param_sets': n_param_sets,
            # 'repop_n_param_sets': repop_n_param_sets,
            'param_names': param_names,
            'bounds': bounds,
        }

    elif agg_type == 'risk': # aggregation after finding risk parameters
        # Get attributes and datasets from file
        attributes_dict, dataset_2023_dict = ut.h5file_export(cfg.dataset_file)
        param_names = list(attributes_dict['param_names'])
        bounds = attributes_dict['bounds']

        # Define all parameter names
        risk_param_names = ['a', 'b', 'c']
        param_names += risk_param_names
        
        # Define all parameter bounds
        for risk_param_name in risk_param_names:
            risk_param_idx = risk_param_names.index(risk_param_name)
            risk_param_array = dataset_dict['risk_param_sets'][:,risk_param_idx]
            risk_param_bounds = np.array([np.min(risk_param_array), np.max(risk_param_array)])
            bounds = np.vstack([bounds, risk_param_bounds])
        
        attributes_dict.update({
            'param_names': param_names,
            'bounds': bounds,
        })

        # Stack parameter sets and redefine dataset_dict
        param_sets = dataset_2023_dict['param_sets']
        risk_param_sets = dataset_dict.pop('risk_param_sets')
        
        dataset_dict['param_sets'] = np.hstack([param_sets, risk_param_sets])

    # Save attributes to file
    ut.h5file_add_attributes(cfg.dataset_file, attributes_dict)

    # Save dataset to file
    ut.h5file_upload(cfg.dataset_file, dataset_dict)
    ut.h5file_view_elements(cfg.dataset_file)

    # if agg_type == 'risk':
    #     _, dataset_dict = ut.h5file_export(cfg.dataset_file)
    #     IMD_ll = dataset_dict['IMD_ll']
    #     print(np.sort(IMD_ll)[::-1][:100])

############### RUN CODE

if __name__ == '__main__':
    main()
