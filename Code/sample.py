import h5py
import numpy as np
import utils as ut

def main():
    # Load config
    args = ut.parse_args()
    cfg = ut.load_config(args.config)

    ############### MAIN LOGIC

    dataset_file = cfg.dataset_2020_file
    attributes_dict, dataset_dict = ut.h5file_export(dataset_file)
    bounds = attributes_dict['bounds']
    n_param_sets = attributes_dict['n_param_sets']
    param_names = list(attributes_dict['param_names'])
    param_sets = dataset_dict['param_sets']
    
    # Update attributes: param_names, bounds
    pandemic_param_priors = {
        'zeta2020': [0,1],
        'zeta2021': [0,1],
    }
    pandemic_param_names = list(pandemic_param_priors.keys())
    pandemic_param_bounds = np.array(list(pandemic_param_priors.values()))
    attributes_dict['param_names'] = param_names + pandemic_param_names
    attributes_dict['bounds'] = np.vstack([bounds, pandemic_param_bounds])

    # Update parameter sets with sampled parameters
    pandemic_param_sets = np.random.uniform(pandemic_param_bounds[:,0], pandemic_param_bounds[:,1], (n_param_sets,len(pandemic_param_names)))

    param_sets = np.hstack([param_sets, pandemic_param_sets])
    dataset_dict = {
        'param_sets': param_sets
    }

    # Save attributes to file
    ut.h5file_add_attributes(dataset_file, attributes_dict)

    # Save dataset to file
    ut.h5file_upload(dataset_file, dataset_dict)
    ut.h5file_view_elements(dataset_file)

############### RUN CODE

if __name__ == '__main__':
    main()
