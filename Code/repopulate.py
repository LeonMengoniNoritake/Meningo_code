import h5py
import numpy as np
import utils as ut

def main():
    # Load config
    args = ut.parse_args(op_type='noise_year')
    noise_year = args.op_type
    cfg = ut.load_config(args.config)

    ############### MAIN LOGIC
    # Get dataset file
    if '2020' in noise_year:
        dataset_file = cfg.dataset_2019_file
        repop_dataset_file = cfg.dataset_2020_file

    # # Create dataset file for repopulated: only parameters are needed
    # repop_dataset_file = os.path.join(cfg.run_dir, f'dataset_2019_repop.h5')
    
    with h5py.File(dataset_file, 'r') as f:
        attributes_dict = dict(f.attrs)
        repop_n_param_sets = attributes_dict['repop_n_param_sets'] 
        n_param_sets = attributes_dict['n_param_sets']

        # Sample indices
        sampled_indices = np.random.choice(n_param_sets, size=repop_n_param_sets, replace=True)

        # Get parameters
        param_sets = f['param_sets'][:]
        n_params = param_sets.shape[1]

        # Define repopulation array to be filled
        repop_param_sets = np.zeros((repop_n_param_sets, n_params)) # (repopulation target size, n_parameters)

        for i in range(n_params):
            means = param_sets[:,i][sampled_indices,...]
            if 'True' in noise_year:
                sorted_i_set = np.sort(param_sets[:,i])
                sigma = np.mean(np.abs(np.diff(sorted_i_set)))        
                repop_param_sets[:,i] = np.random.normal(loc=means, scale=sigma)
            else:
                repop_param_sets[:,i] = means
    
    repop_dataset_dict = {
        'param_sets': repop_param_sets,
    }

    attributes_dict['n_param_sets'] = repop_n_param_sets
    ut.h5file_add_attributes(repop_dataset_file, attributes_dict)
    ut.h5file_upload(repop_dataset_file, repop_dataset_dict)
    ut.h5file_view_elements(repop_dataset_file)

############### RUN CODE

if __name__ == '__main__':
    main()
