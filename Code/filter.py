import numpy as np
import utils as ut
from scipy.stats import binomtest
import core.ModelClass as ModelClass

def main():
    # Load config
    args = ut.parse_args(op_type='filter_type')
    filter_type = args.op_type
    cfg = ut.load_config(args.config)    

    ############### MAIN LOGIC
    # Define model
    model = ModelClass.RAS_model(
        cfg.serogroup,
        cfg.mod_type,
        cfg.A,
        cfg.data_dir,
        cfg.year_groups,
        cfg.pandemic_years,
        cfg.risk,
        cfg.INCLUDE_VAX
    )

    # Get datasets from dataset_file
    attributes_dict, dataset_dict = ut.h5file_export(cfg.dataset_file)
    n_remaining = attributes_dict['n_param_sets']
    print(f'File contains {n_remaining} parameter sets')

    # Choose filter type: 'carriage' or 'risk'
    if filter_type == 'risk':
        param_names = list(attributes_dict['param_names'])
        param_sets = dataset_dict['param_sets']
        
        # Filter out risk parameter sets that are all 0
        risk_param_names = ['a', 'b', 'c']
        risk_param_idx = [param_names.index(param_name) for param_name in risk_param_names]
        mask = ~np.all(param_sets[:, risk_param_idx] == 0, axis=1)
        for key, dataset in dataset_dict.items():
            dataset_dict[key] = dataset[mask]
        n_param_sets = np.sum(mask)
        n_eliminate = n_remaining - n_param_sets
        print(f'Eliminating {n_eliminate} parameter sets where risk parameter estimation was unsuccessful. Remaining: {n_param_sets}')

        # # Update bounds to reflect this filtering
        bounds = attributes_dict['bounds']
        for idx in risk_param_idx:
            bounds[idx,:] = np.array([np.min(param_sets[:,idx]), np.max(param_sets[:,idx])])
        attributes_dict['bounds'] = bounds

    # elif 'carriage' in filter_type:
    #     # Define carriage prevalence interval
    #     carriage_data_k = model.carriage_data_k[0]
    #     carriage_data_n = model.carriage_data_n[0]
    #     result = binomtest(carriage_data_k, carriage_data_n)
    #     ci = result.proportion_ci(confidence_level=0.95, method='exact')
    #     lower_carr_data = ci.low * 0.5
    #     upper_carr_data = ci.high * 1.5

    #     # Filter out parameter sets with non-finite likelihood
    #     ll_carr = dataset_dict['ll_carr']
    #     finite_mask = np.isfinite(ll_carr)
    #     for key, dataset in dataset_dict.items():
    #         dataset_dict[key] = dataset[finite_mask]
    #     n_eliminate = np.sum(~finite_mask)
    #     n_remaining -= n_eliminate
    #     print(f'Eliminating {n_eliminate} parameter sets corresponding to non-finite likelihood. Remaining: {n_remaining}')

    #     # Filter based on model carriage prevalence of each parameter set being within CI of data
    #     mod_carr_prev = dataset_dict['mod_carr_prev']
    #     filter_indices = np.where((mod_carr_prev >= lower_carr_data) & (mod_carr_prev <= upper_carr_data))[0]
    #     for key, dataset in dataset_dict.items():
    #         dataset_dict[key] = dataset[filter_indices]
    #     n_param_sets = len(filter_indices)
    #     n_eliminate = n_remaining - n_param_sets
    #     print(f'Eliminating {n_eliminate} parameter sets with incompatible carriage prevalence. Remaining: {n_param_sets}')

    # Save attributes to file
    attributes_dict['n_param_sets'] = n_param_sets
    ut.h5file_add_attributes(cfg.dataset_file, attributes_dict)

    # # Save dataset to file
    ut.h5file_upload(cfg.dataset_file, dataset_dict)
    ut.h5file_view_elements(cfg.dataset_file)

    # if filter_type == 'risk':
    #     _, dataset_dict = ut.h5file_export(cfg.dataset_file)
    #     IMD_ll = dataset_dict['IMD_ll']
    #     print(np.sort(IMD_ll)[::-1][:100])

############### RUN CODE

if __name__ == '__main__':
    main()
