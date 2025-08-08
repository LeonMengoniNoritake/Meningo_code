import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from core._core_common_imports import *
import core.ModelClass as ModelClass

def main():
    # Load config
    args = ut.parse_args(op_type='batch_id')
    batch_id = int(args.op_type)
    cfg = ut.load_config(args.config)    
    print(f'Running batch simulation {batch_id}/{cfg.n_batches}')

    ############### MAIN LOGIC
    np.random.seed(42 + batch_id)

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

    # Create batch dataset file
    batch_dataset_file = os.path.join(cfg.batch_datasets_dir, f'batch_dataset_{batch_id}.h5')
    
    # Define free parameters and uniform prior intervals
    prior_param_names = list(cfg.priors.keys())

    bounds = np.array([cfg.priors[param_name] for param_name in prior_param_names])
    low = bounds[:, 0]
    high = bounds[:, 1]

    # Run simulations
    n_iter = 10
    n_param_sets_per_iter = 100
    n_param_sets = n_iter * n_param_sets_per_iter
    end_year = 2023

    years = np.arange(2019, end_year+1, 1)
    n_years = len(years)

    # Batch outputs
    # beta_array = np.zeros(n_param_sets_per_iter)
    mod_carr_prev = np.zeros((n_param_sets_per_iter,))
    mod_carr_inc = np.zeros((n_param_sets_per_iter, 3, model.A, n_years)) # 3 is the number of carrier compartments that contribute to IMD

    C_mean = 0.052
    C_sigma = 0.0005 # 0.0014

    print('Starting simulation')
    for iter in range(1, n_iter+1):
        prior_param_sets = np.random.uniform(low, high, (n_param_sets_per_iter, len(cfg.priors)))
        for i, prior_param_set in enumerate(prior_param_sets):
            param_dict = dict(zip(prior_param_names, prior_param_set))
            model.update_params(param_dict)
            # model.mod_params['beta'] = ut.get_correlated_beta_from_gamma(C_mean, C_sigma, gamma=model.mod_params['gamma'])

            start = time.time()
            model.simulate(end_year=end_year, threshold_MSE=1)
            end = time.time()
            duration = end - start
            
            # Print diagnostics
            print(f'Param set {(iter - 1) * n_param_sets_per_iter + i + 1}/{n_param_sets}: {duration:.2f} seconds; {model.n_years} years')

            # Model carriage prevalence: single value
            mod_carriage = model.mod_carriage[0]
            mod_carr_prev[i] = mod_carriage
            
            # Carriage incidence by age: all years from 2019 to 2023
            mask = np.isin(model.years, years)
            new_carriers = model.get_new_carriers(comps='both', grouping='all_ages')[:,:,mask]
            eq_pop = model.get_compartments(comps='population', grouping='all_ages')[:,mask]
            mod_carr_inc_all = new_carriers / eq_pop[None,:,:]
            mod_carr_inc[i] = mod_carr_inc_all[0:3] # keep only incidence of 3 carriage compartments that contribute to IMD 

        # param_sets = np.hstack([prior_param_sets, beta_array[:, np.newaxis]])
        param_sets = prior_param_sets

        # Save dataset to file
        dataset_dict = {
            'param_sets': param_sets,
            'mod_carr_prev': mod_carr_prev,
            'mod_carr_inc': mod_carr_inc,
        }
        ut.h5file_append_or_create(batch_dataset_file, dataset_dict)

        # Empty batch arrays
        # beta_array.fill(0)
        mod_carr_prev.fill(0)
        mod_carr_inc.fill(0)

    ut.h5file_view_elements(batch_dataset_file)

############### RUN CODE

if __name__ == '__main__':
    main()
