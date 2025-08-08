import os
import h5py
import numpy as np
import pandas as pd
import scipy as sp
import utils as ut
import core.ModelClass as ModelClass

def main():
    # Load config
    args = ut.parse_args(op_type='batch_id')
    batch_id = int(args.op_type)
    cfg = ut.load_config(args.config)

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
    len_year_groups = np.array([len(ut.get_range_from_group(group)) for group in model.year_groups])
    model.avg_IMD_data = (model.IMD_data / len_year_groups.reshape(-1,1)).astype(int)
    years = np.arange(2019, 2024, 1)
    n_years = len(years)

    # Create batch dataset file
    batch_dataset_file = os.path.join(cfg.batch_datasets_dir, f'batch_dataset_{batch_id}.h5')

    # Get number of total parameter sets to be processed by batches
    with h5py.File(cfg.dataset_file, 'r') as f:
        attributes_dict = dict(f.attrs)
    n_param_sets = attributes_dict['n_param_sets']
    
    # Get batch size
    base_n_batch = n_param_sets // cfg.n_batches
    remainder = n_param_sets % cfg.n_batches

    start = (batch_id - 1) * base_n_batch
    end = start + base_n_batch
    
    if batch_id == cfg.n_batches:
        end += remainder
    n_batch = end - start

    # Process batch
    with h5py.File(cfg.dataset_file, 'r') as f:
        mod_carr_inc = f['mod_carr_inc'][start:end]

    # Get absolute number of new carriers (model carriage incidence)
    df = pd.read_csv(os.path.join(model.data_path, 'Population/population.csv'))
    df = df[(df['Region'] == 'Italia') & (df['Age'] != 'Total') & (df['Year'] == 2019)]
    pop_real = df['Pop'].values
    mod_carr_inc_pop = (mod_carr_inc * pop_real[None, None, :, None]).sum(axis=1)

    # Minimize for every parameter set
    a = 2.71
    b = 0.10
    c = -3.89
    risk_params_init = [a, b, c]
    n_risk_params = len(risk_params_init)

    risk_param_sets = np.zeros((n_batch, n_risk_params))

    all_ages = np.arange(model.A)
    avg_mod_IMD = np.zeros((n_batch, n_years, model.n_age_groups))
    IMD_ll = np.zeros(n_batch)

    for i, data in enumerate(mod_carr_inc_pop):
        result = sp.optimize.minimize(
            ut.risk_param_neg_log_likelihood, 
            risk_params_init, 
            args=(
                data, 
                model, 
            ),
            method='L-BFGS-B', 
            bounds = [
                (0, 100),    # a: must be positive, choose upper bound conservatively
                (1e-3, 10),  # b: avoid zero or negative values
                (-10, 0),    # c: may be negative but limited
            ],
            # options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6, 'disp': True}
        )

        if result.success:
            risk_param_sets[i] = result.x
        else:
            print(f'Param set {i}: optimization failed')

        n_print = 10
        if i % n_print == n_print - 1:
            print(f'Optimizing likelihood with parameter set: {i + 1}')
        
        # Save avg_mod_IMD and IMD_ll
        risk_function = ut.CC_Trotter(all_ages, *risk_param_sets[i])
        new_IMD_dataset = (risk_function[:,None] * data)
        avg_mod_IMD[i] = ut.aggregate_sol(new_IMD_dataset, model.age_groups).T
        IMD_likelihood = sp.stats.poisson.logpmf(model.avg_IMD_data, mu=avg_mod_IMD[i])
        IMD_ll[i] = IMD_likelihood.sum()

    # Save dataset to file
    dataset_dict = {
        'risk_param_sets': risk_param_sets,
        'avg_mod_IMD': avg_mod_IMD,
        'IMD_ll': IMD_ll,
    }

    ut.h5file_upload(batch_dataset_file, dataset_dict)
    ut.h5file_view_elements(batch_dataset_file)

############### RUN CODE

if __name__ == '__main__':
    main()
