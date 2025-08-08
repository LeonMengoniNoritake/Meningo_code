import numpy as np
import pandas as pd
import scipy as sp
import itertools
import h5py
from matplotlib import pyplot as plt
import os
import yaml
from types import SimpleNamespace
import argparse
import re

# Get max age from age groups

def get_max_from_groups(groups):
    maxs = []
    for group in groups:
        maxs.append(max([int(i) for i in group.split('-')]))
    return max(maxs)

# Get age range from age group string: e.g. '0' -> [0], '2-4' -> [2,3,4]

def get_range_from_group(group):
    limits = [int(i) for i in group.split('-')]
    range = np.arange(limits[0], limits[-1]+1)
    return range

def get_list_from_group(group):
    limits = [int(i) for i in group.split('-')]
    list = [f'{i}' for i in range(limits[0], limits[-1]+1)]
    return list

# Get age group midpoints

def get_group_midpoints(groups): 
    midpoints = np.zeros(len(groups))
    for i, group in enumerate(groups):
        limits = [int(j) for j in group.split('-')]
        midpoints[i] = np.mean(limits)
    return midpoints

def get_year_group_idx_from_year(year, year_groups):
    year_list = np.concatenate([get_range_from_group(year_group) for year_group in year_groups])
    if year not in year_list:
        idx = len(year_groups) - 1
    else:
        for i, year_group in enumerate(year_groups):
            if year in get_range_from_group(year_group):
                idx = i
                break
    return idx

# Aggregate populations

def aggregate_pop(df_pop, age_groups):
    dfs = []
    for age_group in age_groups:
        age_list = get_list_from_group(age_group)
        df_agg = df_pop[df_pop['Age'].isin(age_list)].groupby(['Region', 'Year'], as_index=False)['Pop'].sum()
        df_agg['Age'] = age_group
        dfs.append(df_agg)
    df_agg = pd.concat(dfs)
    return df_agg

def aggregate_mort(df_mort, age_groups):
    dfs = []
    for age_group in age_groups:
        # age_group_lims = [int(age) for age in age_group.split('-')]
        # age_list = [f'{age}' for age in range(age_group_lims[0], age_group_lims[-1]+1)]
        age_list = get_list_from_group(age_group)
        df_agg = df_mort[df_mort['Age'].isin(age_list)].groupby(['Region', 'Year'], as_index=False)[['Deaths', 'Pop']].sum()
        df_agg['Age'] = age_group
        dfs.append(df_agg)
    df_agg = pd.concat(dfs, ignore_index=True)
    df_agg['Mortality rate'] = df_agg['Deaths'] / df_agg['Pop'] # add mortality rate column
    return df_agg

# Auxiliary function

def split_string(y_list):
    size = len(y_list)
    values = np.zeros(size)
    intervals = np.zeros((size,2))
    for i, string in enumerate(y_list):
        y = float(string.split(' ')[0])
        values[i] = y
        intervals[i] = np.array([float(v) for v in string.split('(')[1][:-1].split(', ')])
    return values, intervals
 
# Waning rate from fit of antibody titers

def _titer_waning_func(x, r):
    return np.log(100) - r * x

def get_waning_from_fit(x, y, CI):
    y_log = np.log(y)
    CI_log = np.log(CI)
    sigmas = (CI_log[:,1] - CI_log[:,0]) / (2 * 1.96)
    sigmas[sigmas < 0] = 0 # can't have negative sigma values
    popt, pcov = sp.optimize.curve_fit(_titer_waning_func, x, y_log, sigma=sigmas)
    res = _titer_waning_func(x, popt) - y_log
    ss = np.sum(res**2)
    dof = len(y) - len(popt)
    se = np.sqrt(ss / dof * np.diag(pcov))
    t = sp.stats.t.interval(0.95, dof, loc=0, scale=1)[1]
    err = t * se[0]
    wan = popt[0]
    
    if err < wan:
        CI_wan = np.array([wan - err, wan + err])
    else:
        CI_wan = np.array([wan**2 / (wan + err), wan + err])
    return wan, CI_wan

# Case:carrier ratio functions

def transformed_CC_Trotter(x, alpha, beta, gamma):
    return alpha * np.exp(-beta * x) + gamma

def CC_Trotter(x, alpha, beta, gamma):
    return 10**transformed_CC_Trotter(x, alpha, beta, gamma)

def VE_discretized(a, aV, wan_rate):
    VE = (np.exp(- wan_rate * (a - aV)) - np.exp(- wan_rate * (a + 1 - aV))) / wan_rate
    VE[a < aV] = 0
    return VE

# Aggregate simulation solution according to age groups

def aggregate_sol(sol, age_groups):
    sol_group_list = []
    for age_group in age_groups:
        age_range = get_range_from_group(age_group)
        if len(sol.shape) == 3:
            sol_group = sol[:,age_range].sum(axis=1)
        elif len(sol.shape) == 2:
            sol_group = sol[age_range].sum(axis=0)
        sol_group_list.append(sol_group)

    if len(sol.shape) == 3:
        agg_sol = np.stack(sol_group_list, axis=1)
    elif len(sol.shape) == 2:
        agg_sol = np.stack(sol_group_list, axis=0)
    return agg_sol

# Symmetrize and expand contact matrix so that force of infection stays the same

def symmetrize_contact_matrix(K, N):
    N = N.reshape(-1,1)
    E = np.multiply(K, N)
    K_new = (E + E.T) / (2 * N)
    Ks = np.multiply(K_new, N)
    assert np.allclose(Ks, Ks.T) # Verify symmetric matrix
    
    return K_new

def _get_group_idx(n, age_group_list):
    return next(i for i, small_list in enumerate(age_group_list) if n in small_list)

def expand_contact_matrix(K, N, age_groups):
    N = N.reshape(1,-1) # row vector
    N_groups_expanded = np.zeros_like(N)

    age_group_list = []
    for age_group in age_groups:
        # age_group_lims = [int(age) for age in age_group.split('-')] # keep this format because of possible single-year age groups
        # age_list = [age for age in range(age_group_lims[0], age_group_lims[-1]+1)]
        age_range = get_range_from_group(age_group)
        # N_groups_expanded[:,age_group_lims[0]:age_group_lims[-1]+1] = N[:,age_list].sum()
        N_groups_expanded[:,age_range] = N[:,age_range].sum()
        age_group_list.append(age_range)
    n_ages = age_group_list[-1][-1] + 1

    K_expanded = np.zeros((n_ages, n_ages))
    for i,j in itertools.product(np.arange(n_ages), np.arange(n_ages)):
        gi = _get_group_idx(i, age_group_list)
        gj = _get_group_idx(j, age_group_list)
        K_expanded[i,j] = K[gi,gj]
    
    return N / N_groups_expanded * K_expanded 

# Reduce force of infection for infants

def reduce_FOI_infants(K, A, lactamica_age, lactamica_chi):
    """
    Reduce FOI for infants up to fixed age to account for transmission of Neisseria Lactamica.

    Parameters:
    :K: contact matrix
    :A: number of ages
    :lactamica_age: max age with reduced FOI
    :lactamica_chi: percentage susceptibility reduction
    """
    if lactamica_age is None:
        return K
    else:
        X = np.ones(A)
        X[:lactamica_age+1] = lactamica_chi
        X = X.reshape(-1,1)
        return X * K

def reduce_school_contacts(K, xi, age_groups):
    K_new = K.copy()
    for age_group in age_groups:
        range = get_range_from_group(age_group)
        mask = np.ix_(range, range)
        K_new[mask] = K[mask] * xi
    return K_new

def define_chi_array(lactamica_chi, lactamica_age, A):
    chi = np.ones(A)
    chi[:lactamica_age+1] = lactamica_chi
    return chi

# Utility functions for case:carrier ratio fit with Trotter curve

def average_1D_array_over_age_groups(array, age_groups):
    array_avg = np.zeros(len(age_groups))
    for i, age_group in enumerate(age_groups):
        age_range = get_range_from_group(age_group)
        array_avg[i] = np.mean(array[age_range])
    return array_avg

def expand_1D_array_over_age_groups(array, age_groups):
    max_age = get_max_from_groups(age_groups)
    array_exp = np.zeros(max_age+1)
    for i, age_group in enumerate(age_groups):
        age_range = get_range_from_group(age_group)
        array_exp[age_range] = array[i]
    return array_exp

def Trotter_MSE(params, ages, r_data, r_age_groups):
    r = transformed_CC_Trotter(ages, *params)
    r_pred = average_1D_array_over_age_groups(r, r_age_groups)
    return np.sum((r_pred - np.log10(r_data))**2)

def fit_Trotter_curve(r_data, r_age_groups):
    max_age = get_max_from_groups(r_age_groups)
    a = np.arange(max_age+1)
    a_scaled = a / a.max()

    result = sp.optimize.minimize(
        Trotter_MSE,
        x0=[3.2, 13.9, -4.1],
        args=(
            a_scaled,
            r_data[np.isfinite(r_data)],
            r_age_groups[np.isfinite(r_data)]
        ),
        tol=1e-5
    )
    fit_params = result.x
    r_Trotter = CC_Trotter(a_scaled, *fit_params)
    return fit_params, r_Trotter

# Binomial test

def binomtest_CI(k, n, p=0.5):
    return sp.stats.binomtest(k, n, p).proportion_ci()

# Add year grouping

def add_grouping(df, groups, col):
    dfs = []
    for group in groups:
        range = get_range_from_group(group)
        df_agg = df[df[col].isin(range)].copy()
        df_agg[f'{col} group'] = group
        dfs.append(df_agg)
    df = pd.concat(dfs).sort_index()
    return df

# Get beta dictionary

def get_dict_from_list(list, year_groups):
    dict = {}
    for i, year_group in enumerate(year_groups):
        year_range = get_range_from_group(year_group)
        dict.update({int(year):list[i] for year in year_range})
    return dict

# Gaussian likelihood

def compute_log_norm(tot_inc, mod_tot_inc, sigma_tot_inc):
    log_norm = sp.stats.norm.logpdf(x=tot_inc, loc=mod_tot_inc, scale=sigma_tot_inc)
    return log_norm

# Multinomial likelihood

def compute_log_multinom(imd_array, mod_imd_array):
    p = mod_imd_array / mod_imd_array.sum(axis=1).reshape(-1,1)
    log_multinom = sp.stats.multinomial.logpmf(imd_array, n=imd_array.sum(axis=1), p=p)
    return log_multinom

# Auxiliary function

def get_length(x):
    return len(x) if np.ndim(x) > 0 else 1

# Import age groups specific of IMD data

def import_age_groups(data_path, serogroup):
    df = pd.read_csv(os.path.join(data_path, f'IMD/IMD_Men{serogroup}.csv'))
    age_groups = df['Age'].unique()
    return age_groups

def get_epidem_param_for_year(dict, year):
    try:
        param = dict[year]
    except KeyError:
        param = list(dict.values())[0]
    return param

def get_uptake_rate_for_year(A, a, u_dict, year):
    u = np.zeros(A)
    try:
        u[a] = u_dict[year]
    except KeyError:
        pass
    return u

def get_uptake_from_coverage(coverage):
    arr = np.asarray(coverage)
    if np.any((arr < 0) | (arr > 100)):
        raise ValueError('coverage must be a percentage between 0 and 100!')
    else:
        return - np.log(1 - coverage / 100)

def create_single_legend(fig, axs, **kwargs):
    # If axs is not iterable, make it a list
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    elif isinstance(axs, np.ndarray):
        axs = axs.flatten()

    handles = []
    labels = []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates while preserving order
    unique = dict(zip(labels, handles))
    unique_labels = list(unique.keys())
    unique_handles = list(unique.values())

    # Put legend on one of the axes or on the figure
    fig.legend(unique_handles, unique_labels, **kwargs)

def get_param_label(param_name):
    if param_name == 'alpha': param_label = r'$\alpha$'
    if param_name == 'beta': param_label = r'$\beta$'
    if param_name == 'gamma': param_label = r'$\gamma$'
    if param_name == 'delta': param_label = r'$\delta$'
    if param_name == 'chi_value': param_label = r'$\chi}$'
    if param_name == 'zeta20': param_label = r'$\zeta_{20}$'
    if param_name == 'zeta21': param_label = r'$\zeta_{21}$'
    if param_name == 'xi20': param_label = r'$\xi_{20}$'
    if param_name == 'xi21': param_label = r'$\xi_{21}$'
    if param_name == 'z': param_label = r'$z$'

    return param_label

# Functions for h5py

def h5file_add_attributes(h5file, attributes_dict, mode='a'):
    """Add attributes to HDF5 file."""
    with h5py.File(h5file, mode) as f:
        for name, attribute in attributes_dict.items():
            f.attrs[name] = attribute

def h5file_append_or_create(h5file, dataset_dict, mode='a', compression='gzip'):
    """Append a new chunk to a dataset in the HDF5 file, or create it if it doesn't exist."""
    with h5py.File(h5file, mode) as f:
        for name, data in dataset_dict.items():
            if name not in f:
                maxshape = (None,) + data.shape[1:]  # growable first axis
                f.create_dataset(name, data=data, maxshape=maxshape, chunks=True, compression=compression)
            else:
                dset = f[name]
                dset.resize((dset.shape[0] + data.shape[0]), axis=0) # resize the dataset to make room for the new chunk
                dset[-data.shape[0]:] = data # append chunk

def h5file_upload(h5file, dataset_dict, mode='a', compression='gzip'): # overwrite just the datasets in dataset dict
    with h5py.File(h5file, mode) as f:
        for name, data in dataset_dict.items():
            if name in f: del f[name]
            f.create_dataset(name, data=data, compression=compression)

def h5file_view_elements(h5file):
    with h5py.File(h5file, 'r') as f:
        def print_structure(name, obj): # list all groups and datasets
            print(name, obj, flush=True)

        f.visititems(print_structure)

        print("\nAttributes:", flush=True) # list attributes
        for key, value in f.attrs.items():
            print(f"{key}: {value}", flush=True)

def h5file_export(h5file):
    """Export attributes and datasets of h5file as dictionaries"""
    dataset_dict = {}
    with h5py.File(h5file, 'r') as f:
        attributes_dict = dict(f.attrs)
        for key, dataset in f.items():
            dataset_dict[key] = dataset[:]
    return attributes_dict, dataset_dict

# Corner plot

def plot_corner(param_array, limits=None, param_names=None, save_path=None, s=None, alpha=None, **kwargs):
    """
    :param_array: must be of shape (n_samples, n_params)
    """

    n_samples, n_params = param_array.shape
    if limits is not None:
        param_names = list(limits.keys())
    fig, axs = plt.subplots(n_params, n_params, **kwargs)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(n_params):
        for j in range(n_params):
            ax = axs[i,j]
            if limits is not None:
                xlims = limits[param_names[j]]
                ylims = limits[param_names[i]]

            if i == j: # Diagonal: histogram
                ax.hist(param_array[:,i], bins=30, color='steelblue', edgecolor='black')
                if limits is not None:
                    ax.set_xlim(xlims)
            elif i > j: # Lower triangle: scatter plot
                ax.scatter(param_array[:,j], param_array[:, i], s=s, alpha=alpha)
                if limits is not None:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
            else: # Upper triangle: turn off axis
                ax.axis('off')

            # Labeling
            if i == n_params - 1 and j <= i:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticks([])

            if j == 0 and i >= j:
                ax.set_ylabel(param_names[i])
            else:
                ax.set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

# Fit Trotter function parameters by optimizing function
def risk_param_log_likelihood(theta, data, model):
    # all_ages = np.arange(model.A)
    # risk_function = CC_Trotter(all_ages, *theta)
    # rho_array = np.array([1,1,1,0,0,0])
    # IMD_contributors = rho_array[:,None,None] * data # new carriers that can develop IMD (from compartments with subindex 0)
    # new_IMD_dataset = IMD_contributors * risk_function[None,:,None] # risk function applied only to new carriers of 3 compartments corresponding to no previous infection. 
    # new_IMD_agg_dataset = aggregate_sol(new_IMD_dataset, model.age_groups).sum(axis=0) # sum over all compartments
    # IMD_likelihood = sp.stats.poisson.logpmf(model.avg_IMD_data, mu=new_IMD_agg_dataset.T)

    all_ages = np.arange(model.A)
    risk_function = CC_Trotter(all_ages, *theta)
    new_IMD_dataset = (risk_function[:,None] * data)
    new_IMD_agg_dataset = aggregate_sol(new_IMD_dataset, model.age_groups).T
    IMD_likelihood = sp.stats.poisson.logpmf(model.avg_IMD_data, mu=new_IMD_agg_dataset)

    return IMD_likelihood

def risk_param_neg_log_likelihood(theta, data, model):
    IMD_likelihood = risk_param_log_likelihood(theta, data, model)
    tot_likelihood = IMD_likelihood.sum() # is a function of theta = (alpha, beta, gamma)
    return -tot_likelihood

def parse_args(op_type=None):
    parser = argparse.ArgumentParser()

    if op_type is not None:
        parser.add_argument('op_type', type=str, help='Type of operation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

def get_correlated_beta_from_gamma(C_mean, C_std, gamma_duration=None, gamma=None):
    if (gamma_duration is None) == (gamma is None):
        raise ValueError('You must provide exactly one of `gamma_duration` or `gamma`')

    if gamma_duration is not None:
        gamma = 1 / gamma_duration

    C = -1
    while C <= 0:
        C = np.random.normal(C_mean, C_std)
    beta = C * gamma

    return beta

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else -1  # fallback if no number found
