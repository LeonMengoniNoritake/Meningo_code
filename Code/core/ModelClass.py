from ._core_common_imports import *

class RAS_model():
    def __init__(self, serogroup, mod_type, A, data_path, year_groups, pandemic_years, risk='Trotter', INCLUDE_VAX=False):
        """
        Class initialization. No espidemiological parameters among class arguments

        Parameters:
        :serogroup: 'B' or 'C'
        :mod_type: 'SCS' or 'SCR'
        :A: number of single-year age groups; e.g. A = 101: ages are from 0 to 100, where the latter includes all centenarians
        :data_path: data path
        :year_groups: multi-year groupings
        :pandemic_years: 
        :risk: 
        :INCLUDE_VAX: boolean to indicate whether vaccinations are simulated or not
        """

        # Class initialization parameters
        self.serogroup = serogroup # meningoccous serogroup
        self.mod_type = mod_type # model type
        self.A = A # maximum age (number of single-year age classes)
        self.data_path = data_path # data path
        self.year_groups = year_groups # multi-year groupings
        self.pandemic_years = pandemic_years # years with reduced contacts
        self.risk = risk
        self.INCLUDE_VAX = INCLUDE_VAX # whether vaccinations are included

        # Auxiliary parameters
        self.mod_name = f'{self.mod_type}_Men{self.serogroup}' # model name
        self.n_year_groups = len(self.year_groups)
        self.age_groups = ut.import_age_groups(self.data_path, self.serogroup) # age groups
        self.n_age_groups = len(self.age_groups)
        self.all_ages = [f'{age}' for age in range(self.A)]

        # ODE module variables
        self.ODE_module = self._load_ODE_module() # loads ODE module with equations
        for name, value in vars(self.ODE_module).items():
            if not name.startswith('_'):  # Skip private/internal attributes
                setattr(self, name, value)

        ####### GRAPHICS
        self._set_graphics()

        ####### IMPORT FIXED PARAMETERS
        self._import_params()

    def _load_ODE_module(self):
        """
        Load module containing system of ODEs, i.e. the function to be fed to scipy.solve_ivp, and the structural parameters for the model
        """

        try:
            package_name = 'EpiModels'
            module_name = f'{package_name}.{self.mod_name}_Model'
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            return module
        except ModuleNotFoundError:
            raise ValueError(f'ODE module for type {self.mod_name} not found.')

    def _set_graphics(self):
        if self.serogroup == 'B': base_colors = ['#5E1C3C', '#A12A2A', '#D45E1F', '#EFC637', '#4F7F1F']
        if self.serogroup == 'C': base_colors = ['#7A3E2F', '#EFC637', '#3288BD', '#5E4FA2']

        self.colors = base_colors + ['#%06X' % random.randint(0, 0xFFFFFF) for i in range(self.A - len(base_colors))]
        self.masked_plot_kwargs = {'marker': 'o', 'markersize': 5, 'linestyle': 'dashed'}
        self.non_masked_plot_kwargs = {'marker': 'o', 'markersize': 1, 'linewidth': 1}

    def get_carriage_data(self):
        df = pd.read_csv(os.path.join(self.data_path, 'Carriage/carriage_data.csv'))
        df = df[df['Serogroup'] == self.serogroup]

        self.carriage_years = list(df['Year'])
        self.n_carriage_years = len(self.carriage_years)
        self.carriage_age_groups = list(df['Age group'])
        self.n_carriage_age_groups = len(self.carriage_age_groups)
        # self.carriage_data = list(df['Prevalence'])
        # self.carriage_data_k = list(df['Carriers'])
        # self.carriage_data_n = list(df['Tested'])
        self.carriage_data = df['Prevalence'].values
        self.carriage_data_k = df['Carriers'].values
        self.carriage_data_n = df['Tested'].values

    def get_IMD_data(self):
        df = pd.read_csv(os.path.join(self.data_path, f'IMD/IMD_Men{self.serogroup}.csv'))
        self.data_years = np.sort(df['Year'].unique()) # all years for which we have IMD data
        self.data_start_year = np.min(self.data_years)
        self.data_end_year = np.max(self.data_years)

        df = ut.add_grouping(df, self.year_groups, 'Year')
        df_table = df.pivot_table(values='IMD cases', index=['Year', 'Year group'], columns='Age', sort=False)
        df_table = df_table.groupby('Year group').sum() # sum for poisson likelihood
        self.IMD_data = df_table.to_numpy()

        return

    def _import_params(self):
        """
        Import fixed parameters from Data folder.
        """

        ####### DEMOGRAPHIC PARAMETERS
        N = np.loadtxt(os.path.join(self.data_path, 'Population/equilibrium_pop.txt')) # Model population
        dem_year = 2017 # Year of birth rate and mortality

        # Yearly births
        b = np.zeros(self.A)
        df_bir = pd.read_csv(os.path.join(self.data_path, f'Births/births.csv'))
        b[0] = df_bir.loc[df_bir['Year'] == dem_year, 'Births'].values[0]

        # Mortality
        df_mort = pd.read_csv(os.path.join(self.data_path, f'Deaths/deaths.csv'))
        df_mort = df_mort[(df_mort['Year'] == dem_year) &  (df_mort['Age'] != 'Total')]
        m = np.array(df_mort['Mortality rate']) # mortality rate by age

        # Import contact matrix
        df_K = pd.read_csv(os.path.join(self.data_path, 'Contact_matrix/contact_matrix_age_groups.csv'))
        K_age_groups = list(df_K['Age'])
        K = np.loadtxt(os.path.join(self.data_path, 'Contact_matrix/contact_matrix.txt')) # define base contact matrix
        K = ut.expand_contact_matrix(K, N, K_age_groups) # expand to all A ages
        K = ut.symmetrize_contact_matrix(K, N) # symmetrize the matrix

        ####### IMPORT CARRIAGE AND IMD DATA FOR LIKELIHOOD AND VALIDATION
        self.get_carriage_data()
        self.get_IMD_data()

        ####### OTHER PARAMETERS
        alpha = 1 # proportion of carriers (I0) becoming temporarily immune to IMD (S1)
        beta = 0.07 # transmissibility
        gamma = 365 / 130 # average duration of carriage episode: gamma^-1
        delta = 12 / 36 # delta^-1 average time spend in the R compartment

        # Maximum age of Neisseria lactamica protection
        chi_value = 1 # 0.1 # protection conferred by lactamica
        lactamica_age = 10 # age until which there is protection
        chi = ut.define_chi_array(chi_value, lactamica_age, self.A)

        # Parameter that allows recovered individuals to reacquire carriage, even without being able to develop IMD: z=0 recovered can't regain carriage; z=1 they can.
        z = 0

        # Array of ages
        a = np.arange(self.A)

        # Risk of disease given carriage
        if self.risk == 'Trotter':
            r = np.loadtxt(os.path.join(self.data_path, f'Case_carrier/r_Trotter_Men{self.serogroup}.txt')) # Trotter estimate
        elif self.risk == 'constant':
            r = np.full((self.A,), 1e-2)

        # Risk rescaling for second stage infection
        rho = 0

        # school reductions
        xi_age_groups = ['6-10', '11-13', '14-18']

        if self.serogroup == 'B':
            # Import vaccine uptake rates: routine at age 0 (age 2 data), catch-up at age 12 (age 16 data)
            df = pd.read_csv(os.path.join(self.data_path, 'Coverage/uptake_rate.csv'))
            df = df[df['Serogroup'] == self.serogroup]
            aR, aC = sorted(df['Age'].unique()) # ages for routine and catch-up vaccinations
            uR_dict = df.loc[df['Age'] == aR, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            uC_dict = df.loc[df['Age'] == aC, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            # uC_dict = {k:v for k, v in uC_dict.items() if k >= 2014} # remove data from before 2014 # TODO: check this thing

            # VE against carriage
            epsR = 0 # routine
            epsC = 0 # catch-up

            # VE against disease: case:carrier ratio for vaccinated
            omegaR = 1 / 2
            omegaC = 1 / 8
            kappaR = ut.VE_discretized(a, aR, omegaR) # routine vaccine efficacy
            kappaC = ut.VE_discretized(a, aC, omegaC) # catch-up vaccine efficacy
            rR = (1 - kappaR) * r
            rC = (1 - kappaC) * r

            self.mod_params = {
                'uR_dict': uR_dict,
                'uC_dict': uC_dict,
                'epsR': epsR,
                'epsC': epsC,
                'aR': aR,
                'aC': aC,
                'omegaR': omegaR,
                'omegaC': omegaC,
                'kappaR': kappaR,
                'kappaC': kappaC,
                'rR': rR,
                'rC': rC,
            }

        if self.serogroup == 'C':
            # Import vaccine uptake rates: routine at age 0 (age 2 data), catch-up at age 12 (age 16 data)
            df_u = pd.read_csv(os.path.join(self.data_path, 'Coverage/uptake_rate.csv'))
            df_u = df_u[df_u['Serogroup'] == self.serogroup]
            aR1, aC1 = sorted(df_u[df_u['Vaccine'] == 'MenC']['Age'].unique()) # ages for routine and catch-up vaccinations
            aR4, aC4 = sorted(df_u[df_u['Vaccine'] == 'MenACWY']['Age'].unique())
            uR1_dict = df_u[df_u['Vaccine'] == 'MenC'].loc[df_u['Age'] == aR1, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            uC1_dict = df_u[df_u['Vaccine'] == 'MenC'].loc[df_u['Age'] == aC1, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            uR4_dict = df_u[df_u['Vaccine'] == 'MenACWY'].loc[df_u['Age'] == aR4, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            uC4_dict = df_u[df_u['Vaccine'] == 'MenACWY'].loc[df_u['Age'] == aC4, ['Year', 'Uptake rate']].set_index('Year')['Uptake rate'].to_dict()
            uC1_dict = {k:v for k, v in uC1_dict.items() if k >= 2014} # remove data from before 2014 # TODO: check this thing
            uR4_dict = {k:v for k, v in uR4_dict.items() if k >= 2014} # remove data from before 2014 # TODO: check this thing
            uC4_dict = {k:v for k, v in uC4_dict.items() if k >= 2014} # remove data from before 2014 # TODO: check this thing

            # VE against carriage
            etaR1 = 1 / 3 # rate of loss of protection against carriage for routine MenC vax # ASSUMED TO BE EQUAL TO CATCH-UP VACCINATION
            etaC1 = 1 / 3 # rate of loss of protection against carriage for catch-up MenC vax # from 2002Maiden, 2008Maiden, 2010Campbell

            epsR1 = ut.VE_discretized(a, aR1, etaR1) # MenC routine
            epsC1 = ut.VE_discretized(a, aC1, etaC1) # MenC catch-up
            epsR4 = 0 # MenACWY routine # TODO: assume to be equal to MCC; if I do this, I don't need two vaccines
            epsC4 = 0 # MenACWY catch-up # TODO: assume to be equal to MCC

            # VE against disease: case:carrier ratio for vaccinated
            omegaR1 = 1 / 4 # rate of loss of protection against disease for routine MenC vax # from 2001Richmond, 2010Campbell, 2019Nolan, 2022Neri (3-4 yrs); models: 2015Vickers (4 yrs) 
            omegaC1 = 1 / 10 # rate of loss of protection against disease for catch-up MenC vax # ASSUMED
            omegaR4 = 1 / 4 # rate of loss of protection against disease for routine MenACWY vax # from 2023Cutland; models: 2015Vickers (4 yrs), 2020Beck (4 yrs)
            omegaC4 = 1 / 10 # rate of loss of protection against disease for catch-up MenACWY vax # models: 2020Beck (15 yrs)
            kappaR1 = ut.VE_discretized(a, aR1, omegaR1) # routine vaccine efficacy
            kappaC1 = ut.VE_discretized(a, aC1, omegaC1) # catch-up vaccine efficacy
            kappaR4 = ut.VE_discretized(a, aR4, omegaR4)
            kappaC4 = ut.VE_discretized(a, aC4, omegaC4)
            rR1 = (1 - kappaR1) * r
            rC1 = (1 - kappaC1) * r
            rR4 = (1 - kappaR4) * r
            rC4 = (1 - kappaC4) * r

            self.mod_params = {
                'uR1_dict': uR1_dict,
                'uC1_dict': uC1_dict,
                'uR4_dict': uR4_dict,
                'uC4_dict': uC4_dict,
                'epsR1': epsR1,
                'epsC1': epsC1,
                'epsR4': epsR4,
                'epsC4': epsC4,
                'aR1': aR1,
                'aC1': aC1,
                'aR4': aR4,
                'aC4': aC4,
                # 'omegaR': omegaR, # TODO
                # 'omegaC': omegaC,
                # 'kappaR': kappaR,
                # 'kappaC': kappaC,
                'rR1': rR1,
                'rC1': rC1,
                'rR4': rR4,
                'rC4': rC4,
            }

        self.mod_params.update({
            'N': N,
            'b': b,
            'm': m,
            'K': K,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'lactamica_age': lactamica_age,
            'chi': chi,
            'z': z,
            'r': r,
            'rho': rho,
            'xi_age_groups': xi_age_groups
        })

################################## UPDATE PARAMETERS: MCMC STEP

    def update_params(self, param_dict):
        self.mod_params.update(param_dict)
        
        if 'gamma_duration' in param_dict:
            self.mod_params['gamma'] = 1 / self.mod_params['gamma_duration']
        if 'delta_duration' in param_dict:
            self.mod_params['delta'] = 1 / self.mod_params['delta_duration']
        if 'chi_value' in param_dict:
            self.mod_params['chi'] = ut.define_chi_array(param_dict['chi_value'], self.mod_params['lactamica_age'], self.A)
        if 'Trotter_params' in param_dict:
            a = np.arange(self.A)
            if self.serogroup == 'B':
                r = ut.CC_Trotter(a, *param_dict['Trotter_params'])
                rR = (1 - self.mod_params['kappaR']) * r
                rC = (1 - self.mod_params['kappaC']) * r
                self.mod_params.update({
                    'r': r,
                    'rR': rR,
                    'rC': rC,
                })

################################## SIMULATION

    def set_init_values(self, init_values=None):
        """
        Set initial values for the system of ODEs

        Parameters:
        :init_values: custom initial values
        """

        if init_values is None:
            default_init_values = np.zeros((self.n_comps, self.A))
            default_init_values[self.carrier_idx[0]] = 0.01 * self.mod_params['N'] # infected seed
            default_init_values[0] = self.mod_params['N'] - default_init_values[self.carrier_idx[0]] # Susceptible = N - infected seed
            default_init_values[0,0] = 1e-15 # set 0 age class to 0 (no births yet)
            return default_init_values
        else:
            if list(init_values.shape) != [self.n_comps, self.A]:
                raise ValueError(
                    f'Inconsistent initial values: expected init_values.shape = {[self.n_comps, self.A]} values, but got {init_values.shape}.'
                )
            else:
                return init_values

    def load_ODE_params(self, current_year):
        """
        Load parameters for ODE system, given the current year

        Parameters:
        :current_year: year of simulation
        """

        b = self.mod_params['b']
        m = self.mod_params['m']
        K = self.mod_params['K']
        beta = self.mod_params['beta']
        gamma = self.mod_params['gamma']
        chi = self.mod_params['chi']

        beta, K = self._reduce_pandemic_params(current_year, beta, K)

        if self.serogroup == 'B':
            # Define uptake rates for current_year
            if not self.INCLUDE_VAX: # if no vaccines are included in the model, set uptake to 0
                uR, uC = np.zeros(2)
            else:
                uR = ut.get_uptake_rate_for_year(self.A, self.mod_params['aR'], self.mod_params['uR_dict'], current_year)
                uC = ut.get_uptake_rate_for_year(self.A, self.mod_params['aC'], self.mod_params['uC_dict'], current_year)

            epsR, epsC = [self.mod_params[name] for name in ['epsR', 'epsC']]

            if self.mod_type == 'SCS':
                params = [b, m, K, beta, gamma, chi, uR, uC, epsR, epsC]

            elif self.mod_type == 'SCR' or self.mod_type == 'SCS_2stage':
                alpha = self.mod_params['alpha']
                delta = self.mod_params['delta']
                z = self.mod_params['z']
                params = [b, m, K, alpha, beta, gamma, delta, chi, z, uR, uC, epsR, epsC]

        if self.serogroup == 'C':
            # Define uptake rates for current_year
            if not self.INCLUDE_VAX:
                uR1, uC1, uR4, uC4 = np.zeros(4)
            else:
                uR1 = ut.get_uptake_rate_for_year(self.A, self.mod_params['aR1'], self.mod_params['uR1_dict'], current_year)
                uC1 = ut.get_uptake_rate_for_year(self.A, self.mod_params['aC1'], self.mod_params['uC1_dict'], current_year)
                uR4 = ut.get_uptake_rate_for_year(self.A, self.mod_params['aR4'], self.mod_params['uR4_dict'], current_year)
                uC4 = ut.get_uptake_rate_for_year(self.A, self.mod_params['aC4'], self.mod_params['uC4_dict'], current_year)

            epsR1, epsC1, epsR4, epsC4 = [self.mod_params[name] for name in ['epsR1', 'epsC1', 'epsR4', 'epsC4']]

            if self.mod_type == 'SCS':
                pass
            
            elif self.mod_type == 'SCR':
                pass

        return params

    def simulate_to_equilibrium(self, threshold_MSE, init_values=None, t_eval=None):
        """
        Run simulation to transmission dynamic equilibrium on model system of ODEs  

        Parameters:
        :threshold_MSE: threshold for mean square error at which simulaiton is deemed to have reached equilibrium
        :init_values: initial values for system of ODEs; if None, random default
        :t_eval: array of fixed points for solution of system of ODEs (t_eval of solve_ivp)
        """

        self.EQUILIBRIUM_FAIL = False
        self.init_values = self.set_init_values(init_values)

        t_span = np.array([0,1])
        t = []
        y = []
        y_current = self.init_values

        prev_sol = 0
        current_MSE = 1e6
        i = 0

        while current_MSE > threshold_MSE:
            if i >= 500:
                self.EQUILIBRIUM_FAIL = True
                break

            current_year = 0
            params = self.load_ODE_params(current_year)
            
            t_eval_i = t_eval if t_eval is None else t_eval + i
            sol = solve_ivp(
                fun=lambda t, y: self.ODE_module.ODE_system(t, y, *params, self.n_comps),
                t_span=t_span + i,
                t_eval=t_eval_i,
                y0=y_current.flatten(),
                method='RK45', # LSODA
                rtol=1e-2, # 1e-3
                atol=1e-4, # 1e-6
            )

            # Update solution
            if i == 0:
                t += sol.t.tolist() # concatenate time steps to previous time steps
                y.append(sol.y) # concatenate solutions to previous solutions
            else:
                t += sol.t[1:].tolist() # concatenate time steps to previous time steps: remove first value
                y.append(sol.y[:,1:]) # concatenate solutions to previous solutions: remove first value

            sim = sol.y.reshape(self.n_comps,self.A,-1)

            # Check if equilibrium has been reached
            current_sol = sim[:,:,-1]
            current_MSE = np.sqrt(np.sum((current_sol - prev_sol)**2))
            prev_sol = current_sol

            # Update initial values for next simulation
            init = np.zeros(self.n_comps)
            init[0] = 1e-15 # small number to avoid division by 0
            # y_current = np.insert(sim[:,:-1,-1], 0, init, axis=1) # insert new birth cohort at position 0 and remove oldest cohort
            y_current = np.concatenate([init[:, None], sim[:, :-1, -1]], axis=1)
            i += 1

        self.n_eq_years = i
        self.eq_years = np.arange(self.n_eq_years)
        self.eq_start_year = np.min(self.eq_years)
        self.eq_end_year = np.max(self.eq_years)

        self.eq_y_current = y_current # prepare for next simulation

        self.eq_t = np.array(t)
        self.eq_sol = np.hstack(y).reshape(self.n_comps,self.A,-1)
        self.eq_mask = np.isin(self.eq_t, self.eq_years + 1) # array to filter only time steps corresponding end of each year: 1901 is not included
        assert np.all(self.eq_years == self.eq_t[self.eq_mask] - 1)

        # Save final time step of equilibrium solution
        self.eq_sol_final_time = self.eq_sol[:,:,-1]

    def simulate_vaccinations(self, end_year=None, t_eval=None, fut_vax_rate=None):
        """
        Run simulation after having reached equilibrium by introducing vaccinations.

        Parameters:
        :vax_end_year: last year with vaccination data
        :t_eval: array of fixed points for solution of system of ODEs (t_eval of solve_ivp)
        :fut_vax_rate: R,C (len = 2) if serogroup B; R1, C1, R4, C4 (len = 4) if serogroup C
        """

        # Get first year with vaccinations
        if self.serogroup == 'B': vax_dict_names = ['uR_dict', 'uC_dict']
        if self.serogroup == 'C': vax_dict_names = ['uR1_dict', 'uC1_dict', 'uR4_dict', 'uC4_dict']

        # Get years with vaccination data
        vax_data_years = np.sort(np.unique([year for d in vax_dict_names for year in list(self.mod_params[d].keys())]))
        self.vax_start_year = np.min(vax_data_years) # first year with vax data
        self.vax_end_years = [np.max(list(self.mod_params[d].keys())) for d in vax_dict_names] # get final years for routine and catch-up vax
        self.first_vax_end_year = np.min(self.vax_end_years) # get first last year with vax data among different vax types
        self.end_year = np.max(self.vax_end_years) if end_year is None else end_year # end year of simulation

        if self.end_year < self.vax_start_year: # if simulation ends before vax starts
            # self.vax_start_year = self.end_year
            return

        # Add extra data if you want to simulate future too
        if self.end_year > self.first_vax_end_year: # if end year is after the first year where vax data ends, fill the remaining years with the future vax rate (fut_vax_rate)
            for i, vax_dict_name in enumerate(vax_dict_names):
                vax_dict = self.mod_params[vax_dict_name]
                last_vax_year = self.vax_end_years[i]
                last_vax_rate = vax_dict[last_vax_year] if fut_vax_rate is None else ut.get_uptake_from_coverage(fut_vax_rate[i])
                for year in range(self.first_vax_end_year + 1, self.end_year + 1):
                    try:
                        vax_dict[year]
                    except:
                        vax_dict[year] = last_vax_rate

        self.vax_years = np.arange(self.vax_start_year, self.end_year + 1)
        self.n_vax_years = len(self.vax_years)

        # Initiate arrays
        t_span = np.array([0,1])
        t = []
        y = []

        y_current = self.eq_y_current

        # Launch simulation
        for i in range(self.n_vax_years):
            current_year = self.vax_start_year + i
            params = self.load_ODE_params(current_year)

            t_eval_i = t_eval if t_eval is None else t_eval + i
            sol = solve_ivp(
                fun=lambda t, y: self.ODE_module.ODE_system(t, y, *params, self.n_comps),
                t_span=t_span + i,
                t_eval=t_eval_i,
                y0=y_current.flatten(),
                method='RK45', # LSODA
                rtol=1e-2, # 1e-3
                atol=1e-4, # 1e-6
            )

            # Update solution
            if i == 0:
                t += sol.t.tolist() # concatenate time steps to previous time steps
                y.append(sol.y) # concatenate solutions to previous solutions
            else:
                t += sol.t[1:].tolist() # concatenate time steps to previous time steps: remove first value
                y.append(sol.y[:,1:]) # concatenate solutions to previous solutions: remove first value

            sim = sol.y.reshape(self.n_comps,self.A,-1)

            init = np.zeros(self.n_comps)
            init[0] = 1e-15 # small number to avoid division by 0
            # y_current = np.insert(sim[:,:-1,-1], 0, init, axis=1) # insert new birth cohort at position 0 and remove oldest cohort
            y_current = np.concatenate([init[:, None], sim[:, :-1, -1]], axis=1)

        self.vax_t = np.array(t) + self.vax_start_year
        self.vax_sol = np.hstack(y).reshape(self.n_comps,self.A,-1)
        self.vax_mask = np.isin(self.vax_t, self.vax_years + 1) # array to filter only time steps corresponding end of each year: 1901 is not included
        assert np.all(self.vax_years == self.vax_t[self.vax_mask] - 1)

    def simulate(self, threshold_MSE, init_values=None, end_year=None, t_eval=None, fut_vax_rate=None):
        """
        Simulate model

        Parameters:
        :threshold_MSE: threshold for mean square error at which simulation is deemed to have reached equilibrium
        :init_values: initial values for system of ODEs; if None, random default
        :end_year: last year of simulation
        :t_eval: array of fixed points for solution of system of ODEs (t_eval of solve_ivp)
        :fut_vax_rate: R,C (len = 2) if serogroup B; R1, C1, R4, C4 (len = 4) if serogroup C
        """

        self.simulate_to_equilibrium(threshold_MSE=threshold_MSE, init_values=init_values, t_eval=t_eval)
        self.simulate_vaccinations(end_year=end_year, t_eval=t_eval, fut_vax_rate=fut_vax_rate)
        self.idx = ut.get_year_group_idx_from_year(self.end_year, self.year_groups) # if end_year is greater than last data year, keep whole array

        # NEW part
        if self.end_year < self.vax_start_year:
            self.eq_years += self.end_year - self.n_eq_years + 1 # years until equilibrium is reached
            self.eq_t += self.end_year - self.n_eq_years + 1
            self.years = self.eq_years # all simulation years
            self.t = self.eq_t # all simulation times
            self.sol = self.eq_sol # total simulation solution
            self.mask = self.eq_mask

        else:
            self.eq_years += self.vax_start_year - self.n_eq_years # years until equilibrium is reached
            self.eq_t += self.vax_start_year - self.n_eq_years
            self.years = np.concatenate((self.eq_years, self.vax_years)) # all simulation years
            self.t = np.concatenate((self.eq_t, self.vax_t[1:])) # all simulation times
            self.sol = np.concatenate((self.eq_sol, self.vax_sol[:,:,1:]), axis=2) # total simulation solution
            self.mask = np.concatenate((self.eq_mask, self.vax_mask[1:]))

        # Adjust years
        self.eq_start_year = np.min(self.eq_years)
        self.eq_end_year = np.max(self.eq_years)
        self.n_years = len(self.years)
        self.start_year = np.min(self.years)

        assert np.all(self.eq_years == self.eq_t[self.eq_mask] - 1)
        assert np.all(self.years == self.t[self.mask] - 1)

        # Define auxiliary variables
        self.FOI = self._calculate_FOI() # calculate just once at end of simulation: FOI for every simulated time step

        # Get model carriage and IMD estimates in the same format as the data at the end of simulation run
        self.mod_carriage = self.get_mod_carriage()
        self.mod_IMD = self.get_mod_IMD()

        # Compute likelihood
        self.likelihood = self.compute_likelihood()

    def simulate_pandemic_years(self, new_end_year, prev_end_values, new_start_year=None, t_eval=None, fut_vax_rate=None):
        """
        Resume simulation from 2020, after having simulated the model until the end of 2019. Method needed to avoid simulating equilibrium every time, especially when testing different values for pandemic reduction factors.  

        Parameters:
        :new_end_year: 
        :prev_end_values: end_values of simulation: must be shifted and empty 0-age cohort must be added
        :new_start_year:
        :t_eval: array of fixed points for solution of system of ODEs (t_eval of solve_ivp)
        :fut_vax_rate: coverage; R,C (len = 2) if serogroup B; R1, C1, R4, C4 (len = 4) if serogroup C
        """

        # Get years with vaccination data
        self.new_start_year = self.end_year + 1 if new_start_year is None else new_start_year # start year of new stage of simulation must be same as end year of previous stage
        if self.new_start_year < self.vax_start_year: # if simulation ends before vax starts
            raise ValueError(f'New simulation must start after (including) the first year of vaccinations: {self.vax_start_year}')
        self.new_end_year = np.max(self.vax_end_years) if new_end_year is None else new_end_year # end year of new stage of simulation

        # Get vaccination dicts
        if self.serogroup == 'B': vax_dict_names = ['uR_dict', 'uC_dict']
        if self.serogroup == 'C': vax_dict_names = ['uR1_dict', 'uC1_dict', 'uR4_dict', 'uC4_dict']

        # Add extra data if you want to simulate future too
        if self.new_end_year > self.first_vax_end_year: # if new end year is after the first year where vax data ends, fill the remaining years with the future vax rate (fut_vax_rate)
            for i, vax_dict_name in enumerate(vax_dict_names):
                vax_dict = self.mod_params[vax_dict_name]
                last_vax_year = self.vax_end_years[i]
                last_vax_rate = vax_dict[last_vax_year] if fut_vax_rate is None else ut.get_uptake_from_coverage(fut_vax_rate[i])
                for year in range(self.first_vax_end_year + 1, self.new_end_year + 1):
                    try:
                        vax_dict[year]
                    except:
                        vax_dict[year] = last_vax_rate

        self.new_years = np.arange(self.new_start_year, self.new_end_year + 1)
        self.n_new_years = len(self.new_years)

        # Initiate arrays
        t_span = np.array([0,1])
        t = []
        y = []

        init = np.zeros(self.n_comps)
        init[0] = 1e-15 # small number to avoid division by 0
        y_current = np.concatenate([init[:, None], prev_end_values[:, :-1, -1]], axis=1)

        # Launch simulation
        for i in range(self.n_new_years):
            current_year = self.new_start_year + i
            params = self.load_ODE_params(current_year)

            t_eval_i = t_eval if t_eval is None else t_eval + i
            sol = solve_ivp(
                fun=lambda t, y: self.ODE_module.ODE_system(t, y, *params, self.n_comps),
                t_span=t_span + i,
                t_eval=t_eval_i,
                y0=y_current.flatten(),
                method='RK45', # LSODA
                rtol=1e-2, # 1e-3
                atol=1e-4, # 1e-6
            )

            # Update solution
            if i == 0:
                t += sol.t.tolist() # concatenate time steps to previous time steps
                y.append(sol.y) # concatenate solutions to previous solutions
            else:
                t += sol.t[1:].tolist() # concatenate time steps to previous time steps: remove first value
                y.append(sol.y[:,1:]) # concatenate solutions to previous solutions: remove first value

            sim = sol.y.reshape(self.n_comps,self.A,-1)

            init = np.zeros(self.n_comps)
            init[0] = 1e-15 # small number to avoid division by 0
            y_current = np.concatenate([init[:, None], sim[:, :-1, -1]], axis=1)

        # print(self.new_years)
        self.new_t = np.array(t) + self.new_start_year
        # print(self.new_t)
        self.new_sol = np.hstack(y).reshape(self.n_comps,self.A,-1)
        # print(self.new_sol)
        self.new_mask = np.isin(self.new_t, self.new_years + 1) # array to filter only time steps corresponding end of each year
        assert np.all(self.new_years == self.new_t[self.new_mask] - 1)

        # Define auxiliary variables
        self.idx = ut.get_year_group_idx_from_year(self.new_end_year, self.year_groups) # if end_year is greater than last data year, keep whole array
        self.FOI = self._calculate_FOI(new=True) # calculate just once at end of simulation: FOI for every simulated time step
        self.new_mod_IMD = self.increase_mod_IMD()

################################## GET POPULATIONS

# Internal methods

    def _calculate_FOI(self, new=False):
        """
        Calculate force of infection for every time step. No masking here!
        """

        I = self.get_compartments('carrier', masked=False, grouping='all_ages', split_comps=False, new=new)
        pop = self.get_compartments('population', masked=False, grouping='all_ages', split_comps=False, new=new)

        t = self.new_t if new else self.t
        FOI = np.zeros((self.A, len(t)))

        end_year = self.new_end_year if new else self.end_year
        if end_year < np.min(self.pandemic_years):
            beta = self.mod_params['beta']
            K = self.mod_params['K']
            chi = self.mod_params['chi']
            FOI = beta * chi.reshape(-1,1) * (K @ (I / pop))

        else:
            for i, year in enumerate(range(np.min(self.pandemic_years) - 1, end_year + 1)): # from last pre-pandemic year (2019)
                beta = self.mod_params['beta']
                K = self.mod_params['K']
                chi = self.mod_params['chi']
                beta, K = self._reduce_pandemic_params(year, beta, K)
                
                if i == 0: # if last pre-pandemic year (2019)
                    t_mask = t <= year + 1 # less equal than 2020 (end of 2019)
                else:
                    t_mask = (t > year) & (t <= year + 1) # in (2020, 2021], (2021, 2022], ...

                FOI[:,t_mask] = beta * chi.reshape(-1,1) * (K @ (I[:,t_mask] / pop[:,t_mask]))

        return FOI

# External methods

    def get_compartments(self, comps, masked=True, grouping=None, custom_age_groups=None, split_comps=True, new=False):
        """
        Get population of selected compartments.

        Parameters:
        :comps:
        :grouping: None, 'age_groups', 'all_ages' or 'custom'
        :masked: if False, all timesteps; if True, n_years
        :custom_age_groups:
        :split_comps:
        :new: True or False (if method simulate from vax year is called) 
        """

        if isinstance(comps, list): comp_idx = comps
        elif isinstance(comps, str):
            if comps == 'susceptible': comp_idx = self.susceptible_idx
            elif comps == 'carrier': comp_idx = self.carrier_idx
            elif comps == 'recovered': comp_idx = self.recovered_idx
            elif comps == 'noncarrier': comp_idx = self.noncarrier_idx
            elif comps == 'vaccinated': comp_idx = self.vaccinated_idx
            elif comps in ['population', 'all']:
                comp_idx = np.arange(self.n_comps) # if I want all compartments or total population, I select all compartments in both cases
                if comps == 'population': split_comps = False # if I want the population, I will always sum over all compartments, so no split compartments
            else: raise ValueError(f'Unknown comps string: {comps}')
        else: raise TypeError('comps must be a list of indices or a string.')

        sol = self.new_sol if new else self.sol
        mask = self.new_mask if new else self.mask
        X = sol[comp_idx]
        X = X[:,:,mask] if masked else X
        age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
        X = X.sum(axis=1) if grouping is None else ut.aggregate_sol(X, age_groups)
        if not split_comps: X = X.sum(axis=0)
        return X

    def get_new_carriers(self, comps, masked=True, grouping=None, custom_age_groups=None, split_comps=True, new=False):
        """
        Get population of new carriers. 

        Parameters:
        :age_groups: if None, population is for all timesteps; otherwise specify age_groups or all_ages
        :masked: if False, all timesteps; if True, n_years
        """

        # Check
        if comps not in ['susceptible', 'recovered', 'both']:
            raise ValueError("'comps' must be 'susceptible' or 'recovered'")

        FOI = self.FOI

        if self.serogroup == 'B':
            if self.mod_type == 'SCR':
                S, SVR, SVC = self.get_compartments(comps='susceptible', masked=False, grouping='all_ages', split_comps=True, new=new)
                R, RVR, RVC = self.get_compartments(comps='recovered', masked=False, grouping='all_ages', split_comps=True, new=new)
                epsR = self.mod_params['epsR']
                epsC = self.mod_params['epsC']
                z = self.mod_params['z']
                
                if comps in ['susceptible', 'both']:
                    dB = FOI * S
                    dBVR = FOI * (1 - epsR) * SVR
                    dBVC = FOI * (1 - epsC) * SVC
                else: dB, dBVR, dBVC = [0, 0, 0]

                if comps in ['recovered', 'both']:
                    dB += FOI * z * R
                    dBVR += FOI * (1 - epsR) * z * RVR
                    dBVC += FOI * (1 - epsC) * z * RVC

                dI = np.array([dB, dBVR, dBVC])

            elif self.mod_type == 'SCS_2stage':
                S0, SVR0, SVC0, S1, SVR1, SVC1 = self.get_compartments(comps='susceptible', masked=False, grouping='all_ages', split_comps=True, new=new)
                epsR = self.mod_params['epsR']
                epsC = self.mod_params['epsC']
                z = self.mod_params['z']
                
                dI0 = FOI * S0
                dIVR0 = FOI * (1 - epsR) * SVR0
                dIVC0 = FOI * (1 - epsC) * SVC0
                
                dI1 = z * FOI * S1
                dIVR1 = z * FOI * (1 - epsR) * SVR1
                dIVC1 = z * FOI * (1 - epsC) * SVC1
                
                dI = np.array([dI0, dIVR0, dIVC0, dI1, dIVR1, dIVC1])

        # if self.mod_type == 'SCR_MenC':

        mask = self.new_mask if new else self.mask
        dI = dI[:,:,mask] if masked else dI
        age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
        dI = dI.sum(axis=1) if grouping is None else ut.aggregate_sol(dI, age_groups)
        if not split_comps: dI = dI.sum(axis=0)

        return dI

    def get_new_IMD(self, masked=True, grouping=None, custom_age_groups=None, split_comps=True, new=False):
        """
        Get population of new IMD. 

        Parameters:
        :age_groups: if None, population is for all timesteps; otherwise specify age_groups or all_ages
        :masked: if False, all timesteps; if True, n_years
        """

        if self.serogroup == 'B':
            if self.mod_type == 'SCR':
                dB_S, dBVR_S, dBVC_S = self.get_new_carriers(comps='susceptible', masked=False, grouping='all_ages', custom_age_groups=None, split_comps=True, new=new) # new carriers from S compartment, all times and all ages
                r = self.mod_params['r']
                rR = self.mod_params['rR']
                rC = self.mod_params['rC']

                dIMDS = r.reshape(-1,1) * dB_S
                dIMDR = rR.reshape(-1,1) * dBVR_S
                dIMDC = rC.reshape(-1,1) * dBVC_S

                dIMD = np.array([dIMDS, dIMDR, dIMDC])

            elif self.mod_type == 'SCS_2stage':
                dI0, dIVR0, dIVC0, dI1, dIVR1, dIVC1 = self.get_new_carriers(comps='susceptible', masked=False, grouping='all_ages', custom_age_groups=None, split_comps=True, new=new) # new carriers from S compartment, all times and all ages
                r = self.mod_params['r']
                rR = self.mod_params['rR']
                rC = self.mod_params['rC']
                rho = self.mod_params['rho']

                dIMD0   = r.reshape(-1,1) * dI0
                dIMDR0  = rR.reshape(-1,1) * dIVR0
                dIMDC0  = rC.reshape(-1,1) * dIVC0

                dIMD1   = r.reshape(-1,1) * rho * dI1
                dIMDR1  = rR.reshape(-1,1) * rho * dIVR1
                dIMDC1  = rC.reshape(-1,1) * rho * dIVC1

                dIMD = np.array([dIMD0, dIMDR0, dIMDC0, dIMD1, dIMDR1, dIMDC1])

        # if self.serogroup == 'C':

        mask = self.new_mask if new else self.mask
        dIMD = dIMD[:,:,mask] if masked else dIMD
        age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
        dIMD = dIMD.sum(axis=1) if grouping is None else ut.aggregate_sol(dIMD, age_groups)
        if not split_comps: dIMD = dIMD.sum(axis=0)

        return dIMD

################################## IMD data for likelihood, likelihood, posterior 

# External methods

    def get_pop_table(self, grouping=None, custom_age_groups=None, year_grouping=None, custom_year_groups=None, end_year=None):
        end_year = self.end_year if end_year is None else end_year
        df = pd.read_csv(os.path.join(self.data_path, 'Population/population.csv'))
        df = df[(df['Region'] == 'Italia') & (df['Age'] != 'Total') & (df['Year'] <= end_year)]
        age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
        if grouping is None:
            df = df.groupby(['Region', 'Year'])['Pop'].sum().reset_index()
            df['Age'] = '0-100'
        else:
            df = ut.aggregate_pop(df, age_groups).reset_index(drop=True)
        year_groups = self._get_year_groups_from_grouping_keyword(year_grouping) if year_grouping != 'custom' else custom_year_groups
        df = ut.add_grouping(df, year_groups, 'Year')
        df_table = df.pivot_table(values='Pop', index=['Year', 'Year group'], columns='Age', sort=False)

        return df_table

    def get_mod_carriage(self):
        if self.end_year < np.min(self.carriage_years):
            return

        mod_carriage = np.zeros(len(self.carriage_years))
        for i, year in enumerate(self.carriage_years):
            carriage_age_groups = [self.carriage_age_groups[i]]
            mask = self.years == int(year)
            carriers = self.get_compartments(comps='carrier', grouping='custom', custom_age_groups=carriage_age_groups, split_comps=False)[:,mask][0][0]
            pop = self.get_compartments(comps='population', grouping='custom', custom_age_groups=carriage_age_groups)[:,][0][0]
            mod_carriage[i] = carriers / pop
        return mod_carriage

    def get_mod_IMD(self, new=False):
        end_year = self.new_end_year if new else self.end_year
        if end_year < self.data_start_year:
            return

        years = self.new_years if new else self.years
        mask = (years >= self.data_start_year) & (years <= np.min([self.data_end_year, end_year]))

        # Get real populations
        df_table = self.get_pop_table(grouping='age_groups', year_grouping='year_groups', end_year=end_year)

        # Get simulation populations
        mod_pop_tot = self.get_compartments(comps='population', grouping='age_groups', new=new)[:,mask]
        df_table_sim = df_table.copy()
        df_table_sim.loc[:, df_table.columns] = mod_pop_tot.T

        # Get model IMD cases
        mod_IMD_cases = self.get_new_IMD(grouping='age_groups', split_comps=False, new=new)[:,mask]
        df_table_IMD = df_table.copy()
        df_table_IMD.loc[:, df_table.columns] = mod_IMD_cases.T

        # Aggregate populations
        df_table = df_table.groupby('Year group').sum()
        df_table_sim = df_table_sim.groupby('Year group').sum()
        df_table_IMD = df_table_IMD.groupby('Year group').sum()

        # Get arrays
        pop_array = df_table.to_numpy()
        pop_sim_array = df_table_sim.to_numpy()
        mod_IMD_array = df_table_IMD.to_numpy()

        # Get model IMD cases rescaled on the real Italian population, year-by-year
        mod_IMD_array *= pop_array / pop_sim_array

        return mod_IMD_array

    def increase_mod_IMD(self, new=True):
        end_year = self.new_end_year

        # Get real populations
        df_table = self.get_pop_table(grouping='age_groups', year_grouping='year_groups', end_year=end_year)
        idx = pd.IndexSlice
        df_table = df_table.loc[idx[self.end_year+1:self.new_end_year, :], :]

        # Get simulation populations
        mod_pop_tot = self.get_compartments(comps='population', grouping='age_groups', new=new)[:,0]
        df_table_sim = df_table.copy()
        df_table_sim.loc[:, df_table.columns] = mod_pop_tot.T

        # Get model IMD cases
        mod_IMD_cases = self.get_new_IMD(grouping='age_groups', split_comps=False, new=new)
        df_table_IMD = df_table.copy()
        df_table_IMD.loc[:, df_table.columns] = mod_IMD_cases.T

        # Aggregate populations
        df_table = df_table.groupby('Year group').sum()
        df_table_sim = df_table_sim.groupby('Year group').sum()
        df_table_IMD = df_table_IMD.groupby('Year group').sum()

        # Get arrays
        pop_array = df_table.to_numpy()
        pop_sim_array = df_table_sim.to_numpy()
        mod_IMD_array = df_table_IMD.to_numpy()

        # Get model IMD cases rescaled on the real Italian population, year-by-year
        mod_IMD_array *= pop_array / pop_sim_array
        mod_IMD_array = np.vstack((self.mod_IMD, mod_IMD_array))

        return mod_IMD_array

    def compute_IMD_likelihood(self):
        IMD_data = self.IMD_data[:self.idx+1] # restrict IMD data array
        IMD_likelihood = sp.stats.poisson.logpmf(IMD_data, mu=self.mod_IMD)
        return IMD_likelihood

    def compute_likelihood(self):
        # TODO: add carriage likelihood?
        if self.end_year < self.data_start_year:
            return
        log_likelihood = self.compute_IMD_likelihood().sum()
        return log_likelihood

################################## PLOTS

    def plot_compartments(self, comps, grouping=None, masked=True, perc=True, custom_age_groups=None, new=False, xlim=None, xtick_spacing=5, ylim=None, ytick_spacing=None, figsize=(10,4)):
        if new:
            t = self.new_years if masked else self.new_t
        else:
            t = self.years if masked else self.t
        pop_tot = self.get_compartments(comps='population', masked=masked, new=new) # total population
        X_tot = self.get_compartments(comps=comps, masked=masked, split_comps=False, new=new)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_kwargs = self.masked_plot_kwargs if masked else self.non_masked_plot_kwargs

        if grouping is not None:
            pop = self.get_compartments(comps='population', grouping=grouping, masked=masked, custom_age_groups=custom_age_groups, new=new)
            X = self.get_compartments(comps=comps, grouping=grouping, masked=masked, custom_age_groups=custom_age_groups, split_comps=False, new=new)
            age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
            X_age_group = X / pop * 100 if perc else X
            for i, age_group in enumerate(age_groups):
                ax.plot(t, X_age_group[i], label=f'{age_group}', color=self.colors[i], **plot_kwargs)
        X_all = X_tot / pop_tot * 100 if perc else X_tot
        ax.plot(t, X_all, label='Total', color='black', **plot_kwargs)

        # Aesthetics
        title = f'{comps}' if isinstance(comps, str) else f'Compartments {comps}'
        ax.set_title(title)
        
        # x axis
        start_year, end_year = [self.new_start_year, self.new_end_year] if new else [self.start_year, self.end_year]
        xticks = np.arange(start_year, end_year + 2, xtick_spacing)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=90, fontsize=6)
        ax.set_xlabel('Year')
        if xlim is not None: ax.set_xlim(xlim)
        
        # y axis
        ylabel = r'Percentage (\%)' if perc else 'Population'
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(ymin=0)
        
        ax.grid()
        fig.legend(title='Age', loc='outside right')
        plt.show()

    def plot_incidence(self, type, comps=None, grouping='age_groups', masked=True, split_comps=False, perc=True, custom_age_groups=None, xlim=None, xtick_spacing=5, ylim=None, figsize=(15,5)):
        """
        :type: 'IMD', 'carriage'; no default
        :comps: if type='carriage', comps='susceptible' or 'recovered'; otherwise, None
        """

        # Check function parameters
        if type == 'carriage' and comps is None:
            raise ValueError("For type='carriage', 'comps' must be specified.")
        elif type == 'IMD' and comps is not None:
            raise ValueError("For type='IMD', 'comps' will be ignored, so please leave default (None).")

        # Get total incidence
        t = self.years if masked else self.t
        pop_tot = self.get_compartments(comps='population', masked=masked) # total population
        if type == 'carriage': X_tot = self.get_new_carriers(comps=comps, masked=masked, split_comps=split_comps)
        elif type == 'IMD': X_tot = self.get_new_IMD(masked=masked, split_comps=split_comps)
        if not split_comps: X_tot = X_tot[np.newaxis, ...]

        # Get incidence by grouping
        if grouping is not None:
            pop = self.get_compartments('population', masked, grouping, custom_age_groups)
            if type == 'carriage': X = self.get_new_carriers(comps, masked, grouping, custom_age_groups, split_comps)
            elif type == 'IMD': X = self.get_new_IMD(masked, grouping, custom_age_groups, split_comps)
            if not split_comps: X = X[np.newaxis, ...]

        # Normalize
        if perc:
            X_tot /= pop_tot
            if type == 'IMD': X_tot *= 100000 # IMD incidence per 100'000 individuals
            if type == 'carriage': X_tot *= 100 # carriage incidence as a percentage of population (per 100 individuals)
            if grouping is not None:
                X /= pop
                if type == 'IMD': X *= 100000 # IMD incidence per 100'000 individuals
                if type == 'carriage': X *= 100 # carriage incidence as a percentage of population (per 100 individuals)

        # Define figure and axes
        n_comps = X_tot.shape[0] # column subplots
        fig, axs = plt.subplots(1, n_comps, figsize=figsize, constrained_layout=True)
        axs = np.atleast_1d(axs)
        fig_title = f'New {type} cases'
        fig.suptitle(fig_title)
        idx = self.susceptible_idx if (comps == 'susceptible' or type == 'IMD') else self.recovered_idx
        comp_names = [self.comps[i] for i in idx]
        subplot_titles = [f'From {comp} compartment' for comp in comp_names] if split_comps else [f'From all {comp_names[0]} compartments']
        plot_kwargs = self.masked_plot_kwargs if masked else self.non_masked_plot_kwargs

        # Plot
        for i in range(n_comps):
            ax = axs[i]
            ax.set_title(subplot_titles[i])

            # Plot total
            ax.plot(t, X_tot[i], label='Total', color='black', **plot_kwargs)

            # Plot age groups
            age_groups = self._get_age_groups_from_grouping_keyword(grouping) if grouping != 'custom' else custom_age_groups
            for j, age_group in enumerate(age_groups):
                ax.plot(t, X[i,j], label=f'{age_group}', color=self.colors[j], **plot_kwargs)

        # Aesthetics
        xticks = np.arange(self.start_year, self.end_year + 2, xtick_spacing)
        if type == 'carriage': ylabel = r'Percentage (\%)' if perc else 'Population'
        if type == 'IMD': ylabel = 'Incidence (per 100000)' if perc else 'Population'
        for ax in axs:
            # x axis
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation=90, fontsize=6)
            ax.set_xlabel('Year')
            if xlim is not None: ax.set_xlim(xlim)
            
            # y axis
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(ymin=0)

            ax.grid()

        ut.create_single_legend(fig, axs, title='Age', loc='outside right')
        plt.show()


################################## Extra

# Internal methods

    def _get_age_groups_from_grouping_keyword(self, grouping):
        age_groups = None
        if grouping == 'age_groups': age_groups = self.age_groups
        elif grouping == 'all_ages': age_groups = self.all_ages
        return age_groups
    
    def _get_year_groups_from_grouping_keyword(self, year_grouping):
        year_groups = None
        if year_grouping == 'year_groups': year_groups = self.year_groups
        elif year_grouping == 'all_years': year_groups = [f'{year}' for year in self.years]
        return year_groups

    def _reduce_pandemic_params(self, year, beta, K):
        zeta_name, xi_name = [f'zeta{year}', f'xi{year}'] if year in self.pandemic_years else [None, None]

        try:
            zeta = self.mod_params[zeta_name]
            beta *= zeta
        except:
            pass

        try:
            xi = self.mod_params[xi_name]
            K = ut.reduce_school_contacts(K, xi, self.mod_params['xi_age_groups'])
        except:
            pass

        return beta, K
    