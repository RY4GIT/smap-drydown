import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger
import threading
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
from scipy.stats import t

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""

# Create a logger
log = getLogger(__name__)


def tau_exp_model(t, delta_theta, theta_w, tau):
    """
    Calculate the drydown curve for soil moisture over time using linear loss function model.
    Analytical solution of the linear loss function is exponential function, with the time decaying factor tau

    Parameters:
        t (int): Timestep, in day.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
        tau (float): decay rate, in 1/day.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.

    Reference:
        McColl, K.A., W. Wang, B. Peng, R. Akbar, D.J. Short Gianotti, et al. 2017.
        Global characterization of surface soil moisture drydowns.
        Geophys. Res. Lett. 44(8): 3682–3690. doi: 10.1002/2017GL072819.
    """
    return delta_theta * np.exp(-t / tau) + theta_w


def exp_model(t, ETmax, theta_0, theta_star, theta_w, z=50.0, t_star=0.0):
    """Calculate the drydown curve for soil moisture over time using linear loss function model.
    The above tau_exp_model can be better constrained using the loss function variables, rather than tau models.

    Parameters:
        t (int): Timestep, in day.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
        tau (float): decay rate, in 1/day.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.

    """

    tau = z * (theta_star - theta_w) / ETmax

    if theta_0 > theta_star:
        theta_0_ii = theta_star
    else:
        theta_0_ii = theta_0

    return (theta_0_ii - theta_w) * np.exp(-(t - t_star) / tau) + theta_w


def q_model(t, q, ETmax, theta_0, theta_star, theta_w, z=50.0, t_star=0.0):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
        t (int): Timestep, in day.
        z (float): Soil thicness in mm. Default is 50 mm
        ETmax (float): Maximum evapotranpisration rate in mm/day.
        q (float): Degree of non-linearity in the soil moisture response.
        theta_0 (float): The initial soil moisture after precipitation, in m3/m3
        theta_star (float, optional): Critical soil moisture content, equal to s_star * porosity, in m3/m3
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.
    """
    if theta_0 > theta_star:
        theta_0_ii = theta_star
    else:
        theta_0_ii = theta_0

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    b = (theta_0_ii - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * (t - t_star) + b) ** (1 / (1 - q)) + theta_w


def q_model_piecewise(t, q, ETmax, theta_0, theta_star, theta_w, z=50.0):

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    t_star = (theta_0 - theta_star) / k  # Time it takes from theta_0 to theta_star

    return np.where(
        t_star > t,
        -k * t + theta_0,
        q_model(
            t, q, ETmax, theta_0, theta_star, theta_w, t_star=np.maximum(t_star, 0)
        ),
    )


def exp_model_piecewise(t, ETmax, theta_0, theta_star, theta_w, z=50.0):
    k = ETmax / z
    t_star = (theta_0 - theta_star) / k
    return np.where(
        t_star > t,
        -k * t + theta_0,
        exp_model(t, ETmax, theta_0, theta_star, theta_w, t_star=np.maximum(t_star, 0)),
    )


def drydown_piecewise(t, model, ETmax, theta_0, theta_star, z=50.0):
    """ "
    Calculate the drydown assuming that both Stage I and II are happening. Estimate theta_star
    """

    k = (
        ETmax / z
    )  # Constant term. Convert ETmax to maximum dtheta/dt rate from a unit volume of soil

    t_star = (
        theta_0 - theta_star
    ) / k  # Time it takes from theta_0 to theta_star (Stage II ET)

    return np.where(t_star > t, -k * t + theta_0, model)


def loss_sigmoid(t, theta, theta50, k, a):
    """
    Calculate the loss function (dtheta/dt vs theta relationship) using sigmoid model

    Parameters:
    t (int): Timestep, in day.
    theta (float): Volumetric soil moisture content, in m3/m3.
    theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
    k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
    a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]

    Returns:
    float: Rate of change in soil moisture (dtheta/dt) for the given soil mositure content, in m3/m3/day.
    """
    exp_arg = np.clip(
        -k * (theta - theta50), -np.inf, 10000
    )  # Clip exponent item to avoid failure
    d_theta = -1 * a / (1 + np.exp(exp_arg))
    return d_theta


# Function to solve the DE with given parameters and return y at the time points
def solve_de(t_obs, y_init, parameters):
    """
    The sigmoid loss function is a differential equation of dy/dt = f(y, a, b), which cannot be analytically solved,
    so the fitting of this model to drydown is numerically impelmented.
    solve_ivp finds y(t) approximately satisfying the differential equations, given an initial value y(t0)=y0.

    Parameters:
    t_obs (int): Timestep, in day.
    y_init (float): Observed volumetric soil moisture content, in m3/m3.
    parameters: a list of the follwing parameters
        theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
        k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
        a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]
    """
    theta50, k, a = parameters
    sol = solve_ivp(
        lambda t, theta: loss_sigmoid(t, theta, theta50, k, a),
        [t_obs[0], t_obs[-1]],
        [y_init],
        t_eval=t_obs,
        vectorized=True,
    )
    return sol.y.ravel()


# The objective function to minimize (sum of squared errors)
def objective_function(parameters, y_obs, y_init, t_obs):
    y_model = solve_de(t_obs, y_init, parameters)
    error = y_obs - y_model
    return np.sum(error**2)


class DrydownModel:
    def __init__(self, cfg, Data, Events):

        # ______________________________________________________________________
        # Read input
        self.cfg = cfg
        self.data = Data
        self.events = Events

        # ______________________________________________________________________
        # Read Model config
        self.plot_results = cfg.getboolean("MODEL", "plot_results")
        self.force_PET = cfg.getboolean("MODEL", "force_PET")
        self.run_tau_exp_model = cfg.getboolean("MODEL", "tau_exp_model")
        self.run_exp_model = cfg.getboolean("MODEL", "exp_model")
        self.run_q_model = cfg.getboolean("MODEL", "q_model")
        self.run_sigmoid_model = cfg.getboolean("MODEL", "sigmoid_model")
        self.is_stage1ET_active = cfg.getboolean("MODEL", "is_stage1ET_active")

        # Model parameters
        self.z = self.cfg.getfloat("MODEL_PARAMS", "z")
        self.target_rmsd = self.cfg.getfloat("MODEL_PARAMS", "target_rmsd")

        # ______________________________________________________________________
        # Set normalization factor
        self.norm_max = self.data.max_cutoff_sm
        self.norm_min = self.data.min_sm

        # ______________________________________________________________________
        # Get threadname
        if cfg.get("MODEL", "run_mode") == "parallel":
            current_thread = threading.current_thread()
            current_thread.name = (
                f"[{self.data.EASE_row_index},{self.data.EASE_column_index}]"
            )
            self.thread_name = current_thread.name
        else:
            self.thread_name = "main thread"

    def fit_models(self, output_dir):
        """Loop through the list of events, fit the drydown models, and update the Event intances' attributes"""
        self.output_dir = output_dir

        for i, event in enumerate(self.events):
            try:
                updated_event = self.fit_one_event(event)
                # Replace the old Event instance with updated one
                if updated_event is not None:
                    self.events[i] = updated_event
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        if self.plot_results:
            self.plot_drydown_models_in_timesreies()

    def fit_one_event(self, event):
        """Fit multiple drydown models for one event

        Args:
            event (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Currently, all the three models need to be fitted to return results

        # _____________________________________________
        # Fit tau exponential model
        if self.run_tau_exp_model:
            try:
                popt, pcov, y_opt, r_squared, aic, aicc, bic, _ = (
                    self.fit_tau_exp_model(event)
                )
                event.add_attributes(
                    "tau_exp", popt, pcov, y_opt, r_squared, aic, aicc, bic, ss_res, ss_tot
                )
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None

        # _____________________________________________
        # Fit tau exponential model
        if self.run_exp_model:
            try:
                popt, pcov, y_opt, r_squared, aic, aicc, bic, ss_res, ss_tot, _ = self.fit_exp_model(
                    event
                )

                if self.is_stage1ET_active:
                    est_theta_star = popt[2]
                else:
                    est_theta_star = self.norm_max
                est_theta_w = self.norm_min

                event.add_attributes(
                    "exp",
                    popt,
                    pcov,
                    y_opt,
                    r_squared,
                    aic,
                    aicc,
                    bic,
                    ss_res,
                    ss_tot, 
                    np.nan,
                    est_theta_star,
                    est_theta_w,
                )
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None

        # _____________________________________________
        # Fit q model
        if self.run_q_model:
            try:
                popt, pcov, y_opt, r_squared, aic, aicc, bic,ss_res, ss_tot, p_value = (
                    self.fit_q_model(event)
                )

                if self.is_stage1ET_active:
                    est_theta_star = popt[3]
                else:
                    est_theta_star = self.norm_max
                est_theta_w = self.norm_min

                event.add_attributes(
                    "q",
                    popt,
                    pcov,
                    y_opt,
                    r_squared,
                    aic,
                    aicc,
                    bic,
                    ss_res, 
                    ss_tot,
                    p_value,
                    est_theta_star,
                    est_theta_w,
                )
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None

        # _____________________________________________
        # Fit sigmoid model
        if self.run_sigmoid_model:
            try:
                popt, r_squared, y_opt = self.fit_sigmoid_model(event)
                event.add_attributes("sgm", popt, r_squared, y_opt)
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None
        # _____________________________________________
        # Finalize results for one event
        # if self.plot_results:
        #     self.plot_drydown_models(event)

        return event

    def fit_model(self, event, model, bounds, p0, param_names):
        """Base function for fitting models

        Args:
            event (_type_): _description_
            model (_type_): _description_
            bounds (_type_): _description_
            p0 (_type_): _description_
            norm (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        try:
            y_fit = event.y

            # Fit the model
            popt, pcov = curve_fit(
                f=model, xdata=event.x, ydata=y_fit, p0=p0, bounds=bounds
            )

            # Get the optimal fit
            y_opt = model(event.x, *popt)

            # Calculate the residuals
            r_squared, aic, aicc, bic, ss_res, ss_tot, p_value = self.calc_performance_metrics(
                y_obs=event.y,
                y_pred=y_opt,
                popt=popt,
                pcov=pcov,
                param_names=param_names,
            )

            return popt, pcov, y_opt, r_squared, aic, aicc, bic, ss_res, ss_tot, p_value

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

    def calc_performance_metrics(self, y_obs, y_pred, popt, pcov, param_names):

        # Residual sum of squares
        ss_res = np.sum((y_obs - y_pred) ** 2)

        # Total sum of squares
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)

        # Number of observations and parameters
        n = len(y_obs)
        k = len(popt)
        dof = n - k

        ############################
        # R^2 calculation
        r_squared = 1 - (ss_res / ss_tot)
        ############################

        ############################
        # AIC and BIC
        try:
            aic = n * np.log(ss_res / n) + 2 * k
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        except:
            aic = np.nan
            aicc = np.nan

        try:
            bic = n * np.log(ss_res / n) + k * np.log(n)
        except:
            bic = np.nan
        ############################

        if "q" in param_names:
            # t-scores
            # Find the index of parameter "q"
            q_index = param_names.index("q")

            # Extract estimate and standard error for "q"
            q_estimate = popt[q_index]
            q_variance = pcov[q_index, q_index]
            q_se = np.sqrt(q_variance)

            ############################
            # Compute the t-statistic: t_stat = param - 1 / param_SE
            t_stat = (q_estimate - 1) / q_se
            try:
                p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
            except:
                p_value = np.nan

        else:
            p_value = np.nan

        return r_squared, aic, aicc, bic, ss_res, ss_tot, p_value

    def fit_tau_exp_model(self, event):
        """Fits an exponential model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the boundary condition for optimizing the tau_exp_model(t, delta_theta, theta_w, tau)

        ### Delta_theta ###
        min_delta_theta = 0
        max_delta_theta = self.data.max_sm - self.data.min_sm
        ini_delta_theta = event.subset_sm_range

        ### Theta_w ###
        min_theta_w = self.data.min_sm
        max_theta_w = event.subset_min_sm
        ini_theta_w = (min_theta_w + max_theta_w) / 2

        ### Tau ###
        min_tau = 0  # self.z * (self.data.max_sm - event.subset_min_sm) / event.pet
        max_tau = np.inf
        ini_tau = 1

        bounds = [
            (min_delta_theta, min_theta_w, min_tau),
            (max_delta_theta, max_theta_w, max_tau),
        ]
        p0 = [ini_delta_theta, ini_theta_w, ini_tau]
        param_names = ["delta_theta", "theta_w", "tau"]

        # ______________________________________________________________________________________
        # Execute the event fit
        return self.fit_model(
            event=event,
            model=tau_exp_model,
            bounds=bounds,
            p0=p0,
            param_names=param_names,
        )

    def fit_exp_model(self, event):
        """Fits an exponential model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the boundary condition for optimizing the tau_exp_model(t, delta_theta, theta_w, tau)

        ### ETmax ###
        min_ETmax = 0
        if self.force_PET:
            max_ETmax = event.pet
            min_ETmax = event.pet * 0.2
        else:
            max_ETmax = np.inf
            min_ETmax = 0
        ini_ETmax = max_ETmax * 0.5

        ### theta_0 ###
        first_non_nan = event.y[~np.isnan(event.y)][0]
        min_theta_0 = first_non_nan - self.target_rmsd
        max_theta_0 = np.minimum(
            first_non_nan + self.target_rmsd, self.data.max_cutoff_sm
        )
        ini_theta_0 = first_non_nan

        ### theta_star ###
        # Filter out NaN values
        second_non_nan = event.y[~np.isnan(event.y)][1]

        if self.is_stage1ET_active:
            if np.isnan(event.est_theta_fc):
                max_theta_star = self.data.max_cutoff_sm
            else:
                max_theta_star = event.est_theta_fc

            if np.isnan(event.est_theta_star):
                min_theta_star = second_non_nan
            else:
                min_theta_star = np.maximum(event.est_theta_star, second_non_nan)
            ini_theta_star = (max_theta_star + min_theta_star) / 2

        # ______________________________________________________________________________________
        # Execute the event fit

        if self.is_stage1ET_active:
            bounds = [
                (min_ETmax, min_theta_0, min_theta_star),
                (max_ETmax, max_theta_0, max_theta_star),
            ]
            p0 = [ini_ETmax, ini_theta_0, ini_theta_star]
            param_names = ["ETmax", "theta_0", "theta_star"]
            return self.fit_model(
                event=event,
                model=lambda t, ETmax, theta_0, theta_star: exp_model_piecewise(
                    t=t,
                    ETmax=ETmax,
                    theta_0=theta_0,
                    theta_star=theta_star,
                    theta_w=self.norm_min,
                    z=self.z,
                ),
                bounds=bounds,
                p0=p0,
                param_names=param_names,
            )
        else:
            bounds = [(min_ETmax, min_theta_0), (max_ETmax, max_theta_0)]
            p0 = [ini_ETmax, ini_theta_0]
            param_names = ["ETmax", "theta_0"]
            return self.fit_model(
                event=event,
                model=lambda t, ETmax, theta_0: exp_model(
                    t=t,
                    ETmax=ETmax,
                    theta_0=theta_0,
                    theta_star=self.norm_max,
                    theta_w=self.norm_min,
                    z=self.z,
                ),
                bounds=bounds,
                p0=p0,
                param_names=param_names,
            )

    def fit_q_model(self, event):
        """Fits a q model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the boundary condition for optimizing q_model(t, k, q, delta_theta)

        ### q ###
        min_q = 0  # -np.inf
        max_q = np.inf
        ini_q = 1.0 + 1.0e-03

        ### ETmax ###
        if self.force_PET:
            max_ETmax = event.pet
            min_ETmax = event.pet * 0.2
        else:
            max_ETmax = np.inf
            min_ETmax = 0
        ini_ETmax = max_ETmax * 0.5

        ### theta_0 ###
        first_non_nan = event.y[~np.isnan(event.y)][0]
        min_theta_0 = first_non_nan - self.target_rmsd
        max_theta_0 = np.minimum(
            first_non_nan + self.target_rmsd, self.data.max_cutoff_sm
        )
        ini_theta_0 = first_non_nan

        ### theta_star ###
        second_non_nan = event.y[~np.isnan(event.y)][1]
        if self.is_stage1ET_active:
            if np.isnan(event.est_theta_fc):
                max_theta_star = self.data.max_cutoff_sm
            else:
                max_theta_star = event.est_theta_fc

            if np.isnan(event.est_theta_star):
                min_theta_star = second_non_nan
            else:
                min_theta_star = np.maximum(event.est_theta_star, second_non_nan)
            ini_theta_star = (max_theta_star + min_theta_star) / 2

        # ______________________________________________________________________________________
        # Execute the event fit

        if self.is_stage1ET_active:
            bounds = [
                (min_q, min_ETmax, min_theta_0, min_theta_star),
                (max_q, max_ETmax, max_theta_0, max_theta_star),
            ]
            p0 = [ini_q, ini_ETmax, ini_theta_0, ini_theta_star]
            param_names = ["q", "ETmax", "theta_0", "theta_star"]
            return self.fit_model(
                event=event,
                model=lambda t, q, ETmax, theta_0, theta_star: q_model_piecewise(
                    t=t,
                    q=q,
                    ETmax=ETmax,
                    theta_0=theta_0,
                    theta_star=theta_star,
                    theta_w=self.norm_min,
                    z=self.z,
                ),
                bounds=bounds,
                p0=p0,
                param_names=param_names,
            )
        else:
            bounds = [(min_q, min_ETmax, min_theta_0), (max_q, max_ETmax, max_theta_0)]
            p0 = [ini_q, ini_ETmax, ini_theta_0]
            param_names = ["q", "ETmax", "theta_0"]
            return self.fit_model(
                event=event,
                model=lambda t, q, ETmax, theta_0: q_model(
                    t=t,
                    q=q,
                    ETmax=ETmax,
                    theta_0=theta_0,
                    theta_star=self.norm_max,
                    theta_w=self.norm_min,
                    z=self.z,
                ),
                bounds=bounds,
                p0=p0,
                param_names=param_names,
            )

    def fit_sigmoid_model(self, event):
        """Base function for fitting models

        Args:
            event (_type_): _description_
            model (_type_): _description_
            bounds (_type_): _description_
            p0 (_type_): _description_
            norm (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        try:
            # Observed time series data
            t_obs = event.x
            y_obs = event.y
            y_init = y_obs[
                0
            ]  # Initial condition (assuming that the first observed data point is the initial condition)

            # Initial guess for parameters theta50, k, a
            PET = event.pet

            ini_theta50 = 0.5
            ini_k = 1
            ini_a = PET / 50

            min_theta50 = 0.0
            min_k = 0.0
            min_a = 0.0

            max_theta50 = event.max_sm
            max_k = np.inf
            max_a = PET / 50 * 100

            initial_guess = [ini_theta50, ini_k, ini_a]
            bounds = [
                (min_theta50, max_theta50),
                (min_k, max_k),
                (min_a, max_a),
            ]

            # Perform the optimization
            result = minimize(
                objective_function,
                initial_guess,
                args=(y_obs, y_init, t_obs),
                method="L-BFGS-B",
                bounds=bounds,
            )  # You can choose a different method if needed

            # The result contains the optimized parameters
            theta50_best, k_best, a_best = result.x
            best_solution = solve_ivp(
                lambda t, theta: loss_sigmoid(t, theta, theta50_best, k_best, a_best),
                [t_obs[0], t_obs[-1]],
                [y_init],
                t_eval=t_obs,
            )

            # Get the optimal fit
            y_opt = best_solution.y[0]

            # Calculate the residuals
            popt = result.x
            residuals = event.y - y_opt
            ss_res = np.sum(residuals**2)
            r_squared = 1 - ss_res / np.sum((event.y - np.nanmean(event.y)) ** 2)

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

    def return_result_df(self):
        """Return results in the pandas dataframe format for easier concatination"""

        results = []
        for event in self.events:
            try:
                _results = {
                    "EASE_row_index": self.data.EASE_row_index,
                    "EASE_column_index": self.data.EASE_column_index,
                    "event_start": event.start_date,
                    "event_end": event.end_date,
                    "time": event.x,
                    "sm": event.y,
                    "min_sm": event.min_sm,
                    "max_sm": event.max_sm,
                    "est_theta_fc": event.est_theta_fc,
                    "pet": event.pet,
                }

                if self.run_tau_exp_model:
                    _results.update(
                        {
                            "tauexp_delta_theta": event.tau_exp["delta_theta"],
                            "tauexp_theta_w": event.tau_exp["theta_w"],
                            "tauexp_tau": event.tau_exp["tau"],
                            "tauexp_var_delta_theta": event.tau_exp["var_delta_theta"],
                            "tauexp_var_theta_w": event.tau_exp["var_theta_w"],
                            "tauexp_var_tau": event.tau_exp["var_tau"],
                            "tauexp_cov_delta_theta_theta_w": event.tau_exp[
                                "cov_delta_theta_theta_w"
                            ],
                            "tauexp_cov_delta_theta_tau": event.tau_exp[
                                "cov_delta_theta_tau"
                            ],
                            "tauexp_cov_theta_w_tau": event.tau_exp["cov_theta_w_tau"],
                            "tauexp_r_squared": event.tau_exp["r_squared"],
                            "tauexp_aic": event.tau_exp["aic"],
                            "tauexp_aicc": event.tau_exp["aicc"],
                            "tauexp_bic": event.tau_exp["bic"],
                            "tauexp_ss_res": event.tau_exp["ss_res"],
                            "tauexp_ss_tot": event.tau_exp["ss_tot"],
                            "tauexp_y_opt": event.tau_exp["y_opt"],
                        }
                    )

                if self.run_exp_model:
                    _results.update(
                        {
                            "exp_ETmax": event.exp["ETmax"],
                            "exp_theta_0": event.exp["theta_0"],
                            "exp_theta_star": event.exp["theta_star"],
                            "exp_theta_w": event.exp["theta_w"],
                            "exp_var_ETmax": event.exp["var_ETmax"],
                            "exp_var_theta_0": event.exp["var_theta_0"],
                            "exp_var_theta_star": event.exp["var_theta_star"],
                            "exp_cov_ETmax_theta_0": event.exp["cov_ETmax_theta_0"],
                            "exp_cov_ETmax_theta_star": event.exp[
                                "cov_ETmax_theta_star"
                            ],
                            "exp_cov_theta_0_theta_star": event.exp[
                                "cov_theta_0_theta_star"
                            ],
                            "exp_r_squared": event.exp["r_squared"],
                            "exp_aic": event.exp["aic"],
                            "exp_aicc": event.exp["aicc"],
                            "exp_bic": event.exp["bic"],
                            "exp_ss_res": event.exp["ss_res"],
                            "exp_ss_tot": event.exp["ss_tot"],
                            "exp_y_opt": event.exp["y_opt"],
                        }
                    )

                if self.run_q_model:
                    _results.update(
                        {
                            "q_q": event.q["q"],
                            "q_ETmax": event.q["ETmax"],
                            "q_theta_0": event.q["theta_0"],
                            "q_theta_star": event.q["theta_star"],
                            "q_theta_w": event.q["theta_w"],
                            "q_var_q": event.q["var_q"],
                            "q_var_ETmax": event.q["var_ETmax"],
                            "q_var_theta_0": event.q["var_theta_0"],
                            "q_var_theta_star": event.q["var_theta_star"],
                            "q_cov_q_ETmax": event.q["cov_q_ETmax"],
                            "q_cov_q_theta_0": event.q["cov_q_theta_0"],
                            "q_cov_q_theta_star": event.q["cov_q_theta_star"],
                            "q_cov_ETmax_theta_0": event.q["cov_ETmax_theta_0"],
                            "q_cov_ETmax_theta_star": event.q["cov_ETmax_theta_star"],
                            "q_cov_theta_0_theta_star": event.q[
                                "cov_theta_0_theta_star"
                            ],
                            "q_r_squared": event.q["r_squared"],
                            "q_aic": event.q["aic"],
                            "q_aicc": event.q["aicc"],
                            "q_bic": event.q["bic"],
                            "q_ss_res": event.q["ss_res"],
                            "q_ss_tot": event.q["ss_tot"],
                            "q_eq_1_p": event.q["q_eq_1_p"],
                            "q_y_opt": event.q["y_opt"],
                        }
                    )

                if self.run_sigmoid_model:
                    _results.update(
                        {
                            "sgm_theta50": event.sgm["theta50"],
                            "sgm_k": event.sgm["k"],
                            "sgm_a": event.sgm["a"],
                            "sgm_r_squared": event.sgm["r_squared"],
                            "sgm_y_opt": event.sgm["y_opt"],
                        }
                    )

                # Now, _results contains only the relevant fields based on the boolean flags.
                results.append(_results)

            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                continue

        # Convert results into dataframe
        df_results = pd.DataFrame(results)

        # If the result is empty, return nothing
        if not results:
            return pd.DataFrame()
        else:
            return df_results

    def plot_drydown_models(self, event, ax=None):
        # Plot exponential model
        date_range = pd.date_range(start=event.start_date, end=event.end_date, freq="D")
        x = date_range[event.x]

        # Create a figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        # ______________________________________
        # Plot exponential model
        if self.run_tau_exp_model:
            try:
                ax.plot(
                    x,
                    event.tau_exp["y_opt"],
                    alpha=0.7,
                    linestyle="--",
                    color="tab:blue",
                )

                label = rf"t-exp: $R^2$={event.tau_exp['r_squared']:.2f}; $\tau$={event.tau_exp['tau']:.2f}"

                ax.text(
                    x[0],
                    event.tau_exp["y_opt"][0] + 0.10,
                    f"{label}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="tab:blue",
                )

            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ______________________________________
        # Plot exponential model (more physically constrained)
        if self.run_exp_model:
            try:
                ax.plot(
                    x,
                    event.exp["y_opt"],
                    alpha=0.7,
                    linestyle=":",
                    color="tab:blue",
                )

                label = rf"exp: $R^2$={event.exp['r_squared']:.2f}; $ETmax$={event.exp['ETmax']:.2f}"

                ax.text(
                    x[0],
                    event.exp["y_opt"][0] + 0.08,
                    f"{label}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="tab:blue",
                )

            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ______________________________________
        # Plot q model
        if self.run_q_model:
            try:
                ax.plot(
                    x,
                    event.q["y_opt"],
                    alpha=0.7,
                    linestyle="--",
                    color="tab:orange",
                )

                label = rf"q model: $R^2$={event.q['r_squared']:.2f}; $q$={event.q['q']:.2f}"

                ax.text(
                    x[0],
                    event.q["y_opt"][0] + 0.001,
                    f"{label}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="tab:orange",
                )

            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ______________________________________
        # Plot sigmoid model
        if self.run_sigmoid_model:
            try:
                ax.plot(
                    x,
                    event.sgm["y_opt"],
                    alpha=0.7,
                    linestyle="--",
                    color="blue",
                )

                label = rf"sigmoid model: $R^2$={event.sgm['r_squared']:.2f}; $k$={event.sgm['k']:.2f}"
                ax.text(
                    x[0],
                    event.sgm["y_opt"][0] - 0.03,
                    f"{label}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="blue",
                )

            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # Close the current figure to release resources
        plt.close()

    def plot_drydown_models_in_timesreies(self):
        years_of_record = max(self.data.df.index.year) - min(self.data.df.index.year)
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20 * years_of_record, 5))

        ax11.scatter(
            self.data.df.sm_unmasked.index,
            self.data.df.sm_unmasked.values,
            color="grey",
            label="SMAP observation",
            s=1.0,
        )
        ax11.scatter(
            self.data.df.sm_masked[self.data.df["event_start"]].index,
            self.data.df.sm_masked[self.data.df["event_start"]].bfill().values,
            color="grey",
            alpha=0.5,
        )
        ax11.scatter(
            self.data.df.sm_masked[self.data.df["event_end"]].index,
            self.data.df.sm_masked[self.data.df["event_end"]].bfill().values,
            color="grey",
            marker="x",
            alpha=0.5,
        )
        ax11.axhline(y=self.norm_max, color="tab:grey", linestyle="--", alpha=0.5)
        ax11.set_ylabel("VSWC[m3/m3]")
        self.data.df.precip.plot(ax=ax12, alpha=0.5, color="tab:grey")
        ax12.set_ylabel("Precipitation[mm/d]")

        for event in self.events:
            self.plot_drydown_models(event, ax=ax11)

        # Save results
        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_events_in_ts.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir2, filename))

        plt.close()
