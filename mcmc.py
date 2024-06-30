import numpy as np
import emcee
import multiprocessing
from typing import Callable


class MCMCUtils:
    """
    A event is something contains both datapoints and a model.
    It should always contain following methods:
    1. set_params(params_free, params_to_fit)
    2. get_init_params(params_to_fit), return a list of initial parameters.
    3. ln_prob(), return a scalar value of ln probability.
    4. __str__(), this is optional, but better to have in order to diagnose why the fitting fails.

    If necessary, one can also define a prior function to constrain the parameter space.
    The prior function should has 2 inputs, params_free and params_to_fit, and return a scalar value.
    """

    def __init__(self, params_lowup_dict={}, prior: Callable | None = None) -> None:
        self.params_lowup_dict = params_lowup_dict
        self.prior = prior
        pass

    @staticmethod
    def flat_prior(params_free, params_to_fit, params_lowup_dict={}):
        for i in range(len(params_free)):
            if params_to_fit[i] in params_lowup_dict.keys():
                lower_lim = params_lowup_dict[params_to_fit[i]][0]
                upper_lim = params_lowup_dict[params_to_fit[i]][1]
                if params_free[i] < lower_lim or params_free[i] > upper_lim:
                    return np.inf
        # for i in range(len(params_free)):
        #    if params_to_fit[i] == "vcorr":
        #        if params_free[i] < -1000 or params_free[i] > 1000:
        #            return np.inf
        return 0

    @staticmethod
    def ln_prob(params_free, event, params_to_fit, params_scale_loc_dict):
        event.set_params(params_free, params_to_fit)
        l_prior = MCMCUtils.flat_prior(
            params_free, params_to_fit, params_scale_loc_dict
        )
        l_prob = event.ln_prob()
        if np.isinf(l_prior) or np.isinf(l_prob):
            return -np.inf
        return l_prior + l_prob

    def emcee_fitting(
        self,
        event,
        params_to_fit: list,
        chain_path: str,
        nwalkers: int = 100,
        nburn: int = 1000,
        nstep: int = 1000,
        nthread: int = 1,
        print_progress: bool = False,
        print_info=False,
    ):
        ndim = len(params_to_fit)
        # The initial positions of the walkers should include the init_params
        # The init_params is usually the best fit parameters from other methods.
        # The obname does not matter, it is the same across all models.

        init_params = event.get_init_params(params_to_fit)
        params_scales = []
        # for param_name in params_to_fit:
        #    assert (
        #        param_name in params_scale_loc_dict.keys()
        #    ), f"{param_name} is not in params_scale_loc_dict"
        #    params_scales.append(0.01 * params_scale_loc_dict[param_name]["scale"])

        pos = [init_params + (0.1 * np.random.randn(ndim)) for _ in range(nwalkers - 1)]  # type: ignore
        pos += [init_params]

        if nthread == 1:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                MCMCUtils.ln_prob,
                args=(
                    event,
                    params_to_fit,
                    self.params_lowup_dict,
                ),
            )
            sampler.run_mcmc(pos, nburn + nstep, progress=print_progress)
        else:
            with multiprocessing.Pool(nthread) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    MCMCUtils.ln_prob,
                    args=(
                        event,
                        params_to_fit,
                        self.params_lowup_dict,
                    ),
                    pool=pool,
                )
                sampler.run_mcmc(pos, nburn + nstep, progress=print_progress)
        ## save EMCEE results ##
        sampler.chain.reshape((-1, ndim))
        params_chain = sampler.chain[:, nburn:, :].reshape((-1, ndim), order="F")
        chi2_chain = -2 * sampler.lnprobability[:, nburn:].reshape(-1, order="F")
        header_str = ",".join(params_to_fit) + ",chi2"
        save_chain = np.vstack([params_chain.T, chi2_chain]).T
        if chain_path is not None:
            np.savetxt(
                chain_path,
                save_chain,
                delimiter=",",
                header=header_str,
                comments="",
            )
        ## Find the best fit ##
        params_best = params_chain[np.argmin(chi2_chain), :]
        params_errs = np.percentile(params_chain, q=[16, 84], axis=0)
        if print_info:
            print(f"Best chi2: {chi2_chain.min()}")
            print("Best parameters: ")
            for i in range(len(params_best)):
                print(
                    f"{params_to_fit[i]} = {params_best[i]:.6f} +{params_errs[0, i]} -{params_errs[1, i]:.6f}"
                )
        return params_best, params_errs, np.min(chi2_chain)
