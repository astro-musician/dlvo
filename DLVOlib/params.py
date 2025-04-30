n_walkers = 4

import numpyro
numpyro.set_host_device_count(n_walkers)
import numpy as np 
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
import arviz as az
import pickle
from pathlib import Path

from .dlvo_model import dlvo_model, dlvo_err
from .utils import key_mcmc

class DLVO_parameters:

    def __init__(self,datafile:str,cutoff=0,force_err=0.5,rerun=False):

        """
        datafile : chemin du fichier de données
        cutoff : seuil à partir duquel on souhaite fit le profil DLVO (nm)
        """

        self.datafile = datafile 
        self.cutoff = cutoff 
        self.force_err = force_err
        self.rerun = rerun
        self.data = np.loadtxt(self.datafile, skiprows=1, delimiter=",")  # Ignore la première ligne
        self.data = self.data[~np.isnan(self.data).any(axis=1)]  # Supprime les lignes contenant NaN
        self.chainsfile = f'{self.datafile}_chains.pkl'

        self.setup_data()

        pass

    def model(self):

        alpha = numpyro.sample("alpha",dist.Uniform(low=0,high=1000))
        k = numpyro.sample("k",dist.Uniform(low=0,high=1))
        beta = numpyro.sample("beta",dist.Uniform(low=0,high=1000))

        mock_model = dlvo_model(self.x,alpha,k,beta)

        numpyro.sample("likelihood",dist.Normal(mock_model,self.force_err),obs=self.F)


    def setup_data(self):

        # Extraire x et F
        full_x = self.data[:, 0]  # Distance (nm)
        full_F = self.data[:, 1]  # Force (pN)

        # Trier x si besoin
        idx = np.argsort(full_x)
        full_x, full_F = full_x[idx], full_F[idx]

        self.x = jnp.array(full_x)[full_x>self.cutoff]
        self.F = jnp.array(full_F)[full_x>self.cutoff]

        pass

    def setup_mcmc(self,num_warmup,num_samples):

        self.initial_guess = {
                            'alpha':1,
                            'beta':1,
                            'k':0.01
                        }

        self.kernel = NUTS(self.model,init_strategy=init_to_value(values=self.initial_guess))
        self.mcmc = MCMC(self.kernel,num_warmup=num_warmup,num_samples=num_samples,num_chains=n_walkers,progress_bar=True)

        pass

    def run_mcmc(self,num_warmup=2000,num_samples=2000):

        self.setup_mcmc(num_warmup=num_warmup,num_samples=num_samples)
        self.mcmc.run(key_mcmc(n_walkers))
        self.chains = self.mcmc.get_samples(group_by_chain=True)

        with open(self.chainsfile,'wb') as f:
            pickle.dump(self.chains,f)

        pass

    def get_parameters(self):
        
        if Path(self.chainsfile).is_file() and not self.rerun:

            with open(self.chainsfile,'rb') as f:
                self.chains = pickle.load(f)

        else:
            self.run_mcmc()

        dataset = az.convert_to_inference_data(self.chains)
        print(az.rhat(dataset))
        with az.style.context('arviz-darkgrid', after_reset=True):
            az.plot_trace(dataset)
            plt.savefig(f'{self.datafile}_chains.pdf',dpi=800)
            plt.close()

        self.chains_alpha = self.chains['alpha'].flatten()
        self.chains_beta = self.chains['beta'].flatten()
        self.chains_k = self.chains['k'].flatten()

        self.alpha = np.median(self.chains_alpha)
        self.beta = np.median(self.chains_beta)
        self.k = np.median(self.chains_k)

        pass

    def compare_model(self):

        self.F_model = dlvo_model(self.x,self.alpha,self.k,self.beta)
        self.F_model_err_1s = dlvo_err(self.x,self.chains_alpha,self.chains_k,self.chains_beta)
        self.F_model_err_low = self.F_model - self.F_model_err_1s
        self.F_model_err_high = self.F_model + self.F_model_err_1s

        fig, ax = plt.subplots(1,1,layout='constrained')

        ax.plot(self.x,self.F,label='Data')
        ax.plot(self.x,self.F_model,color='red',label='Best fit')
        ax.fill_between(
                        self.x, 
                        y1=self.F_model_err_low, 
                        y2=self.F_model_err_high, 
                        color='red', 
                        alpha=0.5
                        )
        ax.set_xlabel('Distance (nm)',fontsize=15)
        ax.set_ylabel('Force (pN)',fontsize=15)
        ax.legend(fontsize=15)

        plt.savefig(f'{self.datafile}_compare_model.pdf',dpi=800)
        plt.close()

        pass