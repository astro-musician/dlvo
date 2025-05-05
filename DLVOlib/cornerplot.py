import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class cornerplot:

    def __init__(self,*chains,confidence_percent=68):

        self.n_params = len(chains)
        self.chains = np.array([chain for chain in chains])
        self.estimations = np.median(self.chains,axis=1)
        self.errors_low = np.percentile(self.chains,(100-confidence_percent)/2,axis=1)
        self.errors_high = np.percentile(self.chains,50+confidence_percent/2,axis=1)

        self.magnitude_orders = np.intc(np.floor(np.log10(self.estimations)))

        self.estimate_posteriors()
        # self.estimate_multivariate_posteriors()

        pass

    def estimate_posteriors(self):

        posterior_range = 1000
        self.xs = np.zeros((self.n_params,posterior_range))
        self.posteriors = []

        for i in range(self.n_params):

            chain = self.chains[i]
            self.xs[i] = np.linspace(np.min(chain),np.max(chain),posterior_range)
            self.posteriors.append(gaussian_kde(chain))

        pass

    def estimate_multivariate_posteriors(self):

        self.multivariate_posteriors = []

        for i in range(self.n_params):
            self.multivariate_posteriors.append([])
            for j in range(i):

                X,Y = np.meshgrid(self.xs[i],self.xs[j])
                positions = np.vstack([X.ravel(),Y.ravel()])
                values = np.vstack([self.chains[i],self.chains[j]])
                kernel = gaussian_kde(values)
                self.multivariate_posteriors[i].append(kernel.evaluate(positions).reshape(X.shape).T)

        pass

    def plot(self,*labels):

        fig, axes = plt.subplots(self.n_params,self.n_params,figsize=(7,6))
        plt.subplots_adjust(wspace=0,hspace=0)
        ticks = []

        for i in range(self.n_params):
            axes[i,i].plot(self.xs[i],self.posteriors[i](self.xs[i]))
            axes[i,i].set_xlim([np.min(self.xs[i]),np.max(self.xs[i])])
            axes[i,i].set_ylim([0,None])
            axes[i,i].set_yticks([])
            axes[i,i].set_title(
                f'{labels[i]}={round(float(self.estimations[i]),-self.magnitude_orders[i]+2)}'
                f'$^{{+{round(float(self.errors_high[i]-self.estimations[i]),-self.magnitude_orders[i]+2)}}}'
                f'_{{-{round(float(self.estimations[i]-self.errors_low[i]),-self.magnitude_orders[i]+2)}}}$'
            )

            if i!=self.n_params-1:
                axes[i,i].set_xticks([])
            if i==0:
                axes[i,i].set_ylabel(labels[0])
            if i==self.n_params-1:
                axes[i,i].set_xlabel(labels[-1])

            axes[i,i].ticklabel_format(scilimits=(-3,3))
            axes[i,i].tick_params('y',rotation=90)

        for i in range(self.n_params):
            for j in range(i):
                # axes[i,j].contour(self.xs[j],self.xs[i],self.multivariate_posteriors[i][j],levels=3)
                axes[i,j].scatter(self.chains[j],self.chains[i],s=0.01)
                axes[i,j].set_xlim([np.min(self.xs[j]),np.max(self.xs[j])])
                axes[i,j].set_ylim([np.min(self.xs[i]),np.max(self.xs[i])])
                axes[i,j].ticklabel_format(scilimits=(-3,3))
                axes[i,j].tick_params('y',rotation=90)
                if i!=self.n_params-1:
                    axes[i,j].set_xticks([])
                if (j!=0):
                    axes[i,j].set_yticks([])
                if j==0:
                    axes[i,j].set_ylabel(labels[i])
                    if i==self.n_params-1:
                        axes[i,j].set_ylabel(labels[-1])
                if i==self.n_params-1:
                    axes[i,j].set_xlabel(labels[j])

            for j in range(i+1,self.n_params):
                axes[i,j].axis('off')
                axes[i,j].get_xaxis().set_ticks([])
                axes[i,j].get_yaxis().set_ticks([])

        plt.savefig('test_cornerplot.pdf',dpi=800)

        pass