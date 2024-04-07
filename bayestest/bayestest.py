import numpy as np
import pymc as pm
from tabulate import tabulate
import pandas as pd

class Variant:
    def __init__(self, name, visitors, successes, revenue=None, control=False):
        self.name = name
        self.visitors = visitors
        self.successes = successes
        self.control = control
        self.revenue = revenue  # Initialize without revenue data
        self.trace = None
        self.model = None

    def add_revenue(self, total_revenue):
        self.revenue = total_revenue

class bayesTest:
    def __init__(self):
        self.variants = []
        #self.model = None
        self.trace = None
        self.alpha = 1  # Default alpha for Beta prior
        self.beta = 1   # Default beta for Beta prior
        self.revenue_alpha = None
        self.revenue_test = False
        self.results = None

    def prior(self, alpha=1, beta=1, revenue=False, plot=False):
        if revenue:
            self.revenue_alpha = alpha
            self.revenue_beta = beta
            return 
        else:
            self.alpha = alpha
            self.beta = beta
        
        if plot:
            # plot
            pass

    def add(self, visitors, successes, revenue=None, name=None, control=False):
        variant = Variant(name, visitors, successes, revenue, control)
        self.variants.append(variant)

    def infer(self, samples=5000):
        
        # Sample Conversion Rate posterior
        for variant in self.variants:
            with pm.Model():
                # Copied from PyMC directly:
                p = pm.Beta("p", alpha=self.alpha, beta=self.beta)
                y = pm.Binomial("y", n=variant.visitors, p=p, observed=variant.successes)
                trace = pm.sample(draws=samples)

                variant.trace = trace

    def summary(self):
        
        all_simulations = []
        
        for variant in self.variants:
           all_simulations.append(variant.trace.posterior.p.values.flatten())
        
        # Stack the posterior samples into a single NumPy array for efficient computation
        # The shape of the stacked array would be (num_samples, num_variants)
        posterior_stacked = np.stack(all_simulations, axis=1)
        
        # Identify the index of the variant with the highest value for each sample
        winning_variants = np.argmax(posterior_stacked, axis=1)
        
        # Calculate the winning probability for each variant
        # This computes how often each variant wins, i.e., has the highest posterior sample value
        winning_probabilities = np.mean(winning_variants == np.arange(posterior_stacked.shape[1])[:, None], axis=1)
        

        summary_df = pd.DataFrame({'Variant': [variant.name for variant in self.variants],
                      'Prob. Winner CR': [f'{x:.2f}%' for x in winning_probabilities],
                      'Prob. Winner Rev': ['NA']*len(self.variants)
                      })
        
        print(tabulate(summary_df, headers='keys', tablefmt='grid'))