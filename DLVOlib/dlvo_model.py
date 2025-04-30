import numpy as np
import jax.numpy as jnp

def dlvo_fact_term(h,a):
    fact_term_1 = 2*a**2*(2*h+4*a)/(h**2*(h+4*a)**2)
    fact_term_2 = 4*a**2/(h+2*a)**3
    fact_term_3 = -8*a**2/(h*(h+2*a)*(h+4*a))
    return fact_term_1+fact_term_2+fact_term_3

def dlvo_model(h,alpha,k,beta,a=500):
    alpha_k_term = alpha*jnp.exp(-k*h)
    return alpha_k_term + beta*dlvo_fact_term(h,a)

# def dlvo_err(percent,h,alpha_chain,k_chain,beta_chain,a=500):
#     dlvo_sample = dlvo_model(h[...,None],alpha_chain[None,...],k_chain[None,...],beta_chain[None,...],a=a)
#     low_percent = (100-percent)/2
#     high_percent = 100-(100-percent)/2
#     err_low = np.percentile(dlvo_sample,low_percent,axis=1)
#     err_high = np.percentile(dlvo_sample,high_percent,axis=1)
#     return np.array([err_low,err_high])

def dlvo_err(h,alpha_chain,k_chain,beta_chain,a=500):

    med_alpha = np.median(alpha_chain)
    med_k = np.median(k_chain)
    med_beta = np.median(beta_chain)
    err_alpha = np.std(alpha_chain)
    err_k = np.std(k_chain)
    err_beta = np.std(beta_chain)

    d_dlvo_alpha = np.exp(np.exp(-med_k*h))
    d_dlvo_k = -med_k*med_alpha*np.exp(-med_k*h)
    d_dlvo_beta = dlvo_fact_term(h,a)

    return np.sqrt(d_dlvo_alpha**2*err_alpha**2+d_dlvo_k*err_k**2+d_dlvo_beta*err_beta**2)