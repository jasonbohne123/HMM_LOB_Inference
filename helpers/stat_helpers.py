import numpy as np
from scipy.stats import t

def compute_pval(row):
    """ Perform two t-test to determine if distributions are significantly different 
    """
    test_stat=(row.loc['mu2']-row.loc['mu1'])/np.sqrt((row.loc['sigma1']**2)+(row.loc['sigma2']**2)/20)
  
    df_num=((row.loc['sigma1']**2)+(row.loc['sigma2']**2)/20)**2
    df_dem=(((row.loc['sigma1']**2)/20)**2)/19+(((row.loc['sigma2']**2)/20)**2)/19
    df=df_num/df_dem


    p_val=2*np.round(t.cdf(-abs(test_stat),df),4)
    return p_val