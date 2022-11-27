import numpy as np
import psg
import pandas as pd
from datetime import date, timedelta
from hmmlearn.hmm import GaussianHMM

from features import prep_features
from stat_helpers import compute_pval



def extract_params(param_dict,method):
    """ Extract Params from optimized model 
    """
    if method==1:
        param_df=pd.DataFrame.from_dict(param_dict,orient='index',columns=['p1','p2','a11','a12', 'a21', 'a22','mu1','si1','mu2','si2'])
        param_df=param_df.rename(columns={'si1':'sigma1','si2':'sigma2'})
        
        p_vals=[]
        # standardizing mu1 < mu2 as arbitrarily labeled, swapping mus and sigmas if needed 
        for x,row in param_df.iterrows():

            p_vals.append(compute_pval(row))

            if row.loc['mu1']<row.loc['mu2']:
                continue
            
            mu1,mu2,sigma1,sigma2=row.loc['mu1'],row.loc['mu2'],row.loc['sigma1'],row.loc['sigma2']
            
            row.loc['mu1'],row.loc['mu2'],row.loc['sigma1'],row.loc['sigma2']=mu2,mu1,sigma2,sigma1

        param_df=param_df.drop(columns=['p1','p2'])
        param_df['p_val']=pd.Series(p_vals,index=param_df.index)
        return param_df

    elif method==2:
        param_df=pd.DataFrame.from_dict(param_dict,orient='index')
        means= pd.DataFrame(param_df['Mean'].to_list(),columns=['mu1','mu2'])
        covar=pd.DataFrame(param_df['Sigma'].to_list(), columns = ['sigma1', 'sigma2'])
        transit=pd.DataFrame(param_df['Transition'].to_list(), columns = ['a11', 'a12','a21','a22'])
        dates=pd.DataFrame(param_df.index,columns=['date'])
        
        new_param_df=pd.DataFrame(pd.concat([means,covar,transit,dates],axis=1))
        new_param_df.index=new_param_df['date'].values
        new_param_df=new_param_df.drop(columns=['date'])

        p_vals=[]

       # standardizing mu1 < mu2 as arbitrarily labeled, swapping mus and sigmas if needed 
        for x,row in new_param_df.iterrows():
            p_vals.append(compute_pval(row))
            if row.loc['mu1']<row.loc['mu2']:
                continue
            mu1,mu2,sigma1,sigma2=row.loc['mu1'],row.loc['mu2'],row.loc['sigma1'],row.loc['sigma2']
            
            row.loc['mu1'],row.loc['mu2'],row.loc['sigma1'],row.loc['sigma2']=mu2,mu1,sigma2,sigma1

        new_param_df=new_param_df[['a11','a12','a21','a22','mu1','sigma1','mu2','sigma2']]
        
        new_param_df['p_val']=pd.Series(p_vals,index=new_param_df.index)
        return new_param_df


def fit_hmm(method):
    """ Fit HMM model with PSG and HMMLearn 
    """
    start=date(2020,1,1)
    days=[start+timedelta(days=i) for i in range(0,30)]

    spread_params={}
    bidsize_params={}
    offersize_params={}
    bookimbalance_params={}
    
    # psg training
    if method==1:
        for dt in days:
            try:
                dt_features=prep_features(dt)
            except:
                continue
            
            # formatted as numpy float 
            np.savetxt(r'psg_text_hmm/vector_bidsize.txt', dt_features['Bid_Size'])
            np.savetxt(r'psg_text_hmm/vector_offersize.txt', dt_features['Offer_Size'])
            np.savetxt(r'psg_text_hmm/vector_bookimbalance.txt', dt_features['OB_IB'])
            np.savetxt(r'psg_text_hmm/vector_spread.txt', dt_features['spread'])


            psg_spread_prob = psg.psg_importfromtext('./psg_text_hmm/problem_hmm_normal_spread.txt')
            psg_spread_prob['problem_statement'] = '\n'.join(psg_spread_prob['problem_statement'])
            spread_solution=psg.psg_solver(psg_spread_prob)
            params=list(spread_solution.values())[4][1]
            spread_params[dt]=params

            psg_bidsize_prob = psg.psg_importfromtext('./psg_text_hmm/problem_hmm_normal_bidsize.txt')
            psg_bidsize_prob['problem_statement'] = '\n'.join(psg_bidsize_prob['problem_statement'])
            bidsize_solution=psg.psg_solver(psg_bidsize_prob)
            params=list(bidsize_solution.values())[4][1]
            bidsize_params[dt]=params

            psg_offersize_prob = psg.psg_importfromtext('./psg_text_hmm/problem_hmm_normal_offersize.txt')
            psg_offersize_prob['problem_statement'] = '\n'.join(psg_offersize_prob['problem_statement'])
            offersize_solution=psg.psg_solver(psg_offersize_prob)
            params=list(offersize_solution.values())[4][1]
            offersize_params[dt]=params

            psg_bookimbalance_prob = psg.psg_importfromtext('./psg_text_hmm/problem_hmm_normal_bookimbalance.txt')
            psg_bookimbalance_prob['problem_statement'] = '\n'.join(psg_bookimbalance_prob['problem_statement'])
            bookimbalance_solution=psg.psg_solver(psg_bookimbalance_prob)
            params=list(bookimbalance_solution.values())[4][1]
            bookimbalance_params[dt]=params
            
    elif method==2:
        for dt in days:
            
            try:
                dt_features=prep_features(dt)
            except:
                continue
            print(f"Fitting HMM usign HMM-Learn for {dt}")
            
            
            spread=dt_features['spread'].reshape(-1, 1)
            spread_model=GaussianHMM(n_components=2,algorithm='viterbi',covariance_type="spherical",min_covar=1e-4, n_iter=1000,tol=1e-8)
            fitted_spread_model=spread_model.fit(spread)
            spread_mu=fitted_spread_model.means_.flatten()
            spread_covar=fitted_spread_model.covars_.flatten()
            spread_transit=fitted_spread_model.transmat_.flatten()
            spread_params[dt]={"Mean":spread_mu, "Sigma":spread_covar,"Transition":spread_transit}

            bidsize=dt_features['Bid_Size'].reshape(-1, 1)
            bidsize_model=GaussianHMM(n_components=2,algorithm='viterbi',covariance_type="spherical",min_covar=1e-4, n_iter=1000,tol=1e-8)
            fitted_bidsize_model=bidsize_model.fit(bidsize)
            bidsize_mu=fitted_bidsize_model.means_.flatten()
            bidsize_covar=fitted_bidsize_model.covars_.flatten()
            bidsize_transit=fitted_bidsize_model.transmat_.flatten()
            bidsize_params[dt]={"Mean":bidsize_mu, "Sigma":bidsize_covar,"Transition":bidsize_transit}

            offersize=dt_features['Offer_Size'].reshape(-1, 1)
            offersize_model=GaussianHMM(n_components=2,algorithm='viterbi',covariance_type="spherical",min_covar=1e-4, n_iter=1000,tol=1e-8)
            fitted_offersize_model=offersize_model.fit(offersize)
            offersize_mu=fitted_offersize_model.means_.flatten()
            offersize_covar=fitted_offersize_model.covars_.flatten()
            offersize_transit=fitted_offersize_model.transmat_.flatten()
            offersize_params[dt]={"Mean":offersize_mu, "Sigma":offersize_covar,"Transition":offersize_transit}

            bookimbalance=dt_features['OB_IB'].reshape(-1, 1)
            bookimbalance_model=GaussianHMM(n_components=2,algorithm='viterbi',covariance_type="spherical",min_covar=1e-4, n_iter=1000,tol=1e-8)
            fitted_bookimbalance_model=bookimbalance_model.fit(bookimbalance)
            bookimbalance_mu=fitted_bookimbalance_model.means_.flatten()
            bookimbalance_covar=fitted_bookimbalance_model.covars_.flatten()
            bookimbalance_transit=fitted_bookimbalance_model.transmat_.flatten()
            bookimbalance_params[dt]={"Mean":bookimbalance_mu, "Sigma":bookimbalance_covar,"Transition":bookimbalance_transit}


    else:
        print("Not a valid method")
        return

    features_labels=["spread","bidsize","offersize","bookimbalance"]
    spread_df=extract_params(spread_params,method)
    bidsize_df=extract_params(bidsize_params,method)
    offersize_df=extract_params(offersize_params,method)
    bookimbalance_df=extract_params(bookimbalance_params,method)

    dict_df=dict(zip(features_labels,[spread_df,bidsize_df,offersize_df,bookimbalance_df]))
    return dict_df