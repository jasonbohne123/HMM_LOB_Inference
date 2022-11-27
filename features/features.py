import pandas as pd 
from preprocessing import remove_duplicates

def prep_features(dt):
    """ Prep features by accessing grouped feature csv 
    """
    grouped_features=pd.read_csv(f'data/agg_features/grouped_features_{dt}.csv')
    bidsize=remove_duplicates(grouped_features['Bid_Size'].values)
    offersize=remove_duplicates(grouped_features['Offer_Size'].values)
    bookimbalance=remove_duplicates(grouped_features['OB_IB'].values)
    spread=remove_duplicates(grouped_features['spread'].values)

    feature_dict=dict(zip(['Bid_Size','Offer_Size','OB_IB','spread'],[bidsize,offersize,bookimbalance,spread]))

    return feature_dict