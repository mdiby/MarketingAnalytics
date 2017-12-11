# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:24:57 2017

@author: zhouyonglong
"""

## Attribution Modeling Increases Efficiency of Bidding in Display Advertising


import pandas as pd
import numpy as np

from scipy.optimize import minimize


DATA_FILE='E:/Datasets/Advertising/Criteo Attribution/criteo_attribution_small.csv'
#df = pd.read_csv(DATA_FILE, sep='\t', compression='gzip')
df = pd.read_csv(DATA_FILE, sep=',')


'''
def rand_rows(df, num_rows = 5):
    from numpy import random as rm
    subset = rm.choice(df.index.values, size = num_rows)    
    return df.loc[subset]

df_small = rand_rows(df,1000000)
df_small.to_csv('E:/Datasets/Advertising/Criteo Attribution/criteo_attribution_small.csv',
                index=False,encoding='utf-8')
'''



df['day'] = np.floor(df.timestamp / 86400.).astype(int)
df.day.hist(bins=len(df.day.unique()))


df['gap_click_sale'] = -1
df.loc[df.conversion == 1, 'gap_click_sale'] = df.conversion_timestamp - df.timestamp


FEATURES = ['campaign', 'cat1', 'cat2', 'cat3', 
            'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
INFOS = ['cost', 'cpo', 'time_since_last_click']


df['last_click'] = df.attribution * (df.click_pos == df.click_nb - 1).astype(int)
df['first_click'] = df.attribution * (df.click_pos == 0).astype(int)
df['all_clicks'] = df.attribution
df['uniform'] = df.attribution / (df.click_nb).astype(float)
INFOS += ['last_click', 'first_click', 'all_clicks', 'uniform']



from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import log_loss


def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    data = np.array(data)
    stats = []
    for _ in range(num_samples):
        idx = np.random.randint(0, n, n)
        samples = data[idx]
        stats += [statistic(samples)]
    stats = np.array(sorted(stats))
    return (stats[int((alpha/2.0)*num_samples)],
            stats[int((1-alpha/2.0)*num_samples)])

# negative log-likelihood
def attr_nllh(l,x,y):
    loss = 0.0
    lamb = l[0]
    n = x.shape[0]
    for i in range(n):
        if y[i] == 1:
            loss += lamb*x[i]
        else:
            loss -= np.log(1 - np.exp(-lamb*x[i]))  
    return loss/float(n)

# negative log-likelihood grad
def attr_nllh_grad(l,x,y):
    grad = 0.0
    lamb = l[0]
    n = x.shape[0]
    for i in range(n):
        grad += x[i]*y[i] / (1 - np.exp(-lamb*x[i]))
    return np.array([grad/float(n)])


def optimize_lambda(tts, attrib):
    return minimize(attr_nllh, 
                    1e-3, 
                    method='L-BFGS-B', 
                    jac=attr_nllh_grad, 
                    options={'disp': True, 'maxiter': 20 }, 
                    bounds=((1e-15, 1e-4),), 
                    args=(tts,attrib)).x[0]


def optimize_lambda_sgd(tts, attrib):
    return 0


def learn_attribution_model(df_view, 
                            test_day, 
                            learning_duration, 
                            verbose=False, 
                            ci=False, 
                            rescale=1., 
                            optimizer=optimize_lambda):
    df_train = df_view[(df_view.day >= test_day - learning_duration) & (df_view.day < test_day)]
    df_conv = df_train[df_train.click_pos == df_train.click_nb - 1]
    
    x = df_conv.gap_click_sale.values
    y = df_conv.attribution.values
    
    avg_tts = x.mean()
    tts_ci = bootstrap(x, 100, np.mean, .05)
    tts_ci = tts_ci[1] - tts_ci[0]

    lamb = optimize_lambda(x, y)
    
    lambs = []
    n_bootstraps = 30
    alpha=.05
    if ci:
        for _ in range(n_bootstraps):
            idx = np.random.randint(0, x.shape[0], x.shape)
            xx = x[idx]
            yy = y[idx]
            lambs += [optimize_lambda(xx, yy)]

    if verbose:
        print('\t\t-avg_tts', avg_tts, '+/-', tts_ci, 
              ' = ', avg_tts / 3600., 'hours = ', avg_tts / 86400., 'days')
        if ci:
            print('\t\t-lambda', lamb, '+/-', (lambs[int((1-alpha/2.)*n_bootstraps)] - lambs[int((alpha/2.)*n_bootstraps)]))
        else:
            print('\t\t-lambda', lamb)
    
    return avg_tts, lamb


#global_avg_tts, global_lamb =  learn_attribution_model(df, 21, 20)


def compute_aa_attributions(test_info, normalize=True):
    test_info['idx'] = test_info.index
    converted =  test_info[test_info.all_clicks==1]
    
    # to propoerly compute  normalized attribution factors, 
    # we have to reconstruct the timelines for each conversion
    # conversion_ids = converted['conversion_id'].unique()
    # now reconstruct timeline and apply attribution
    by_conversion = converted[['conversion_id', 'timestamp', 'idx', 
                               'bf_pred', 'time_since_last_click', 
                               'last_click']].groupby('conversion_id')
    new_clicks_data = []
    
    s_attr = []
    s_attr_lc = []
    
    # for each conversion compute attribution for each click
    for conv, evts in by_conversion:
        #按照时间排序
        sorted_clicks = sorted(evts.values.tolist(), key=lambda x: x[1])
        # 出价因子
        bf_pred = [_[3] for _ in sorted_clicks]
        sum_bf = np.sum(bf_pred)
        # last_click 数量
        sum_lc = np.sum([_[5] for _ in sorted_clicks])
        # 
        sum_attr = 0.0
        
        for pos, (_, _, idx_, bf_, tslc_, lc_) in enumerate(sorted_clicks):
            
            aa_attr = bf_pred[pos]
            
            if normalize:
                if sum_bf>0.0:
                    aa_attr = aa_attr / sum_bf
                else:
                    aa_attr = 0.0
            
            sum_attr += aa_attr
            new_clicks_data.append((idx_, aa_attr))
        
        s_attr.append(sum_attr)
        s_attr_lc.append(sum_lc)
    
    # now for each click, apply attribution from computed data
    new_clicks_df = pd.DataFrame(columns=['click_idx', 'aa_attribution'])
    cidx, attr = zip(*new_clicks_data)
    
    new_clicks_df['click_idx'] = cidx
    new_clicks_df['aa_attribution'] = attr
    new_clicks_df = new_clicks_df.set_index('click_idx')
    
    joined = test_info.join(new_clicks_df)
    joined['aa_attribution'] = joined['aa_attribution'].fillna(value = 0.0)
    
    return joined['aa_attribution']


#learn global attribution model
avg_tts, lamb = learn_attribution_model(df, 21, 20)
print(avg_tts)
print(lamb)

# compute the bid factor from aa attribution for each display
#出价公式中的第二项（每一次conversion里面的每次click都要计算一个factor）
gap_test = df.time_since_last_click.values
previous_tslc_mask = (df.time_since_last_click >=0).astype(float)
attr_pred = np.exp(-lamb*gap_test)
attr_pred *= previous_tslc_mask
bf_pred = 1 - attr_pred
df['bf_pred'] = bf_pred


#基于计算出来的bid因子计算一次转化内各个点击的贡献
df['AA_normed'] = compute_aa_attributions(df, normalize=True)

df['AA_not_normed'] = compute_aa_attributions(df, normalize=False)
INFOS += ['bf_pred', 'AA_normed', 'AA_not_normed']



from scipy.special import gammainc


def empirical_utility(a, v, c, p):
    won = np.array(p*v > c, dtype=np.int)
    return (a*v)*won, -c*won


def expected_utility(a, v, c, p, beta=1000):
    return a*v*gammainc(beta*c+1, beta*p*v) - \
            ((beta*c+1)/beta)*gammainc(beta*c+2, beta*p*v)


def evaluate_utility(y_pred, utilities, betas, test_info):
    partial_score = dict()
    for utility in utilities:
        attribution = test_info[utility]
        for beta in betas:
            if np.isinf(beta):
                est_utility = empirical_utility(attribution, test_info.cpo, test_info.cost, y_pred)
            else:
                est_utility = expected_utility(attribution, test_info.cpo, test_info.cost, y_pred, beta=beta)
            beta_str = str(beta) if not np.isinf(beta) else 'inf'
            partial_score['utility-'+utility+'-beta'+beta_str] = est_utility
    return partial_score



def get_naive_baseline(y_train, X_test):
    return np.mean(y_train)*np.ones(X_test.shape[0])



def evaluate_day_for_bidder(df_view, 
                            test_day, 
                            learning_duration, 
                            bidder, 
                            utilities,
                            betas,
                            hash_space=None,
                            features=None, 
                            clf=None,
                            AA_bidder_label=None, 
                            recalibrate=True):
    score = dict()
    bid_profile = dict()
    label = bidder
    if bidder == 'AA':
        label = AA_bidder_label
    # get data slice
    (X_train, y_train), (X_test, y_test), test_info, y_train_lc_mean = \
        get_train_test_slice(df_view,
                             test_day,
                             learning_duration,
                             label=label, 
                             hash_space = hash_space,
                             features=features)           
    # learn the model
    clf.fit(X_train, y_train)
    
    # get test predictions
    y_pred = clf.predict_proba(X_test)[:,1]            
    
    # if aa bidder: modulate the bids by bid_factor computed from attribution model
    # 如果是基于转化率提升的出价，则在每次计算后乘以出价因子
    if bidder == 'AA':
        y_pred *= test_info['bf_pred']
    
    # compute the loss
    loss = log_loss(y_test, y_pred, normalize=0)
    
    # loss of baseline model
    baseline_loss = log_loss(y_test, 
                             get_naive_baseline(y_train, X_test), 
                             normalize=0)
    score['nllh'] = loss
    score['nllh_naive'] = baseline_loss
    
    # do we recalibrate output? (i.e recalibrate mean prediction). 
    # This is usually done by a control system.
    if recalibrate:
        y_pred *= (y_train_lc_mean / y_pred.mean())
    
    #how many displays are won?
    won = (y_pred * test_info.cpo > test_info.cost).astype(int)
    
    score['won'] = np.sum(won)
    score['n_auctions'] = y_pred.shape[0]
    
    # compute the scores on this slice
    score.update(evaluate_utility(y_pred, utilities, betas, test_info))
    
    #store bid profiles
    bid_profile['time_since_last_click'] = test_info.time_since_last_click
    bid_profile['bid'] = y_pred
    
    return score, bid_profile



def get_train_test_slice(df_view, 
                         test_day, 
                         learning_duration, 
                         label, features=None, 
                         hash_space=2**24,
                         nrows=None, 
                         infos=None):
    df_test = df_view[df_view.day == test_day]
    if nrows is not None:
        df_test = df_test[:nrows]
    if features is None:
        features = FEATURES
    if infos is None:
        infos = INFOS
    df_train = df_view[(df_view.day >= test_day - learning_duration) & (df_view.day < test_day)]
    if nrows is not None:
        df_train = df_train[:nrows]
  
    X_train = df_train[features]
    X_test = df_test[features]
    
    hasher = FeatureHasher(n_features=hash_space, non_negative=1)
    
    def to_dict_values(df_view):
        return [dict([(_[0]+str(_[1]),1) for _ in zip(features,l)]) for l in df_view.values]
    
    X_train_h = hasher.fit_transform(to_dict_values(X_train))
    X_test_h = hasher.transform(to_dict_values(X_test))
    
    y_train = df_train[label]
    y_test = df_test[label]
    
    return (X_train_h, y_train), (X_test_h, y_test), df_test[infos], df_train.last_click.mean()



def merge_utility_score(score):
    updates = dict()
    for k,v in score.items():
        if not 'utility' in k:
            continue
        if 'inf' in k:
            revenue, cost = v
            updates[k] = np.sum(cost) + np.sum(revenue)
            updates[k+'~revenue'] = np.sum(revenue)
            updates[k+'~cost'] = np.sum(cost)
            v = revenue + cost
        else:
            updates[k] = np.sum(v)
        bounds = bootstrap(v, 100, np.sum, .05)
        delta = (bounds[1]-bounds[0])/2.
        updates[k+'-delta'] = delta
    score.update(updates)





from datetime import datetime
def evaluate_slices(df_view,
                    bidders=['last_click', 'first_click', 'AA'],
                    utilities=['last_click','first_click',
                               'AA_normed', 'AA_not_normed'],
                    betas=[np.inf, 10, 1000],
                    test_days=[22],
                    learning_duration=21,
                    hash_space=2**24,
                    features=None,
                    AA_bidder_label='all_clicks',
                    clf = LogisticRegression(solver='lbfgs', n_jobs=4),
                    recalibrate = True):
    bid_profiles = []
    scores = []
    
    for bidder in bidders:
        print ('*'*80)
        print("EVALUATING BIDDER:", bidder)
        score = dict()
        bid_profile = dict()
        for test_day in test_days:
            start = datetime.now()
            print('\t- day:', test_day)
            partial_score, partial_bid_profile = evaluate_day_for_bidder(
                df_view, 
                test_day, 
                learning_duration, 
                bidder, 
                utilities, 
                betas,
                hash_space=hash_space, 
                features=features, 
                clf=clf, 
                AA_bidder_label=AA_bidder_label, 
                recalibrate=recalibrate
            )
            update_score(partial_score, score)
            for k, v in partial_bid_profile.items():
                bid_profile[k] = np.append(bid_profile.get(k, np.array([])), v)
            print('\t- took', datetime.now() - start)
        score['bidder'] = bidder
        bid_profile['bidder'] = bidder
        score['nllh_comp_vn'] = \
            (score['nllh_naive'] - score['nllh']) / np.abs(score['nllh_naive'])
        score['win_rate'] = score['won'] / score['n_auctions']
        merge_utility_score(score)
        scores.append(score)
        bid_profiles.append(bid_profile)
    return pd.DataFrame(scores), pd.DataFrame(bid_profiles)



def update_score(partial_score, score):
    for k, v in partial_score.items():
        if 'utility' in k:
            if 'inf' in k:
                revenue, cost = v
                print('\t\t', k, np.sum(cost)+np.sum(revenue))
                current_revenue, current_cost = score.get(k, (np.array([]),np.array([])))
                score[k] = (
                    np.append(current_revenue, revenue),
                    np.append(current_cost, cost)
                )
            else:
                print('\t\t', k, np.sum(v))
                score[k] = np.append(score.get(k, np.array([])), v)
        else:
            print('\t\t', k, v)
            score[k] = score.get(k, 0) + v




full_run = False
#full run
if full_run:
    scores, bid_profiles = evaluate_slices(df,
                                       bidders=['last_click',
                                                'first_click',
                                                'AA'],
                                       utilities=['last_click',
                                                  'first_click',
                                                  'AA_normed',
                                                  'AA_not_normed'],
                                       test_days=range(22,29),
                                       learning_duration=21,
                                       hash_space = 2**18,
                                       AA_bidder_label='all_clicks')
else:
    scores, bid_profiles = evaluate_slices(df,
                                       bidders=['last_click',
                                                'AA'],
                                       utilities=['last_click',
                                                  'AA_normed'],
                                       test_days=range(22,23),
                                       learning_duration=5,
                                       hash_space = 2**13,
                                       AA_bidder_label='all_clicks')





