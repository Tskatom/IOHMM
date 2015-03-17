__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'
import pandas as pds
import numpy as np
import base
import cProfile
from util import extmath
np.set_printoptions(precision=4)

all_events_csv = "/home/weiwang/To_Jieping/icews_allevent_count/egypt_icews_parent_event_counts_2012_228.csv"
events_count = pds.DataFrame.from_csv(all_events_csv, sep='\t', index_col=1)
del events_count['20']
del events_count['country']
events_count = events_count.sort_index()['2012-01-01':]
events_count = events_count.resample('W', how='sum').fillna(0)
columns = events_count.columns
target = "14"
excepts = ["14"]
features = [col for col in columns if col not in excepts]

# construct the training and test set
Ys = events_count["18"]
Xs = events_count[['12','13','14','16', '17']]

trainX = Xs['2012-01-08':'2014-12-28'].values
trainY = Ys['2012-01-15':'2015-01-04'].values


# add dummy parameter to the input matrix

normX = np.matrix(trainX)
mean_x = np.mean(normX, axis=0)
std_x = np.std(normX, axis=0)
normX = (normX - mean_x)/std_x

dummy = np.matrix(np.ones(len(normX))).T
normX = np.hstack([dummy, normX])
normY = np.matrix(trainY).T

n_components = 4
input_dim = 6

start_prob = np.array([0.25, 0.25, 0.25,0.25])
np.random.seed(20)
trans_weight_mat = np.random.random(n_components * n_components * input_dim)
trans_weight_mat = trans_weight_mat.reshape(n_components, n_components, input_dim)

obs_weight_mat = np.random.random(n_components * input_dim)
obs_weight_mat = np.matrix(obs_weight_mat.reshape(n_components, input_dim))


T = len(trainX)
trans_prob = np.zeros((T+1, n_components))
trans_prob[0,:] = start_prob
for t in range(1, T+1):
    # generate the step t probability
    u = normX[t-1, :]
    for i in range(n_components):
        work_buffer = 0.0
        for j in range(n_components):
            tw_mat = np.matrix(trans_weight_mat[j, :, :])
            cond_p_ij = np.exp(u * tw_mat[i, :].T) / np.sum(np.exp(u * tw_mat.T))
            work_buffer += trans_prob[t-1, j] * cond_p_ij
        trans_prob[t, i] = float(work_buffer)

trans_prob = trans_prob[1:,:]

# Generate the observations
synthetic_obs = []
for t in range(T):
    cum_p = np.cumsum(trans_prob[t])
    p = np.random.uniform()
    target_status = np.argmax(cum_p > p)
    u = normX[t]
    nu = float(u * obs_weight_mat[target_status].T)
    lam = np.exp(nu)
    obs = np.random.poisson(lam)
    synthetic_obs.append(obs)

print synthetic_obs

"""
Generate the synthetic observation data
"""
new_synthetic_obs = []
last_status = []
# generate the first hidden status
cum_start_prob = np.cumsum(start_prob)
p = np.random.uniform()
pre_status = np.argmax(cum_start_prob >= p)
u = normX[0]
tw_mat = np.matrix(trans_weight_mat[pre_status, :, :])
curr_probs = np.squeeze(np.array(extmath.safe_softmax(u * tw_mat.T)))
cum_curr_probs = np.cumsum(curr_probs)
p = np.random.uniform()
print 'p',p
curr_status = np.argmax(cum_curr_probs >= p)
nu = float(u * obs_weight_mat[target_status].T)
lam = np.exp(nu)
obs = np.random.poisson(lam)
new_synthetic_obs.append(obs)
last_status.append(curr_status)
for t in range(1, T):
    pre_status = last_status[t-1]
    u = normX[t]
    tw_mat = np.matrix(trans_weight_mat[pre_status, :, :])
    curr_probs = np.squeeze(np.array(extmath.safe_softmax(u * tw_mat.T)))
    cum_curr_probs = np.cumsum(curr_probs)
    p = np.random.uniform()
    curr_status = np.argmax(cum_curr_probs >= p)
    #print 'pre_status',pre_status,'curr_status', curr_status, 'curr_probs', curr_probs, 'p',p, curr_probs >= p
    nu = float(u * obs_weight_mat[curr_status].T)
    lam = np.exp(nu)
    obs = min(np.random.poisson(lam), 300)
    new_synthetic_obs.append(obs)
    last_status.append(curr_status)

print new_synthetic_obs

"""
Use the synthetic observation
"""

hmm = base._BaseIOHMM(n_components, [trainX], [np.array(new_synthetic_obs)], start_prob=start_prob)
#cProfile.run("hmm.fit()")
hmm.fit()
print "Original_Trans_Weight_Mat ", trans_weight_mat
print "Original_Obs_Weight_Mat ", obs_weight_mat
print "Fitted obs weight mat", hmm.obs_weight_mat
print "Fitted trans weight mat",((hmm.trans_weight_mat))

T = len(trainX)
fitted_trans_prob = np.zeros((T+1, n_components))
fitted_trans_prob[0,:] = start_prob
for t in range(1, T+1):
    # generate the step t probability
    u = normX[t-1, :]
    for i in range(n_components):
        work_buffer = 0.0
        for j in range(n_components):
            tw_mat = np.matrix(hmm.trans_weight_mat[j, :, :])
            _max = np.max(u * tw_mat.T)
            cond_p_ij = np.exp(u * tw_mat[i, :].T - _max) / np.sum(np.exp(u * tw_mat.T - _max))
            work_buffer += fitted_trans_prob[t-1, j] * cond_p_ij
        fitted_trans_prob[t, i] = float(work_buffer)
fitted_trans_prob = fitted_trans_prob[1:,:]
#fitted_trans_prob = fitted_trans_prob[1:,:]

"""
Generate the synthetic observation data
"""
final_synthetic_obs = []
last_status = []
# generate the first hidden status
cum_start_prob = np.cumsum(start_prob)
p = np.random.uniform()
pre_status = np.argmax(cum_start_prob >= p)
u = normX[0]
tw_mat = np.matrix(hmm.trans_weight_mat[pre_status, :, :])
curr_probs = np.squeeze(np.array(extmath.safe_softmax(u * tw_mat.T)))
cum_curr_probs = np.cumsum(curr_probs)
p = np.random.uniform()
curr_status = np.argmax(cum_curr_probs >= p)
nu = float(u * np.matrix(hmm.obs_weight_mat)[target_status].T)
lam = np.exp(nu)
obs = np.random.poisson(lam)
final_synthetic_obs.append(obs)
last_status.append(curr_status)
for t in range(1, T):
    pre_status = last_status[t-1]
    u = normX[t]
    tw_mat = np.matrix(hmm.trans_weight_mat[pre_status, :, :])
    curr_probs = np.squeeze(np.array(extmath.safe_softmax(u * tw_mat.T)))
    cum_curr_probs = np.cumsum(curr_probs)
    p = np.random.uniform()
    curr_status = np.argmax(cum_curr_probs >= p)
    #print 'pre_status',pre_status,'curr_status', curr_status, 'curr_probs', curr_probs, 'p',p, curr_probs >= p
    nu = float(u * np.matrix(hmm.obs_weight_mat)[curr_status].T)
    lam = np.exp(nu)
    obs = int(np.floor(lam))
    final_synthetic_obs.append(obs)
    last_status.append(curr_status)

print 'predict:', final_synthetic_obs
print 'Observed:', new_synthetic_obs

def eveluation(pred, truth):
    occ_score = 0.5 * ((pred > 0) == (truth > 0))
    accu_score = 3.5 * (1 - 1.0*abs(pred-truth)/max(pred, truth, 4))
    return occ_score + accu_score

scores = map(eveluation, final_synthetic_obs, new_synthetic_obs)
print "Scores,", scores
print "mean scores", np.mean(scores)