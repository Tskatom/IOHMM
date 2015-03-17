"""
Created on Feb 13 2015
@author Wei Wang

Implement the base class for IOHMM
The implementation is based on:
    - Input-Output HMM's for Sequence processing
    - scikit-learn HMM implementation
"""
from __future__ import division
__author__ = 'Wei Wang'
__email__ = 'tskatom@vt.edu'

from functools import partial
import numpy as np
from util import extmath
from scipy.optimize import minimize
from scipy.sparse import dia_matrix
import sys
from poisson_regression import PoissonRegression

EPS = np.finfo(float).eps


class _BaseIOHMM():
    """
    Input output Hidden Markov model base class
    Representation of a IO-HMM model probability distribution.
    This class allows for sampling from, evaluation and maximum-likelihood
    estimation of the parameters of a HMM

    Attributes
    ----------
    n_components : int
        Numbers of the states in the model

    ins : list
        list of array-like observation sequences, each of which has shape
                (n_i, n_features), which n_i is the length of the i_th observation

    startprob : array, shape ('n_components')
        Initiate state occupation distribution
        
    algorithm : string, one of the decoder_algorithms
        Decoder Algorithms

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform

    thresh : float, optional
        Convergence threshold

    params: string, optional
        Controls which parameters are updated in the training process.

    init_params: string, optional
        Controls which parameters are initiated prior to training

    Interfaces
    ----------
    This class implements the public interface for all IOHMMs that derived from it,
    including all of the machinery for the forward-backward Viterbi algorithm. Subclass
    only need to implement _generate_sample_from_state(), _compute_likelihood(),
    _init(), _initiatelize_sufficient_statistics() and _do_mstep()
    """

    def __init__(self, n_components, ins, obs, start_prob=None, algorithm="viterbi",
                 random_state=None, n_iter=70, thresh=1e-2):

        self.n_components = n_components
        self.ins = ins
        self.obs = obs
        self.n_iter = n_iter
        self.thresh = thresh
        self.start_prob = start_prob
        self.algorithm = algorithm
        self.random_state = random_state
        self.gamma = .5

        # add dummy parameter to the input matrix
        for p in range(len(self.ins)):
            # normalize the input data
            x = np.matrix(self.ins[p])
            mean_x = np.mean(x, axis=0)
            std_x = np.std(x, axis=0 )
            self.ins[p] = (x - mean_x)/std_x

            dummy = np.matrix(np.ones(len(self.ins[p]))).T
            self.ins[p] = np.hstack([dummy, self.ins[p]])
            self.obs[p] = np.matrix(self.obs[p]).T

        self.start_prob = np.matrix(start_prob).T

        self.input_dim = self.ins[0].shape[1]  # the dimension of input

        # construct the transition weighted matrix
        self.trans_weight_mat = np.random.random(self.n_components * self.n_components * self.input_dim)
        self.trans_weight_mat = self.trans_weight_mat.reshape(self.n_components, self.n_components, self.input_dim)
        #self.trans_weight_mat = np.ones((self.n_components, self.n_components, self.input_dim))
        # construct the weight matrix for poisson regression
        # Fitted a poisson regression model to get initiate obs parameters

        poisson_model = PoissonRegression(self.ins[0], self.obs[0])
        poisson_model.fit()

        self.obs_weight_mat = np.zeros((self.n_components, self.input_dim))
        for i in range(self.n_components):
            self.obs_weight_mat[i,:] = np.squeeze(np.array(poisson_model.beta))


    def fit(self):
        """ Estimate the model parameters

        An initialization step is performed before enter te EM algorithm
        :param obs:
            obs : list
                list of array-like observation sequences, each of which has shape
                (n_i, n_features), which n_i is the length of the i_th observation

        """
        obs = self.obs
        ins = self.ins
        log_prob = []
        for n in range(self.n_iter):
            # Expectation step
            for i in range(len(obs)):
                obs_seq = obs[i]
                ins_seq = ins[i]

                log_trans_mat = self._compute_log_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
                log_frame_prob = self._compute_log_obs_prob(ins_seq, obs_seq)  # compute p(y|U, x_t=i) with shape (t, n)

                log_likelihood, fwd_lattice = self._do_forward_pass(log_trans_mat, log_frame_prob)
                bwd_lattice = self._do_backward_pass(log_trans_mat, log_frame_prob)

                # compute the sufficient statistic: transition posterior and state posterior
                self._compute_sufficient_static(log_trans_mat, log_frame_prob,
                                                fwd_lattice, bwd_lattice, log_likelihood)
            log_prob.append(log_likelihood)
            print n,'--------------log_prob---------,', log_prob[n]
            if n > 1 and abs(log_prob[-1] - log_prob[-2]) < self.thresh:
                print 'Converged'
                break

            # Maximization step
            self._do_maxstep()
        print "Fitted Results:  "
        print "trans_weight_mat", self.trans_weight_mat
        print "obs_weight_mat", self.obs_weight_mat

    def optimize_trans_beta4(self, ins_seq, obs_seq, j, n_iter, threshold=1e-6):
        trans_theta = np.matrix(self.trans_weight_mat[j])
        trans_post = np.matrix(self.trans_posts[:, j, :])
        X = ins_seq
        Y = obs_seq

        def obj_func(thetas):
            thetas_mat = np.matrix(thetas).reshape(self.n_components, self.input_dim)
            T = len(X)
            obj = 0.0
            for t in range(T):
                u = X[t].T
                e_js = thetas_mat * u
                for i in range(self.n_components):
                    #print "e_js[i]", e_js[i], "extmath.logsumexp(e_js)", extmath.logsumexp(e_js), "trans_post[t][i]", trans_post[t, i]
                    obj += trans_post[t, i] * (e_js[i] - extmath.logsumexp(e_js))
            print float(np.squeeze(np.array(obj)))
            return -1 * float(np.squeeze(np.array(obj)))

        def jac_func(thetas):
            thetas_mat = np.matrix(thetas).reshape(self.n_components, self.input_dim)
            jac_array = np.zeros((self.n_components, self.input_dim))
            I = np.matrix(np.identity(self.n_components))
            nu = X * thetas_mat.T
            prob_mu = extmath.safe_softmax(nu)
            for s in range(self.n_components):
                I_s = I[:, s]
                prob_mu_s = np.squeeze(np.array(prob_mu[:, s]))
                jac_s = np.squeeze(np.array(X.T * trans_post * I_s)) - \
                        np.squeeze(np.array(X.T *
                                            dia_matrix((prob_mu_s, 0), shape=(len(prob_mu_s), len(prob_mu_s))) *
                                            np.sum(trans_post, axis=1)))

                jac_array[s, :] = jac_s
            jac_vec = jac_array.reshape(self.input_dim * self.n_components, 1)
            return -1 * jac_vec

        def hess_func(thetas):
            hess_array = np.zeros((self.input_dim * self.n_components, self.input_dim * self.n_components))
            thetas_mat = np.matrix(thetas).reshape(self.n_components, self.input_dim)
            I = np.matrix(np.identity(self.n_components))
            nu = X * thetas_mat.T
            prob_mu = extmath.safe_softmax(nu)
            for s in range(self.n_components):
                sum_trans_post = np.squeeze(np.array(np.sum(trans_post, axis=1)))
                for p in range(self.n_components):
                    I_sp = I[s, p]
                    prob_s = prob_mu[:, s]
                    prob_p = prob_mu[:, p]
                    prob_item = np.squeeze(np.array(np.multiply(prob_s, prob_p) - I_sp * prob_s))
                    hess_item = X.T * dia_matrix((prob_item, 0), shape=(len(prob_item), len(prob_item))) * dia_matrix((sum_trans_post, 0), shape=(len(sum_trans_post), len(sum_trans_post))) * X
                    hess_array[(s * self.input_dim):((s + 1) * self.input_dim), (p * self.input_dim):((p + 1) * self.input_dim)] = np.array(hess_item)

            return -1 * hess_array

        ini_theta = self.trans_weight_mat[j].flatten()

        res = minimize(obj_func, ini_theta, method='Newton-CG',jac=jac_func, hess=hess_func,options={'disp': True})
        print res
        self.trans_weight_mat[j] = res.x.reshape(self.n_components, self.input_dim)
        #self.trans_weight_mat[j, :, :] = np.array(trans_theta)

    def optimize_trans_beta3(self, ins_seq, obs_seq, j, n_iter, threshold=1e-3):
        trans_theta = np.matrix(self.trans_weight_mat[j, :, :])
        trans_post = np.matrix(self.trans_posts[:, j, :])
        difference = []
        T = len(ins_seq)
        X = ins_seq
        for n in range(n_iter):
            consize_jac_array = np.zeros((self.n_components, self.input_dim))
            I = np.matrix(np.identity(self.n_components))
            nu = X * trans_theta.T
            prob_mu = extmath.safe_softmax(nu)
            for s in range(self.n_components):
                I_s = I[:, s]
                prob_mu_s = np.squeeze(np.array(prob_mu[:, s]))
                jac_s = np.squeeze(np.array(X.T * trans_post * I_s)) - \
                        np.squeeze(np.array(X.T *
                                            dia_matrix((prob_mu_s, 0), shape=(len(prob_mu_s), len(prob_mu_s))) *
                                            np.sum(trans_post, axis=1)))

                consize_jac_array[s, :] = jac_s
            print "consize_jac_array", consize_jac_array

            jac_array = np.zeros((self.n_components, self.input_dim))
            I = np.identity(self.n_components)
            for s in range(self.n_components):
                jac_s = np.zeros(self.input_dim)
                prob_tmps_s = []
                for t in range(T):
                    u = ins_seq[t]
                    work_buffer = .0
                    prob_s = np.exp(u * trans_theta[s].T) / np.sum(np.exp(u * trans_theta.T))
                    prob_tmps_s.append(prob_s)
                    for i in range(self.n_components):
                        work_buffer += trans_post[t, i] * (I[s, i] - prob_s)
                    jac_s += np.squeeze(np.asarray(u)) * float(work_buffer)
                jac_array[s, :] = jac_s
            print "jac_array", jac_array

            jac_vec = np.matrix(jac_array).flatten().T

            consize_hess_array = np.zeros((self.input_dim * self.n_components, self.input_dim * self.n_components))

            for s in range(self.n_components):
                sum_trans_post = np.squeeze(np.array(np.sum(trans_post, axis=1)))
                for p in range(self.n_components):
                    I_sp = I[s, p]
                    prob_s = prob_mu[:, s]
                    prob_p = prob_mu[:, p]
                    prob_item = np.squeeze(np.array(np.multiply(prob_s, prob_p) - I_sp * prob_s))
                    hess_item = X.T * dia_matrix((prob_item, 0), shape=(len(prob_item), len(prob_item))) * dia_matrix((sum_trans_post, 0), shape=(len(sum_trans_post), len(sum_trans_post))) * X
                    consize_hess_array[(s * self.input_dim):((s + 1) * self.input_dim), (p * self.input_dim):((p + 1) * self.input_dim)] = np.array(hess_item)

            print "consize_hess_array", consize_hess_array

            hess_array = np.zeros((self.n_components * self.input_dim, self.n_components * self.input_dim))
            for s in range(self.n_components):
                for p in range(self.n_components):
                    hess_item = np.zeros((self.input_dim, self.input_dim))
                    for t in range(T):
                        u = ins_seq[t]
                        prob_s = np.exp(u * trans_theta[s].T) / np.sum(np.exp(u * trans_theta.T))
                        prob_p = np.exp(u * trans_theta[p].T) / np.sum(np.exp(u * trans_theta.T))
                        work_buffer = .0
                        for i in range(self.n_components):
                            work_buffer += trans_post[t, i] * prob_s * (prob_p - I[s, p])
                        work_buffer = float(work_buffer)
                        hess_item += np.array(u.T * u * work_buffer)
                    hess_array[self.input_dim * (s) : self.input_dim * (s + 1), self.input_dim * (p) : self.input_dim * (p + 1)] = hess_item

            print "hess_array", hess_array
            trans_theta_old = trans_theta
            consize_trans_theta_old = trans_theta
            trans_theta = trans_theta - (np.matrix(np.linalg.pinv(hess_array)) * jac_vec).reshape(self.n_components, self.input_dim)
            difference.append(np.max(trans_theta_old - trans_theta))

            try:
                consize_trans_theta = consize_trans_theta_old - np.reshape(np.linalg.pinv(consize_hess_array) * np.matrix(consize_jac_array).flatten().T, (self.n_components, self.input_dim))
            except Exception as e:
                print 'Failed to Converge!'
                print 'jac_vec', jac_vec
                print hess_array
                sys.exit()
            #print "trans_theta_old ", trans_theta_old
            #print "trans_theta_new", trans_theta
            difference.append(np.max(np.abs(trans_theta_old - trans_theta)))
            print "consize_trans_theta", consize_trans_theta
            print "trans_theta", trans_theta
            sys.exit()

            if difference[-1] <= threshold:
                break
        self.trans_weight_mat[j, :, :] = np.array(trans_theta)

    def optimize_trans_beta2(self, ins_seq, obs_seq, j, n_iter, threshold=1e-3):
        trans_theta = np.matrix(self.trans_weight_mat[j, :, :])
        trans_post = np.matrix(self.trans_posts[:, j, :])
        difference = []
        T = len(ins_seq)
        for n in range(n_iter):
            jac_array = np.zeros((self.n_components, self.input_dim))
            I = np.identity(self.n_components)
            for s in range(self.n_components):
                jac_s = np.zeros(self.input_dim)
                for t in range(T):
                    u = ins_seq[t]
                    work_buffer = .0
                    prob_s = np.exp(u * trans_theta[s].T) / np.sum(np.exp(u * trans_theta.T))
                    for i in range(self.n_components):
                        work_buffer += trans_post[t, i] * (I[s, i] - prob_s)
                    jac_s += np.squeeze(np.asarray(u)) * float(work_buffer)
                jac_array[s, :] = jac_s

            jac_vec = np.matrix(jac_array).flatten().T

            hess_array = np.zeros((self.n_components * self.input_dim, self.n_components * self.input_dim))
            for s in range(self.n_components):
                for p in range(self.n_components):
                    hess_item = np.zeros((self.input_dim, self.input_dim))
                    for t in range(T):
                        u = ins_seq[t]
                        prob_s = np.exp(u * trans_theta[s].T) / np.sum(np.exp(u * trans_theta.T))
                        prob_p = np.exp(u * trans_theta[p].T) / np.sum(np.exp(u * trans_theta.T))
                        work_buffer = .0
                        for i in range(self.n_components):
                            work_buffer += trans_post[t, i] * prob_s * (prob_p - I[s, p])
                        work_buffer = float(work_buffer)
                        hess_item += np.array(u.T * u * work_buffer)
                    hess_array[self.input_dim * (s) : self.input_dim * (s + 1), self.input_dim * (p) : self.input_dim * (p + 1)] = hess_item

            trans_theta_old = trans_theta
            trans_theta = trans_theta - (np.matrix(np.linalg.pinv(hess_array)) * jac_vec).reshape(self.n_components, self.input_dim)
            difference.append(np.max(trans_theta_old - trans_theta))
            if difference[-1] <= threshold:
                break
        self.trans_weight_mat[j, :, :] = np.array(trans_theta)

    def optimize_trans_beta(self, ins_seq, obs_seq, j, n_iter, threshold=1e-6):
        trans_theta = np.matrix(self.trans_weight_mat[j])
        trans_post = np.matrix(self.trans_posts[:, j, :])
        X = ins_seq
        Y = obs_seq
        difference = []

        for n in range(n_iter):
            #print n, "Before The innder trans_obj_cost", self.obj_trans_subnet(trans_theta, j)
            jac_array = np.zeros((self.n_components, self.input_dim))
            I = np.matrix(np.identity(self.n_components))
            nu = X * trans_theta.T
            prob_mu = extmath.safe_softmax(nu)
            for s in range(self.n_components):
                I_s = I[:, s]
                prob_mu_s = np.squeeze(np.array(prob_mu[:, s]))
                jac_s = np.squeeze(np.array(X.T * trans_post * I_s)) - \
                        np.squeeze(np.array(X.T *
                                            dia_matrix((prob_mu_s, 0), shape=(len(prob_mu_s), len(prob_mu_s))) *
                                            np.sum(trans_post, axis=1)))
                jac_s = jac_s - 2 * self.gamma * np.squeeze(np.array(trans_theta[s,:]))
                jac_array[s, :] = jac_s

                # check for the NAN in records
                if np.isnan(np.min(jac_s)):
                    print 'Encounter NAN', jac_s, n, s, jac_array
                    print 'Debug: '
                    print "trans_post: ", trans_post
                    print "prob_mu_s", prob_mu_s
                    print "I_s", I_s

                    sys.exit()
            jac_vec = np.matrix(jac_array.reshape(self.input_dim * self.n_components, 1))
            hess_array = np.zeros((self.input_dim * self.n_components, self.input_dim * self.n_components))

            for s in range(self.n_components):
                sum_trans_post = np.squeeze(np.array(np.sum(trans_post, axis=1)))
                for p in range(self.n_components):
                    I_sp = I[s, p]
                    prob_s = prob_mu[:, s]
                    prob_p = prob_mu[:, p]
                    prob_item = np.squeeze(np.array(np.multiply(prob_s, prob_p) - I_sp * prob_s))
                    hess_item = X.T * dia_matrix((prob_item, 0), shape=(len(prob_item), len(prob_item))) * dia_matrix((sum_trans_post, 0), shape=(len(sum_trans_post), len(sum_trans_post))) * X
                    hess_item = hess_item - 2 * self.gamma * I_sp * np.identity(self.input_dim)
                    hess_array[(s * self.input_dim):((s + 1) * self.input_dim), (p * self.input_dim):((p + 1) * self.input_dim)] = np.array(hess_item)

            hess_array = np.matrix(hess_array)
            trans_theta_old = trans_theta

            try:
                trans_theta = trans_theta - np.reshape(np.linalg.pinv(hess_array) * jac_vec, (self.n_components, self.input_dim))
            except Exception as e:
                print 'Failed to Converge!'
                print 'jac_vec', jac_vec
                print hess_array
                sys.exit()
            #print n, "After The innder trans_obj_cost", self.obj_trans_subnet(trans_theta, j)
            #print "trans_theta_old ", trans_theta_old
            #print "trans_theta_new", trans_theta
            difference.append(np.max(np.abs(trans_theta_old - trans_theta)))
            if difference[-1] <= threshold:
                break
        self.trans_weight_mat[j, :, :] = np.array(trans_theta)

    def optimize_obs_beta2(self, ins_seq, obs_seq, j, n_iter, threshold=1e-3):
        obs_beta = np.matrix(self.obs_weight_mat[j]).T
        difference = []
        state_post = self.state_posts[:, j]
        T = len(ins_seq)
        for n in range(n_iter):
            jac_vec = np.matrix(np.zeros(self.input_dim)).T
            hess = np.matrix(np.zeros((self.input_dim, self.input_dim)))
            for t in range(T):
                u = ins_seq[t].T
                o = float(obs_seq[t])
                work_buffer = float(state_post[t] * (o - np.exp(u.T * obs_beta)))
                jac_vec += work_buffer * u
                hess -= float(state_post[t]) * float(np.exp(u.T * obs_beta)) * u * u.T

            obs_beta_old = obs_beta
            obs_beta = obs_beta - np.matrix(np.linalg.pinv(hess)) * jac_vec
            difference.append(np.max(np.abs(obs_beta_old - obs_beta)))
            if difference[-1] <= threshold:
                break
        self.obs_weight_mat[j,:] = np.squeeze(np.array(obs_beta))

    def optimize_obs_beta(self, ins_seq, obs_seq, j, n_iter, threshold=1e-6):
        Y = obs_seq
        X = ins_seq

        obs_beta = np.matrix(self.obs_weight_mat[j]).T
        g = np.squeeze(self.state_posts[:, j])
        diag_g = dia_matrix(([g], 0), shape=(len(g), len(g)))
        difference = []

        log_g = np.matrix(self.log_state_posts[:, j]).T

        g_y = np.multiply(np.matrix(self.state_posts[:, j]).T, Y)

        for n in range(n_iter):
            #print n, "Before The innder obs_obj_cost", self.obj_obs_subnet(obs_beta, j)
            nu = X * obs_beta
            mu = np.exp(nu)
            w_data = np.squeeze(np.array(mu))
            W = dia_matrix(([w_data], 0), shape=(len(w_data), len(w_data)))
            grad = X.T * diag_g * (Y - mu) - 2 * self.gamma * obs_beta
            hess = -1 * X.T * diag_g * W * X - 2 * self.gamma * np.identity(self.input_dim)

            beta_old = obs_beta
            try:
                obs_beta = obs_beta - np.linalg.pinv(hess) * grad
            except Exception as e:
                print 'grad', n, grad, log_g[1:4], nu[1:4]
                sys.exit()
            #print n, "After The innder obs_obj_cost", self.obj_obs_subnet(obs_beta, j)
            difference.append(np.max(beta_old - obs_beta))
            if difference[-1] <= threshold:
                break
        self.obs_weight_mat[j, :] = np.squeeze(np.array(obs_beta))


    def _do_maxstep(self):
        # do maximization step in HMM. In base class we do M step to update the parameters for transition
        # weight matrix
        # Based on Yoshua Bengio, Paolo Frasconi. Input output HMM's for sequence processing

        # Maximize the observation subnetwork
        n_iter = 50
        for p in range(len(self.ins)):
            ins_seq = self.ins[p]
            obs_seq = self.obs[p]
            #compute the previous cost
            pre_obj = 0.0
            for j in range(self.n_components):
                pre_obj += self.obj_trans_subnet(np.matrix(self.trans_weight_mat[j, :, :]), j)
                pre_obj += self.obj_obs_subnet(np.matrix(self.obs_weight_mat[j,:]).T, j)
            print "pre_obj", pre_obj
            after_obj = 0.0
            for j in range(self.n_components):
                self.optimize_trans_beta(ins_seq, obs_seq, j, n_iter, threshold=1e-6)
                self.optimize_obs_beta(ins_seq, obs_seq, j, n_iter, threshold=1e-6)
                after_obj += self.obj_trans_subnet(np.matrix(self.trans_weight_mat[j, :, :]), j)
                after_obj += self.obj_obs_subnet(np.matrix(self.obs_weight_mat[j,:]).T, j)

            print "after_obj", after_obj
    def _compute_log_transmat(self, ins_seq):
        """ Compute the dynamic transition weight matrix for each time step
        phi_(ij,t) = p(x_t=i|x_{t-1}=j, u_t). In the weight matrix w[j, i] = p(x_t+1=i | x_t=j)
        """
        # initiate the dynamic transition matrix
        log_trans_mat = np.tile(0.0, (len(ins_seq), self.n_components, self.n_components))
        for t in range(len(ins_seq)):
            u = ins_seq[t].T # transform u into column vector
            for j in range(self.n_components):
                weight_mat = np.matrix(self.trans_weight_mat[j])
                alphas = np.squeeze(np.array(weight_mat * u))
                prob = alphas - extmath.logsumexp(alphas)
                log_trans_mat[t, j, :] = prob
        return log_trans_mat

    def _compute_log_obs_prob(self, ins_seq, obs_seq):
        """
        Compute the poisson regression probability
        """
        obs_weight_mat = np.matrix(self.obs_weight_mat)
        log_prob = np.zeros((len(ins_seq), self.n_components))
        for j in range(self.n_components):
            nu = ins_seq * obs_weight_mat[j].T
            lam = np.exp(nu)
            log_prob[:, j] = np.squeeze(np.array(extmath.log_poisson_likelihood(lam, obs_seq)))
        return log_prob

    def _compute_sufficient_static(self, log_trans_mat, log_frame_prob, log_fwd_lattice, log_bwd_lattice, log_likelihood):
        if np.isnan(np.min(log_fwd_lattice)) or np.isnan(np.min(log_bwd_lattice)):
            print log_fwd_lattice, log_bwd_lattice
            sys.exit()

        log_start_prob = np.log(self.start_prob)
        T = len(log_frame_prob)
        # compute the transition posterior
        trans_posts = np.tile(.0, (T, self.n_components, self.n_components))
        # Initiate the first step
        for j in range(self.n_components):
            for i in range(self.n_components):
                trans_posts[0][j][i] = np.exp(log_frame_prob[0, i] + log_start_prob[j, 0] + log_bwd_lattice[0, i] + log_trans_mat[0, j, i] - log_likelihood)

        for t in range(1, T):
            for j in range(self.n_components):
                for i in range(self.n_components):
                    trans_posts[t][j][i] = np.exp(log_frame_prob[t, i] + log_fwd_lattice[t-1, j] + log_bwd_lattice[t, i] + log_trans_mat[t, j, i] - log_likelihood)

        self.trans_posts = trans_posts
        # compute the state posterior
        state_posts = np.zeros((T, self.n_components))
        log_state_posts = np.zeros((T, self.n_components))
        for t in range(T):
            for i in range(self.n_components):
                state_posts[t][i] = np.exp(log_fwd_lattice[t, i] + log_bwd_lattice[t, i] - log_likelihood)
                log_state_posts[t][i] = log_fwd_lattice[t, i] + log_bwd_lattice[t, i] - log_likelihood
        self.state_posts = state_posts
        self.log_state_posts = log_state_posts

    def _do_forward_pass(self, log_trans_mat, log_frame_prob):
        """  Compute the forward lattice
        :param log_trans_mat:
        :param log_frame_prob:
        :return: p(obs_seq|ins_seq) and p(x_t=i, y_(1:t)|u_(1:t))
        """
        T = len(log_frame_prob)
        log_fwd_lattice = np.zeros((T, self.n_components))
        log_start_prob = np.log(self.start_prob)
        for i in range(self.n_components):
            work_buffer = np.zeros(self.n_components)
            for j in range(self.n_components):
                work_buffer[j] = log_start_prob[j, 0] + log_trans_mat[0][j][i]

            log_fwd_lattice[0][i] = log_frame_prob[0][i] + extmath.logsumexp(work_buffer)

        for t in range(1, T):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    work_buffer[j] = log_fwd_lattice[t - 1][j] + log_trans_mat[t][j][i]
                log_fwd_lattice[t][i] = log_frame_prob[t][i] + extmath.logsumexp(work_buffer)
        log_likelihood = extmath.logsumexp(log_fwd_lattice[-1, :])
        return log_likelihood, log_fwd_lattice

    def _do_backward_pass(self, log_trans_mat, log_frame_prob):
        # using the same scaling_factor as forward_pass
        T = len(log_frame_prob)
        log_bwd_lattice = np.ones((T, self.n_components))
        work_buffer = np.zeros(self.n_components)
        for i in range(self.n_components):
            log_bwd_lattice[T - 1][i] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    work_buffer[j] = log_bwd_lattice[t + 1, j] + log_frame_prob[t + 1][j] + log_trans_mat[t + 1][i][j]
                log_bwd_lattice[t][i] = extmath.logsumexp(work_buffer)
        return log_bwd_lattice

    def obj_trans_subnet(self, trans_theta, j):
        # maximize the subnetwork one by one
        # theta is a K * D matrix (K: status#, D: dim#)
        X = self.ins[0]
        obj = 0.0
        nu = X * trans_theta.T
        prob_mu = extmath.safe_softmax(nu)
        for p in range(len(self.ins)):
            ins_seq = self.ins[p]
            for t in range(len(ins_seq)):
                u = ins_seq[t].T
                for i in range(self.n_components):
                    obj += self.trans_posts[t, j, i] * np.log(prob_mu[t, i])
        return float(obj)

    def obj_obs_subnet(self, beta, j):
        # maximize the subnetwork one by one
        # theta is a D * 1 vector
        obj = 0.0
        for p in range(len(self.obs)):
            obs_seq = self.obs[p]
            ins_seq = self.ins[p]
            for t in range(len(obs_seq)):
                o = obs_seq[t]
                u = ins_seq[t].T
                obj += self.state_posts[t][j] * (-1 * np.exp(u.T * beta) + o * u.T * beta - extmath.logfac(o))
        return float(obj)