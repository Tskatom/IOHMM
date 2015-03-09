from __future__ import division
__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'


from base import _BaseIOHMM
import numpy as np
import unittest
from util import extmath
import numpy.testing


class TestIOHMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n_components = 3
        startprob = np.array([0.2, 0.2, 0.6])

        sd_noise = 0.25
        poisson_mean = 2.0
        sample_size = 20
        Y = np.random.poisson(poisson_mean,sample_size)
        X1 = 3 * Y + 2 + np.random. np.random.normal(0,sd_noise,sample_size)
        X2 = 7 * Y + 5 + np.random.normal(0,sd_noise,sample_size)
        X = np.matrix(np.vstack((X1,X2)))
        X = X.transpose()
        self.hmm = _BaseIOHMM(n_components, [X], [Y], startprob=startprob)

    def tearDown(self):
        del self.hmm

    def _test_compute_transmat(self):
        trans_prob = self.hmm._compute_transmat(self.hmm.ins[0])
        print trans_prob.shape

    def _test_obs_prob(self):
        obs_seq = self.hmm.obs[0]
        ins_seq = self.hmm.ins[0]
        frame_prob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        print frame_prob

    def _test_forward_lattice(self):
        ins_seq = self.hmm.ins[0]
        obs_seq = self.hmm.obs[0]
        transmat = self.hmm._compute_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
        frameprob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        lpr, fwdlattice = self.hmm._do_forward_pass(transmat, frameprob)
        print lpr, fwdlattice

    def _test_backward_lattice(self):
        ins_seq = self.hmm.ins[0]
        obs_seq = self.hmm.obs[0]
        transmat = self.hmm._compute_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
        frameprob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        lpr, fwdlattice = self.hmm._do_forward_pass(transmat, frameprob)
        bwdlattice = self.hmm._do_backward_pass(transmat, frameprob)
        print bwdlattice, lpr, fwdlattice

    def _test_softmax(self):
        alphas = np.array([0, 0, 0])
        result = extmath.softmax(alphas)
        numpy.testing.assert_array_equal(np.tile(1/3,(1,3)), result[np.newaxis])

    def _test_poisson_likelihood(self):
        lam = 1
        y = 0
        expected = np.exp(-1)
        result = extmath.poisson_likelihood(lam, y)
        self.assertEqual(expected, result)

        lam = np.array([1, 1])
        y = 0
        result = extmath.poisson_likelihood(lam, y)
        expected = np.array([np.exp(-1), np.exp(-1)])
        numpy.testing.assert_array_equal(result, expected)

    def _test_compute_obs_prob(self):
        ins = self.hmm.ins
        obs = np.array([1, 1, 3, 4])
        probs = self.hmm._compute_obs_prob(ins[0], obs)

    def _test_compute_sufficient_static(self):
        ins_seq = self.hmm.ins[0]
        obs_seq = self.hmm.obs[0]
        transmat = self.hmm._compute_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
        frameprob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        lpr, fwdlattice = self.hmm._do_forward_pass(transmat, frameprob)
        bwdlattice = self.hmm._do_backward_pass(transmat, frameprob)
        self.hmm._compute_sufficient_static(transmat, frameprob, fwdlattice, bwdlattice)
        print self.hmm.trans_posts
        print self.hmm.state_posts

    def _test_opt_obs_beta(self):
        ins_seq = self.hmm.ins[0]
        obs_seq = self.hmm.obs[0]
        transmat = self.hmm._compute_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
        frameprob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        lpr, fwdlattice = self.hmm._do_forward_pass(transmat, frameprob)
        bwdlattice = self.hmm._do_backward_pass(transmat, frameprob)
        self.hmm._compute_sufficient_static(transmat, frameprob, fwdlattice, bwdlattice)
        self.hmm.optimize_obs_beta(ins_seq, obs_seq, 0, 100)

    def _test_optimize_trans_beta(self):
        ins_seq = self.hmm.ins[0]
        obs_seq = self.hmm.obs[0]
        transmat = self.hmm._compute_transmat(ins_seq)  # compute dynamic transition matrix with shape (t, n, n)
        frameprob = self.hmm._compute_obs_prob(ins_seq, obs_seq)
        lpr, fwdlattice = self.hmm._do_forward_pass(transmat, frameprob)
        bwdlattice = self.hmm._do_backward_pass(transmat, frameprob)
        self.hmm._compute_sufficient_static(transmat, frameprob, fwdlattice, bwdlattice)
        self.hmm.optimize_trans_beta(ins_seq, obs_seq, 0, 100)

    def test_fit(self):
        self.hmm.fit()



if __name__ == "__main__":
    unittest.main()
