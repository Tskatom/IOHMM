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
        n_components = 2
        startprob = np.array([0.2, 0.8])
        ins = np.array([[1, 2], [1, 4], [2, 5], [3, 8]])
        self.hmm = _BaseIOHMM(n_components, [ins], startprob=startprob)

    def tearDown(self):
        del self.hmm

    def test_compute_transmat(self):
        trans_prob = self.hmm._compute_transmat(self.hmm.ins[0])
        expected = np.tile([0.5, 0.5], (len(self.hmm.ins[0]), self.hmm.n_components, 1))

        numpy.testing.assert_array_equal(expected, trans_prob)

    def test_softmax(self):
        alphas = np.array([0, 0, 0])
        result = extmath.softmax(alphas)
        numpy.testing.assert_array_equal(np.tile(1/3,(1,3)), result[np.newaxis])

    def test_poisson_likelihood(self):
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

    def test__compute_obs_prob(self):
        ins = self.hmm.ins
        obs = np.array([1, 1, 3, 4])
        probs = self.hmm._compute_obs_prob(ins[0], obs)

    def test_fit(self):
        obs = np.array([1, 3, 2, 2])
        self.hmm.fit([obs])



if __name__ == "__main__":
    unittest.main()
