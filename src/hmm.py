import numpy as np
import pickle
from tqdm import tqdm
from math import inf


class HMM(object):
    def __init__(self, N, M, emission, lamb=[]):
        # Size of the hidden state, i.e. hanzi number.
        self.N = N
        # Size of the obervable state, i.e. pinyin number.
        self.M = M
        self.start_prob = np.zeros((self.N)) + 0.01
        self.emission = emission
        self.bitrans_mat = {}
        self.lamb = lamb

    def fit(self, X):
        # MLE
        pass

    def predict(self, O):
        return list(map(self._viterbi, tqdm(O)))

    def save(self, model_dir):
        with open(model_dir, "wb") as fout:
            pickle.dump(self, fout)

    def load(self, model_dir):
        with open(model_dir, 'rb') as fin:
            self.__dict__.update(pickle.load(fin).__dict__)

    def _count_bi(self, i, j, delta):
        if (i, j) not in self.bitrans_mat:
            self.bitrans_mat[(i, j)] = delta
        else:
            self.bitrans_mat[(i, j)] += delta

    def _viterbi(self, o):
        pass

    def set_lamb(self, lamb):
        self.lamb = lamb


class BigramHMM(HMM):
    def __init__(self, N, M, emission, lamb=[0.01, 0.99]):
        super(BigramHMM, self).__init__(N, M, emission, lamb)

    def fit(self, X):
        for x in tqdm(X):
            n = len(x)
            for i in range(n):
                self.start_prob[x[i]] += 1
        for x in tqdm(X):
            n = len(x)
            for i in range(n - 1):
                self._count_bi(x[i], x[i + 1], 1 / self.start_prob[x[i]])
        self.start_prob /= self.start_prob.sum()

    def _viterbi(self, o):
        """
        Viterbi algorithm
        Retern the MAP estimation of the state trajectory of HMM.

        Parameters
        ------------------
        o: Observation state sequence.

        Returns
        ------------------
        x: The MAP estimation of the state trajectory.
        """
        T = len(o)
        PI, S, BP = [{-1: 1}], [[-1]], []
        for i in range(T):
            S.append(self.emission[o[i]])
            PI.append({})
            BP.append({})
            for v in S[i + 1]:
                max_u = None
                max_p = -inf
                for u in S[i]:
                    p = self._trans(u, v) * PI[i][u]
                    if p > max_p:
                        max_u = u
                        max_p = p
                PI[i + 1][v] = max_p
                BP[i][v] = max_u
        if T == 0:
            return []
        x = [0] * T
        x[T - 1] = max(PI[T], key=PI[T].get)
        for i in reversed(range(0, T - 1)):
            x[i] = BP[i + 1][x[i + 1]]
        return x

    def _trans(self, i, j):
        if i == -1:
            return self.start_prob[j]
        return self.bitrans_mat.get(
            (i, j), 0) * self.lamb[1] + self.start_prob[j] * self.lamb[0]


class TrigramHMM(HMM):
    def __init__(self, N, M, emission, lamb=[1e-4, 1e-2, 1]):
        super(TrigramHMM, self).__init__(N, M, emission, lamb)
        self.tritrans_mat = {}

    def fit(self, X):
        for x in tqdm(X):
            for i in x:
                self.start_prob[i] += 1

        for x in tqdm(X):
            n = len(x)
            for i in range(n - 1):
                self._count_bi(x[i], x[i + 1], 1 / self.start_prob[x[i]])

        for x in tqdm(X):
            n = len(x)
            for i in range(n - 2):
                self._count_tri(
                    x[i], x[i + 1], x[i + 2],
                    1 / (self.start_prob[x[i]] *
                         self.bitrans_mat[(x[i], x[i + 1])]))

        self.start_prob /= self.start_prob.sum()

    def _viterbi(self, o):
        """
        Viterbi algorithm
        Retern the MAP estimation of the state trajectory of HMM.

        Parameters
        ------------------
        o: Observation state sequence.

        Returns
        ------------------
        x: The MAP estimation of the state trajectory.
        """
        T = len(o)
        PI, S, BP = [{(-1, -1): 1}], [[-1], [-1]], []
        for i in range(T):
            S.append(self.emission[o[i]])
            PI.append({})
            BP.append({})
            for u in S[i + 1]:
                for v in S[i + 2]:
                    max_w = None
                    max_p = -inf
                    for w in S[i]:
                        p = self._trans(w, u, v) * PI[i][(w, u)]
                        if p > max_p:
                            max_w = w
                            max_p = p
                    PI[i + 1][(u, v)] = max_p
                    BP[i][(u, v)] = max_w
        x = [0] * T
        if T < 2:
            return list(max(PI[T], key=PI[T].get))[2 - T:]
        (x[T - 2], x[T - 1]) = max(PI[T], key=PI[T].get)
        for i in reversed(range(0, T - 2)):
            x[i] = BP[i + 2][x[i + 1], x[i + 2]]
        return x

    def _count_tri(self, i, j, k, delta):
        if (i, j, k) not in self.tritrans_mat:
            self.tritrans_mat[(i, j, k)] = delta
        else:
            self.tritrans_mat[(i, j, k)] += delta

    def _trans(self, i, j, k):

        if j == -1:
            return self.start_prob[k]
        if i == -1:
            return self.bitrans_mat.get(
                (j, k), 0) * self.lamb[1] + self.start_prob[k] * self.lamb[0]
        return self.tritrans_mat.get(
            (i, j, k), 0) * self.lamb[2] + self.bitrans_mat.get(
                (j, k), 0) * self.lamb[1] + self.start_prob[k] * self.lamb[0]


class QuadgramHMM(HMM):
    def __init__(self, N, M, emission, lamb=[1e-6, 1e-4, 1e-2, 1]):
        super(QuadgramHMM, self).__init__(N, M, emission, lamb)
        self.tritrans_mat = {}
        self.quadtrans_mat = {}

    def fit(self, X):
        for x in tqdm(X):
            for i in x:
                self.start_prob[i] += 1

        for x in tqdm(X):
            n = len(x)
            for i in range(n - 1):
                self._count_bi(x[i], x[i + 1], 1 / self.start_prob[x[i]])

        for x in tqdm(X):
            n = len(x)
            for i in range(n - 2):
                self._count_tri(
                    x[i], x[i + 1], x[i + 2],
                    1 / (self.start_prob[x[i]] *
                         self.bitrans_mat[(x[i], x[i + 1])]))

        for x in tqdm(X):
            n = len(x)
            for i in range(n - 3):
                self._count_quad(
                    x[i], x[i + 1], x[i + 2], x[i + 3], 1 /
                    (self.start_prob[x[i]] * self.bitrans_mat[(x[i], x[i + 1])]
                     * self.tritrans_mat[(x[i], x[i + 1], x[i + 2])]))

        self.start_prob /= self.start_prob.sum()

    def _viterbi(self, o):
        """
        Viterbi algorithm
        Retern the MAP estimation of the state trajectory of HMM.

        Parameters
        ------------------
        o: Observation state sequence.

        Returns
        ------------------
        x: The MAP estimation of the state trajectory.
        """
        T = len(o)
        PI, S, BP = [{(-1, -1, -1): 1}], [[-1], [-1], [-1]], []
        for i in range(T):
            S.append(self.emission[o[i]])
            PI.append({})
            BP.append({})
            for b in S[i + 1]:
                for c in S[i + 2]:
                    for d in S[i + 3]:
                        max_a = None
                        max_p = -inf
                        for a in S[i]:
                            p = self._trans(a, b, c, d) * PI[i][(a, b, c)]
                            if p > max_p:
                                max_a = a
                                max_p = p
                        PI[i + 1][(b, c, d)] = max_p
                        BP[i][(b, c, d)] = max_a
        x = [0] * T
        if T < 3:
            return list(max(PI[T], key=PI[T].get))[3 - T:]
        (x[T - 3], x[T - 2], x[T - 1]) = max(PI[T], key=PI[T].get)
        for i in reversed(range(0, T - 3)):
            x[i] = BP[i + 3][(x[i + 1], x[i + 2], x[i + 3])]
        return x

    def _count_tri(self, i, j, k, delta):
        if (i, j, k) not in self.tritrans_mat:
            self.tritrans_mat[(i, j, k)] = delta
        else:
            self.tritrans_mat[(i, j, k)] += delta

    def _count_quad(self, i, j, k, l, delta):
        if (i, j, k, l) not in self.quadtrans_mat:
            self.quadtrans_mat[(i, j, k, l)] = delta
        else:
            self.quadtrans_mat[(i, j, k, l)] += delta

    def _trans(self, i, j, k, l):
        if k == -1:
            return self.start_prob[l]
        if j == -1:
            return self.bitrans_mat.get(
                (k, l), 0) * self.lamb[1] + self.start_prob[l] * self.lamb[0]
        if i == -1:
            return self.tritrans_mat.get(
                (j, k, l), 0) * self.lamb[2] + self.bitrans_mat.get(
                    (k, l),
                    0) * self.lamb[1] + self.start_prob[l] * self.lamb[0]
        return self.tritrans_mat.get(
            (i, j, k, l), 0) * self.lamb[3] + self.tritrans_mat.get(
                (j, k, l), 0) * self.lamb[2] + self.bitrans_mat.get(
                    (k, l),
                    0) * self.lamb[1] + self.start_prob[l] * self.lamb[0]
