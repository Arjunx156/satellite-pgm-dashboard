"""Section 8 of the notebook: two-state HMM (0 = Nominal, 1 = Anomalous).

Filtering  P(S_t | O_1:t)   uses only past and current evidence: online / predictive use.
Smoothing  P(S_t | O_1:T)   also uses future evidence: retrospective / offline analysis only.
Prediction P(S_t+1 | O_1:t) = P(S_t | O_1:t) @ A, built on filtering, never on smoothing.
"""

import numpy as np


def fit_transitions(seqs, y, train_mask):
    """pi and A counted from training labels only, Laplace smoothed, within each channel."""
    pi_count, A_count = np.ones(2), np.ones((2, 2))
    for ix in seqs:
        if train_mask[ix[0]]:
            pi_count[y[ix[0]]] += 1
        for a, b in zip(ix[:-1], ix[1:]):
            if train_mask[a] and train_mask[b]:             # both labels from the training split
                A_count[y[a], y[b]] += 1
    pi = pi_count / pi_count.sum()
    A = A_count / A_count.sum(axis=1, keepdims=True)
    assert np.allclose(A.sum(axis=1), 1) and np.isclose(pi.sum(), 1)
    return pi, A


def forward(loge, A, pi):
    """Normalised forward recursion. Row t is P(S_t | O_1:t); it never looks past t."""
    B = np.exp(loge - loge.max(axis=1, keepdims=True))   # per-step scaling cancels on normalisation
    alpha = np.zeros((len(B), 2))
    alpha[0] = pi * B[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, len(B)):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        alpha[t] /= alpha[t].sum()
    return alpha


def filter_step(prior_belief, loge_row, A):
    """One forward step from the previous filtered belief and one new observation."""
    b = np.exp(loge_row - loge_row.max())
    alpha = (prior_belief @ A) * b
    return alpha / alpha.sum()


def initial_step(loge_row, pi):
    """Filtered belief for the first segment of a sequence (no history)."""
    b = np.exp(loge_row - loge_row.max())
    alpha = pi * b
    return alpha / alpha.sum()


def predict_next(belief, A):
    """P(S_t+1 | O_1:t) = sum_s P(S_t+1 | S_t = s) P(S_t = s | O_1:t)."""
    return np.asarray(belief) @ A


def forward_backward(loge, A, pi):
    """Notebook implementation, kept verbatim. Returns filtered and smoothed P(anomalous)."""
    B = np.exp(loge - loge.max(axis=1, keepdims=True))
    T = len(B)
    alpha, beta, c = np.zeros((T, 2)), np.ones((T, 2)), np.zeros(T)
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return alpha[:, 1], gamma[:, 1]
