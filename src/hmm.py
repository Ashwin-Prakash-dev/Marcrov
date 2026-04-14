
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GaussianHMM:
    """
    Baum-Welch Gaussian HMM with a sticky Dirichlet prior on transitions.

    Parameters
    ----------
    n_states : int
        Number of latent states (domain default: 4 → Calm/Elevated/Stressed/Crisis).
    n_iter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence tolerance.
    random_state : int
        Seed for GMM initialisation.
    covar_reg : float
        Ridge regularisation added to every covariance matrix.
    sticky_kappa : float
        Dirichlet concentration added to the diagonal of the transition count
        matrix before normalisation.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 300, tol: float = 1e-6,
                 random_state: int = 42, covar_reg: float = 1e-3,
                 sticky_kappa: float = 100.0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rs = random_state
        self.covar_reg = covar_reg
        self.sticky_kappa = sticky_kappa

        # Learned parameters 
        self.mu  = None
        self.cov = None
        self.A   = None   # (K, K) transition matrix
        self.pi  = None   # (K,)  initial distribution
        self.ll_ = -np.inf
        self.is_fitted = False

    # --------------------------------------- #
    #  Initialisation                         #
    # --------------------------------------- #

    def _init(self, X: np.ndarray):
        """Initialise parameters with a GMM warm-start."""
        K, D = self.K, X.shape[1]
        g = GaussianMixture(
            n_components=K,
            covariance_type='full',
            n_init=15,
            random_state=self.rs,
            reg_covar=self.covar_reg,
        )
        g.fit(X)
        self.mu  = g.means_.copy()
        self.cov = g.covariances_.copy()

        # Near-diagonal transition matrix with small off-diagonal mass
        off_diag = max(0.01, 0.5 / (K - 1)) if K > 1 else 0.0
        self.A = np.full((K, K), off_diag)
        np.fill_diagonal(self.A, 1.0 - off_diag * (K - 1))
        self.pi = np.full(K, 1.0 / K)

    # ----------------------------------- #
    #  Emission log-probabilities         #
    # ----------------------------------- #

    def _log_emit(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) matrix of Gaussian log-emission probabilities."""
        T, K = len(X), self.K
        logB = np.zeros((T, K))
        for k in range(K):
            logB[:, k] = multivariate_normal.logpdf(
                X, mean=self.mu[k], cov=self.cov[k], allow_singular=True
            )
        return logB

    # ------------------------------------- #
    #  Forward pass                         #
    # ------------------------------------- #

    def _forward(self, logB: np.ndarray):
        """
        Scaled forward pass.
        
        Returns
        -------
        alpha  : (T, K)  scaled forward variables
        scale  : (T,)    per-step scale factors
        shifts : (T,)    per-step row-max shifts used for emit
        """
        T, K = logB.shape
        alpha  = np.zeros((T, K))
        scale  = np.zeros(T)
        shifts = np.zeros(T)

        shift0    = logB[0].max()
        emit0     = np.exp(logB[0] - shift0)
        alpha[0]  = self.pi * emit0
        scale[0]  = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]
        shifts[0] = shift0

        for t in range(1, T):
            shift_t   = logB[t].max()
            emit_t    = np.exp(logB[t] - shift_t)
            raw       = (alpha[t - 1] @ self.A) * emit_t
            scale[t]  = raw.sum() or 1e-300
            alpha[t]  = raw / scale[t]
            shifts[t] = shift_t

        return alpha, scale, shifts

    # ------------------- #
    #  Backward pass      #
    # ------------------- #

    def _backward(self, logB: np.ndarray, scale: np.ndarray,
                  shifts: np.ndarray) -> np.ndarray:
        """
        Scaled backward pass.
        Uses the same per-step emit shift (shifts[t]) as the forward pass
        """
        T, K = logB.shape
        beta = np.ones((T, K))

        for t in range(T - 2, -1, -1):
            emit_t1 = np.exp(logB[t + 1] - shifts[t + 1])
            raw     = self.A * emit_t1 * beta[t + 1]   # (K, K)
            beta[t] = raw.sum(axis=1) / (scale[t + 1] or 1e-300)

        return beta

    # ------------------- #
    #  EM fit             #
    # ------------------- #

    def fit(self, X: np.ndarray):
        """
        Fits the HMM to a (T, D) array of observations via Baum-Welch EM.
        Parameters
        ----------
        X : np.ndarray, shape (T, D)
            Pre-processed (scaled + PCA-reduced) feature matrix.
        """
        self._init(X)
        T, K, D = len(X), self.K, X.shape[1]
        prev_ll = -np.inf

        print(f"    Fitting {K}-state HMM (sticky_kappa={self.sticky_kappa})...")

        for it in range(self.n_iter):
            # E-step
            logB               = self._log_emit(X)
            alpha, scale, shifts = self._forward(logB)
            beta               = self._backward(logB, scale, shifts)

            # Smoothed state posteriors γ
            gamma = alpha * beta
            gamma = np.clip(gamma, 0, None)
            rs    = gamma.sum(axis=1, keepdims=True)
            gamma /= np.where(rs == 0, 1e-300, rs)

            emit_tp1 = np.exp(logB[1:] - shifts[1:, np.newaxis])   # (T-1, K)
            xi = (alpha[:-1, :, None]      # (T-1, K, 1)
                  * self.A[None]           # (1,   K, K)
                  * emit_tp1[:, None, :]   # (T-1, 1, K)
                  * beta[1:, None, :])     # (T-1, 1, K)
            xs = xi.sum(axis=(1, 2), keepdims=True)
            xi /= np.where(xs == 0, 1e-300, xs)

            # M-step: initial distribution
            self.pi = gamma[0] / gamma[0].sum()

            # M-step: transitions with sticky Dirichlet prior
            A_counts = xi.sum(axis=0)
            A_counts[np.arange(K), np.arange(K)] += self.sticky_kappa
            self.A = A_counts / A_counts.sum(axis=1, keepdims=True)

            # M-step: Gaussian parameters
            gsum = gamma.sum(axis=0)
            for k in range(K):
                w = gamma[:, k]
                self.mu[k] = (w @ X) / (gsum[k] or 1e-300)
                d = X - self.mu[k]
                self.cov[k] = (
                    (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
                    / (gsum[k] or 1e-300)
                    + np.eye(D) * self.covar_reg
                )

            ll = np.sum(np.log(np.clip(scale, 1e-300, None)))
            if abs(ll - prev_ll) < self.tol:
                print(f"    Converged at iter={it + 1}  LL={ll:.2f}")
                break
            prev_ll = ll
        else:
            print(f"    Max iterations reached  LL={prev_ll:.2f}")

        self.ll_ = prev_ll
        self.is_fitted = True
        return self

    # ----------------------- #
    #  Inference              #
    # ----------------------- #

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return (T, K) smoothed posterior probabilities via forward-backward.

        Parameters
        ----------
        X : np.ndarray, shape (T, D)
            Must be transformed with the *same* scaler and PCA as training data.

        Returns
        -------
        gamma : np.ndarray, shape (T, K)
            Each row sums to 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("Model to be fitted before calling predict_proba().")

        logB               = self._log_emit(X)
        alpha, scale, shifts = self._forward(logB)
        beta               = self._backward(logB, scale, shifts)

        gamma = alpha * beta
        gamma = np.clip(gamma, 0, None)
        rs    = gamma.sum(axis=1, keepdims=True)
        gamma /= np.where(rs == 0, 1e-300, rs)
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding —> returns (T,) array of most-likely state indices.
        """
        if not self.is_fitted:
            raise RuntimeError("Model to be fitted before calling predict().")

        logB  = self._log_emit(X)
        T, K  = logB.shape
        logA  = np.log(np.clip(self.A,  1e-300, None))
        logpi = np.log(np.clip(self.pi, 1e-300, None))

        delta = logpi + logB[0]
        psi   = np.zeros((T, K), int)

        for t in range(1, T):
            tr     = delta[:, None] + logA
            psi[t] = tr.argmax(0)
            delta  = tr.max(0) + logB[t]

        s = np.zeros(T, int)
        s[-1] = delta.argmax()
        for t in range(T - 2, -1, -1):
            s[t] = psi[t + 1, s[t + 1]]
        return s

    # ------------------------------------------ #
    #  Model selection & diagnostics             #
    # ------------------------------------------ #

    def bic(self, X: np.ndarray) -> float:
        """
        BIC with corrected parameter count.
        
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing BIC.")

        T, D, K = len(X), X.shape[1], self.K
        n_params = (
            K * D                    # means
            + K * D * (D + 1) // 2  # symmetric covariance matrices
            + K * (K - 1)           # transition rows (each sums to 1 → K-1 free)
            + (K - 1)               # initial distribution
        )
        return -2 * self.ll_ + n_params * np.log(T)

    def log_transition_diagnostics(self) -> np.ndarray:
        """
        Print self-transition probabilities.

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        diag = self.A.diagonal()
        print("  Self-transition probabilities (target: >0.95 for daily data):")
        for k, p in enumerate(diag):
            flag = "" if p >= 0.95 else "  ← WARNING: low stickiness"
            print(f"    state {k}: {p:.4f}{flag}")
        return diag

    def sort_states_by_feature_mean(
        self,
        gamma_train: np.ndarray,
        feature_series: np.ndarray,
    ) -> list:
        """
        Using training weights avoids OOS contamination in the state ordering step.

        Parameters
        ----------
        gamma_train : np.ndarray, shape (T_train, K)
            Smoothed posteriors on the training set.
        feature_series : np.ndarray, shape (T_train,)
            Risk proxy aligned to gamma_train rows (e.g. VIX values).

        Returns
        -------
        state_order : list of int
            State indices sorted low-risk -> high-risk.
        """
        means = {}
        for k in range(self.K):
            w = gamma_train[:, k]
            means[k] = float((feature_series * w).sum() / (w.sum() or 1e-300))
        return sorted(means, key=means.get)
