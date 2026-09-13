from __future__ import annotations
import numpy as np
import pandas as pd


class KalmanFilterPair:
    """
    2D Kalman filter for online tracking of (beta_t, alpha_t).
    """
    def __init__(
        self,
        delta: float = 1e-4,
        observation_var: float = 1e-3,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
        initial_covariance: float = 1.0,
    ):
        self.delta = delta
        self.observation_var = observation_var
        # State vector: [beta, alpha]^T
        self.theta = np.array([initial_beta, initial_alpha], dtype=float)
        # State estimation covariance matrix P
        self.P = np.eye(2, dtype=float) * initial_covariance
        # Process noise covariance W
        self.W = (delta / (1.0 - delta)) * np.eye(2, dtype=float)

    def update(self, y: float, x: float) -> tuple[float, float, float, float]:
        """
        Processes observation (y_t, x_t) and updates state estimates.
        Returns (beta_t, alpha_t, error_t, innovation_variance_Q_t).
        """
        # H matrix: [x, 1]
        H = np.array([x, 1.0], dtype=float)

        # 1. State Prediction (Random walk assumption: theta_{t|t-1} = theta_{t-1})
        P_pred = self.P + self.W

        # 2. Measurement Prediction & Innovation
        y_pred = float(np.dot(H, self.theta))
        error = y - y_pred

        # Innovation variance Q = H * P_pred * H^T + R
        Q = float(np.dot(H, np.dot(P_pred, H)) + self.observation_var)

        # 3. Kalman Gain K = P_pred * H^T / Q
        K = np.dot(P_pred, H) / Q

        # 4. State & Covariance Update
        self.theta = self.theta + K * error
        self.P = P_pred - np.outer(K, H).dot(P_pred)

        beta_t = float(self.theta[0])
        alpha_t = float(self.theta[1])
        return beta_t, alpha_t, error, Q


def run_kalman_filter(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    observation_var: float = 1e-3,
    initial_beta: float = 1.0,
    initial_alpha: float = 0.0,
) -> pd.DataFrame:
    """
    Runs the Kalman filter over entire series y and x.
    """
    df = pd.concat([y, x], axis=1).dropna()
    y_vals = df.iloc[:, 0].values
    x_vals = df.iloc[:, 1].values
    n = len(df)

    kf = KalmanFilterPair(
        delta=delta,
        observation_var=observation_var,
        initial_beta=initial_beta,
        initial_alpha=initial_alpha,
    )

    betas = np.zeros(n)
    alphas = np.zeros(n)
    errors = np.zeros(n)
    var_q = np.zeros(n)

    for i in range(n):
        b, a, e, q = kf.update(y_vals[i], x_vals[i])
        betas[i] = b
        alphas[i] = a
        errors[i] = e
        var_q[i] = q

    out = pd.DataFrame(
        {
            "beta": betas,
            "alpha": alphas,
            "error": errors,
            "sqrt_q": np.sqrt(var_q),
            "spread": y_vals - (betas * x_vals + alphas),
        },
        index=df.index,
    )
    return out
