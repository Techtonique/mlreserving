"""
Machine-learning based loss reserving with self-contained conformal prediction intervals.

Dependencies: numpy, pandas, scikit-learn, scipy  (no nnetsauce required)

Statistical notes
-----------------
* Conformal coverage guarantee holds in arcsinh-transformed space.
  After the sinh back-transformation the intervals are approximate and
  asymmetric; this is documented explicitly where it matters.
* When type_pi is not None the mean is computed as the average of
  back-transformed simulations (bias-corrected), not sinh(E[Z]).
* log_calendar is intentionally excluded from the default feature set to
  avoid calendar-year leakage through the scaler.  It can be re-enabled
  via use_calendar_feature=True if the user accepts the trade-off.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Public named-tuple used as return type throughout
# ---------------------------------------------------------------------------

DescribeResult = namedtuple("DescribeResult", ("mean", "lower", "upper"))


# ---------------------------------------------------------------------------
# Variance-stabilising transforms
# ---------------------------------------------------------------------------

def _arcsinh(x: np.ndarray) -> np.ndarray:
    """arcsinh variance-stabilising transform.  Maps 0 -> 0 exactly."""
    return np.arcsinh(np.asarray(x, dtype=float))


def _inv_arcsinh(x: np.ndarray) -> np.ndarray:
    """Inverse of _arcsinh."""
    return np.sinh(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Triangle pivot helper
# ---------------------------------------------------------------------------

def _df_to_triangle(
    df: pd.DataFrame,
    origin_col: str,
    dev_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Pivot long DataFrame -> origin x development triangle."""
    return df.pivot(index=origin_col, columns=dev_col, values=value_col)


# ---------------------------------------------------------------------------
# Self-contained split-conformal regressor
# ---------------------------------------------------------------------------

class _SplitConformalRegressor(BaseEstimator, RegressorMixin):
    """Split-conformal prediction intervals with optional simulation.

    Parameters
    ----------
    base_model : sklearn-compatible regressor
        Any estimator with fit() / predict().  Cloned internally.
    level : float
        Coverage level in percent (e.g. 95 -> 95% intervals).
    type_pi : {None, 'bootstrap', 'kde'}
        None        -> plain symmetric conformal band.
        'bootstrap' -> resample calibration residuals (faster).
        'kde'       -> KDE-smoothed residual distribution.
    replications : int or None
        Simulation draws when type_pi is not None.  Defaults to 200.
    calibration_fraction : float
        Fraction of training rows reserved for conformal calibration,
        taken from the *end* of the sorted sequence to respect time order.
    random_state : int

    Attributes (post-fit)
    ----------------------
    _residuals       : signed calibration residuals (y_cal - y_hat_cal)
    _abs_residuals   : |_residuals|
    _quantile        : finite-sample conformal half-width (plain mode)
    _last_sims       : (n_test, reps) array in arcsinh space (simulation mode)
    """

    def __init__(
        self,
        base_model=None,
        level: float = 95,
        type_pi: str | None = None,
        replications: int | None = None,
        calibration_fraction: float = 0.5,
        random_state: int = 42,
    ) -> None:
        if base_model is None:
            base_model = RidgeCV(alphas=np.logspace(-4, 2, 30))
        self.base_model = base_model
        self.level = level
        self.type_pi = type_pi
        self.replications = replications
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

        self.alpha_: float = 1.0 - level / 100.0
        self._fitted_model = None
        self._residuals: np.ndarray | None = None
        self._abs_residuals: np.ndarray | None = None
        self._quantile: float | None = None
        self._last_sims: np.ndarray | None = None   # (n_test, reps), arcsinh space

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SplitConformalRegressor":
        """Sequential split: first (1-cal_frac) rows train, last cal_frac calibrate.

        Sequential (rather than random) split respects the time ordering of
        the loss triangle: earlier observed cells train the model, later
        observed cells calibrate the conformal band.
        """
        n = X.shape[0]
        split = int(n * (1.0 - self.calibration_fraction))
        split = max(split, 1)      # guarantee at least one training sample
        split = min(split, n - 1)  # guarantee at least one calibration sample

        n_cal = n - split
        if n_cal < 10:
            warnings.warn(
                f"Only {n_cal} calibration samples available. Conformal coverage "
                "may be unreliable. Consider reducing calibration_fraction or "
                "using a larger triangle.",
                UserWarning,
                stacklevel=2,
            )
        if self.type_pi == "kde" and n_cal < 30:
            warnings.warn(
                f"KDE fitted on only {n_cal} residuals; bandwidth estimation may "
                "be unstable. Consider type_pi='bootstrap' for small calibration sets.",
                UserWarning,
                stacklevel=2,
            )

        X_train, X_cal = X[:split], X[split:]
        y_train, y_cal = y[:split], y[split:]

        self._fitted_model = clone(self.base_model)
        self._fitted_model.fit(X_train, y_train)

        preds_cal = self._fitted_model.predict(X_cal)
        self._residuals = y_cal - preds_cal
        self._abs_residuals = np.abs(self._residuals)

        # Finite-sample-corrected conformal quantile (Vovk et al. 2005):
        #   k = ceil((n_cal + 1) * (1 - alpha)), then take the k-th order statistic.
        # This guarantees exact marginal coverage (unlike raw np.quantile).
        k = int(np.ceil((n_cal + 1) * (1.0 - self.alpha_)))
        k = min(k, n_cal)
        self._quantile = float(np.sort(self._abs_residuals)[k - 1])

        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        return_pi: bool = False,
    ) -> np.ndarray | DescribeResult:
        """Point predictions, optionally with prediction intervals.

        All values are returned in arcsinh-transformed space.
        Back-transformation and bias correction are applied by the caller.

        Parameters
        ----------
        X : (n_test, n_features) array
        return_pi : bool

        Returns
        -------
        np.ndarray  or  DescribeResult(mean, lower, upper)
        """
        if self._fitted_model is None:
            raise ValueError("Call fit() before predict().")

        point = self._fitted_model.predict(X)

        if not return_pi:
            return point

        use_simulation = (self.replications is not None) or (self.type_pi is not None)

        if use_simulation:
            mean_, lower_, upper_ = self._simulate_intervals(point)
            # _last_sims stored for bias-corrected back-transform in caller
            return DescribeResult(mean_, lower_, upper_)
        else:
            # Plain symmetric conformal band; coverage exact in arcsinh space
            return DescribeResult(point, point - self._quantile, point + self._quantile)

    # ------------------------------------------------------------------
    def _simulate_intervals(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw residual simulations; populate _last_sims; return (mean, lower, upper).

        _last_sims has shape (n_obs, replications) and is in arcsinh space.
        """
        n_obs = len(point)
        reps = self.replications if self.replications is not None else 200
        pi_type = self.type_pi if self.type_pi is not None else "kde"

        if pi_type not in ("bootstrap", "kde"):
            raise ValueError("type_pi must be 'bootstrap', 'kde', or None.")

        rng = np.random.default_rng(self.random_state)

        if pi_type == "bootstrap":
            idx = rng.integers(0, len(self._residuals), size=(n_obs, reps))
            residual_draws = self._residuals[idx]   # (n_obs, reps)
        else:  # kde
            kde = gaussian_kde(self._residuals)
            draws_flat = kde.resample(
                size=n_obs * reps, seed=int(self.random_state)
            ).ravel()
            residual_draws = draws_flat.reshape(n_obs, reps)

        self._last_sims = point[:, None] + residual_draws  # (n_obs, reps), arcsinh space

        q_lo = self.alpha_ / 2.0
        q_hi = 1.0 - self.alpha_ / 2.0
        lower_ = np.quantile(self._last_sims, q=q_lo, axis=1)
        upper_ = np.quantile(self._last_sims, q=q_hi, axis=1)
        # Mean in arcsinh space returned here; bias-corrected mean in original
        # space is computed by caller via mean(sinh(sims)).
        mean_ = self._last_sims.mean(axis=1)

        return mean_, lower_, upper_


# ---------------------------------------------------------------------------
# MLReserving
# ---------------------------------------------------------------------------

class MLReserving:
    """Machine-learning based loss reserving with conformal prediction intervals.

    Internal workflow
    -----------------
    1. Input validation; cumulative -> incremental conversion (first period
       set to its cumulative value; no bfill artefact).
    2. arcsinh variance-stabilising transform on observed incremental cells.
    3. Feature matrix built from training cells only; scaler fit on training.
    4. _SplitConformalRegressor fit (sequential split respecting time order).
    5. Predict unobserved cells in arcsinh space.
    6. Back-transform:
       - Simulation mode: mean = mean(sinh(Z_sim)) over replications
         (bias-corrected for Jensen's inequality).
       - Plain mode: mean = sinh(point_pred).
       - Bounds: sinh(lower/upper in arcsinh space), floored at 0.
    7. Compute IBNR = sum of predicted incremental cells per origin year.
    8. Reconstruct cumulative triangle by appending cumsum of predicted
       incremental cells to the last observed cumulative value per origin.
       Observed cells are never re-cumulated.

    Coverage note
    -------------
    The conformal coverage guarantee holds in arcsinh space. The sinh
    back-transformation is non-linear, so interval coverage in original loss
    space is approximate and asymmetric.  This is a known trade-off of
    transformation-based conformal methods.

    Parameters
    ----------
    model : sklearn-compatible regressor, optional
        Defaults to RidgeCV with a log-spaced alpha grid.
    level : float
        Coverage level in percent (default 95).
    type_pi : {None, 'bootstrap', 'kde'}
        Simulation strategy for prediction intervals.
    replications : int or None
        Number of simulation draws when type_pi is not None.
    calibration_fraction : float
        Fraction of observed cells held back for conformal calibration
        (time-ordered: later cells -> calibration). Default 0.5.
    use_factors : bool
        True  -> OneHotEncode origin + development period as features.
        False -> log(origin), log(dev) only (default; no calendar leakage).
    use_calendar_feature : bool
        Adds log(calendar year) to the feature set. Disabled by default
        because the scaler is fit on the full rectangular grid (including
        future cells), which leaks future calendar trend into training.
        Enable only if you accept and understand this trade-off.
    random_state : int
    """

    def __init__(
        self,
        model=None,
        level: float = 95,
        type_pi: str | None = None,
        replications: int | None = None,
        calibration_fraction: float = 0.5,
        use_factors: bool = False,
        use_calendar_feature: bool = False,
        random_state: int = 42,
    ) -> None:
        if model is None:
            model = RidgeCV(alphas=np.logspace(-4, 2, 30))

        self._base_model = model
        self.level = level
        self.type_pi = type_pi
        self.replications = replications
        self.calibration_fraction = calibration_fraction
        self.use_factors = use_factors
        self.use_calendar_feature = use_calendar_feature
        self.random_state = random_state

        # Set by fit()
        self.origin_col: str | None = None
        self.development_col: str | None = None
        self.value_col: str | None = None
        self.cumulated: bool | None = None
        self.max_dev: int | None = None
        self.origin_years: np.ndarray | None = None
        self.latest_cumulative_: pd.Series | None = None

        # Set by predict()
        self.ibnr_mean_: pd.Series | None = None
        self.ibnr_lower_: pd.Series | None = None
        self.ibnr_upper_: pd.Series | None = None
        self.ultimate_: pd.Series | None = None
        self.ultimate_lower_: pd.Series | None = None
        self.ultimate_upper_: pd.Series | None = None

        # Internal fitted objects
        self._conformal: _SplitConformalRegressor | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._origin_encoder: OneHotEncoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self._dev_encoder: OneHotEncoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self._feature_cols: list[str] = []
        self._full_data: pd.DataFrame | None = None   # long-format, incremental
        self._X_test: np.ndarray | None = None        # scaled test features

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MLReserving(level={self.level}, type_pi={self.type_pi!r}, "
            f"replications={self.replications}, "
            f"calibration_fraction={self.calibration_fraction}, "
            f"use_factors={self.use_factors}, "
            f"use_calendar_feature={self.use_calendar_feature})"
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        data: pd.DataFrame,
        origin_col: str,
        development_col: str,
        value_col: str,
    ) -> None:
        missing_cols = [c for c in (origin_col, development_col, value_col)
                        if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Column(s) not found in data: {missing_cols}")
        if data[value_col].isna().all():
            raise ValueError("All values in value_col are NaN — nothing to fit.")
        n_neg = (data[value_col].dropna() < 0).sum()
        if n_neg:
            warnings.warn(
                f"{n_neg} negative value(s) detected in '{value_col}'. "
                "arcsinh handles negatives numerically but negative incremental "
                "losses are unusual; verify the data.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        full_data: pd.DataFrame,
        fit_encoders: bool,
        train_mask: np.ndarray,
    ) -> np.ndarray:
        """Build raw (unscaled) feature matrix for all rows of full_data.

        Encoders are fit only on training rows (fit_encoders=True on first
        call) to prevent leakage.  Calendar feature is opt-in with a warning.
        """
        if self.use_factors:
            train_origin = full_data.loc[train_mask, [self.origin_col]]
            train_dev = full_data.loc[train_mask, ["dev"]]

            if fit_encoders:
                self._origin_encoder.fit(train_origin)
                self._dev_encoder.fit(train_dev)

            origin_enc = self._origin_encoder.transform(full_data[[self.origin_col]])
            dev_enc = self._dev_encoder.transform(full_data[["dev"]])
            X = np.hstack([origin_enc, dev_enc])

            self._feature_cols = (
                list(self._origin_encoder.get_feature_names_out([self.origin_col]))
                + list(self._dev_encoder.get_feature_names_out(["dev"]))
            )
        else:
            log_origin = np.log(full_data[self.origin_col].astype(float).values)
            log_dev = np.log(full_data["dev"].astype(float).values)
            X = np.column_stack([log_origin, log_dev])
            self._feature_cols = ["log_origin", "log_dev"]

        if self.use_calendar_feature:
            warnings.warn(
                "use_calendar_feature=True: log(calendar) is computed on the full "
                "rectangular grid (including future cells) before the scaler is fit. "
                "This leaks future calendar-year trend into training feature scaling. "
                "Set use_calendar_feature=False to avoid this.",
                UserWarning,
                stacklevel=3,
            )
            cal = np.log(
                (full_data[self.origin_col] + full_data["dev"] - 1)
                .astype(float).values
            )
            X = np.column_stack([X, cal])
            self._feature_cols.append("log_calendar")

        return X

    # ------------------------------------------------------------------
    # Cumulative -> incremental conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _cum_to_inc(s: pd.Series) -> pd.Series:
        """Convert a sorted cumulative series to incremental.

        First observed period: incremental = cumulative value.
        Missing first values remain NaN (not filled by bfill).
        """
        inc = s.diff()
        if pd.notna(s.iloc[0]):
            inc.iloc[0] = s.iloc[0]
        return inc

    # ------------------------------------------------------------------
    # Public fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        origin_col: str = "origin",
        development_col: str = "development",
        value_col: str = "values",
        cumulated: bool = True,
    ) -> "MLReserving":
        """Fit the ML reserving model.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format triangle.  Upper-right (future) cells should be absent
            or NaN.  Must contain origin_col, development_col, value_col.
        origin_col, development_col, value_col : str
        cumulated : bool
            True if value_col contains cumulative losses; False if incremental.

        Returns
        -------
        self
        """
        self._validate_input(data, origin_col, development_col, value_col)

        self.origin_col = origin_col
        self.development_col = development_col
        self.value_col = value_col
        self.cumulated = cumulated

        df = data[[origin_col, development_col, value_col]].copy()
        df["dev"] = (df[development_col] - df[origin_col] + 1).astype(int)
        df = df.sort_values([origin_col, "dev"]).reset_index(drop=True)

        # ---- Capture latest observed cumulative BEFORE any differencing ----
        if cumulated:
            self.latest_cumulative_ = (
                df.groupby(origin_col)[value_col]
                .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
            )
        else:
            # Incremental input: latest cumulative = sum of all observed increments
            self.latest_cumulative_ = (
                df.groupby(origin_col)[value_col]
                .apply(lambda s: s.dropna().sum())
            )

        # ---- Convert cumulative -> incremental (if needed) -----------------
        if cumulated:
            df[value_col] = df.groupby(origin_col)[value_col].transform(
                self._cum_to_inc
            )

        # ---- Build full rectangular grid -----------------------------------
        self.max_dev = int(df["dev"].max())
        self.origin_years = np.sort(df[origin_col].unique())

        full_grid = pd.MultiIndex.from_product(
            [self.origin_years, range(1, self.max_dev + 1)],
            names=[origin_col, "dev"],
        ).to_frame(index=False)

        full_data = (
            pd.merge(full_grid, df[[origin_col, "dev", value_col]],
                     on=[origin_col, "dev"], how="left")
            .reset_index(drop=True)
        )
        full_data["to_predict"] = full_data[value_col].isna()
        self._full_data = full_data

        train_mask = ~full_data["to_predict"].values
        test_mask = full_data["to_predict"].values

        if not test_mask.any():
            warnings.warn(
                "No missing cells detected. The triangle appears complete. "
                "predict() will return the observed triangle with zero IBNR.",
                UserWarning,
                stacklevel=2,
            )

        # ---- Feature matrix (encoders fit on training rows only) -----------
        X_all = self._build_feature_matrix(
            full_data, fit_encoders=True, train_mask=train_mask
        )
        X_train_raw = X_all[train_mask]
        X_test_raw = X_all[test_mask]

        # Scaler fit on training rows only
        X_train = self._scaler.fit_transform(X_train_raw)
        self._X_test = (
            self._scaler.transform(X_test_raw)
            if test_mask.any()
            else np.empty((0, X_train_raw.shape[1]))
        )

        # ---- Transform response (arcsinh of incremental training values) ---
        y_train = _arcsinh(
            full_data.loc[train_mask, value_col].astype(float).values
        )

        # ---- Fit conformal regressor ---------------------------------------
        self._conformal = _SplitConformalRegressor(
            base_model=self._base_model,
            level=self.level,
            type_pi=self.type_pi,
            replications=self.replications,
            calibration_fraction=self.calibration_fraction,
            random_state=self.random_state,
        )
        self._conformal.fit(X_train, y_train)

        return self

    # ------------------------------------------------------------------

    def predict(self) -> DescribeResult:
        """Predict unobserved cells and compute IBNR / ultimate reserves.

        When simulation is used (type_pi is not None), the mean prediction
        is bias-corrected:
            E_orig = mean(sinh(Z_sim))  over replications
        rather than sinh(mean(Z_sim)), correcting for Jensen's inequality.

        Returns
        -------
        DescribeResult(mean, lower, upper)
            Each field is a pd.DataFrame (origin x development period).
            Rows = origin years, columns = development lags (integer).
        """
        if self._conformal is None:
            raise ValueError("Call fit() before predict().")

        test_mask = self._full_data["to_predict"].values

        # ---- Handle fully observed triangle --------------------------------
        if not test_mask.any():
            empty = pd.Series(0.0, index=self.origin_years)
            self.ibnr_mean_ = self.ibnr_lower_ = self.ibnr_upper_ = empty.copy()
            self.ultimate_ = self.latest_cumulative_.copy()
            self.ultimate_lower_ = self.latest_cumulative_.copy()
            self.ultimate_upper_ = self.latest_cumulative_.copy()
            tri = _df_to_triangle(self._full_data, self.origin_col, "dev", self.value_col)
            return DescribeResult(tri, tri.copy(), tri.copy())

        # ---- Predict in arcsinh space --------------------------------------
        raw = self._conformal.predict(self._X_test, return_pi=True)

        use_simulation = (self.replications is not None) or (self.type_pi is not None)

        if use_simulation and self._conformal._last_sims is not None:
            # Bias-corrected mean: average of sinh(simulations) in original space
            sims_orig = np.maximum(0.0, _inv_arcsinh(self._conformal._last_sims))
            mean_inc = sims_orig.mean(axis=1)
        else:
            mean_inc = np.maximum(0.0, _inv_arcsinh(raw.mean))

        lower_inc = np.maximum(0.0, _inv_arcsinh(raw.lower))
        upper_inc = np.maximum(0.0, _inv_arcsinh(raw.upper))

        # ---- IBNR per origin year ------------------------------------------
        origins_test = self._full_data.loc[test_mask, self.origin_col].values

        def _aggregate(values: np.ndarray) -> pd.Series:
            return (
                pd.Series(values, index=origins_test, dtype=float)
                .groupby(level=0).sum()
                .reindex(self.origin_years)
                .fillna(0.0)
            )

        self.ibnr_mean_ = _aggregate(mean_inc)
        self.ibnr_lower_ = _aggregate(lower_inc)
        self.ibnr_upper_ = _aggregate(upper_inc)

        # ---- Ultimate = latest observed cumulative + IBNR ------------------
        self.ultimate_ = self.latest_cumulative_ + self.ibnr_mean_
        self.ultimate_lower_ = self.latest_cumulative_ + self.ibnr_lower_
        self.ultimate_upper_ = self.latest_cumulative_ + self.ibnr_upper_

        # ---- Assemble triangles --------------------------------------------
        mean_tri = self._build_triangle(mean_inc, test_mask)
        lower_tri = self._build_triangle(lower_inc, test_mask)
        upper_tri = self._build_triangle(upper_inc, test_mask)

        return DescribeResult(mean_tri, lower_tri, upper_tri)

    # ------------------------------------------------------------------

    def _build_triangle(
        self,
        future_inc: np.ndarray,
        test_mask: np.ndarray,
    ) -> pd.DataFrame:
        """Assemble a full triangle DataFrame from observed + predicted cells.

        Observed cells: incremental values stored in _full_data, re-cumulated
        via np.cumsum per origin.  Index-based (not positional) assignment is
        used throughout to handle ragged triangles safely.

        Future cells: predicted incremental, prepended with the last observed
        cumulative, then cumsum'd.
        """
        fd = self._full_data.copy()
        fd.loc[test_mask, self.value_col] = future_inc

        if not self.cumulated:
            return _df_to_triangle(fd, self.origin_col, "dev", self.value_col)

        rows = []
        for origin in self.origin_years:
            grp = fd[fd[self.origin_col] == origin].sort_values("dev").copy()
            obs_mask = ~grp["to_predict"].values
            fut_mask = grp["to_predict"].values
            cum_vals = grp[self.value_col].copy().astype(float)

            # Observed cells: cumsum of stored incremental (index-safe)
            if obs_mask.any():
                obs_idx = grp.index[obs_mask]
                cum_vals.loc[obs_idx] = np.cumsum(
                    grp.loc[obs_idx, self.value_col].astype(float).values
                )

            # Future cells: last known cumulative + cumsum of predicted increments
            if fut_mask.any():
                last_cum = float(self.latest_cumulative_[origin])
                fut_idx = grp.index[fut_mask]
                cum_vals.loc[fut_idx] = (
                    last_cum
                    + np.cumsum(grp.loc[fut_idx, self.value_col].astype(float).values)
                )

            grp = grp.copy()
            grp[self.value_col] = cum_vals.values
            rows.append(grp)

        return _df_to_triangle(
            pd.concat(rows, ignore_index=True),
            self.origin_col, "dev", self.value_col,
        )

    # ------------------------------------------------------------------
    # Actuarial output methods
    # ------------------------------------------------------------------

    def get_ibnr(self) -> DescribeResult:
        """IBNR estimates by origin year.

        Returns
        -------
        DescribeResult(mean, lower, upper) — each a pd.Series indexed by origin.
        """
        if self.ibnr_mean_ is None:
            raise ValueError("Call predict() before get_ibnr().")
        return DescribeResult(self.ibnr_mean_, self.ibnr_lower_, self.ibnr_upper_)

    def get_ultimate(self) -> DescribeResult:
        """Ultimate loss estimates by origin year.

        Returns
        -------
        DescribeResult(mean, lower, upper) — each a pd.Series indexed by origin.
        """
        if self.ultimate_ is None:
            raise ValueError("Call predict() before get_ultimate().")
        return DescribeResult(self.ultimate_, self.ultimate_lower_, self.ultimate_upper_)

    def get_latest(self) -> pd.Series:
        """Last observed cumulative loss per origin year (pre-fit)."""
        if self.latest_cumulative_ is None:
            raise ValueError("Call fit() before get_latest().")
        return self.latest_cumulative_.copy()

    def get_residual_diagnostics(self) -> dict:
        """Calibration residuals for diagnostic plots and coverage assessment.

        Returns
        -------
        dict with keys:
            'residuals'     : signed calibration residuals (arcsinh space)
            'abs_residuals' : absolute values of calibration residuals
            'quantile'      : finite-sample conformal half-width (plain mode)
        """
        if self._conformal is None or self._conformal._residuals is None:
            raise ValueError("Call fit() before get_residual_diagnostics().")
        return {
            "residuals": self._conformal._residuals.copy(),
            "abs_residuals": self._conformal._abs_residuals.copy(),
            "quantile": self._conformal._quantile,
        }

    def get_summary(self) -> dict:
        """Reserving summary table.

        Returns
        -------
        dict with keys:
            'ByOrigin' : pd.DataFrame — one row per origin year with Latest,
                         Mean Ultimate, Mean IBNR, and interval bounds.
            'Totals'   : pd.Series — column totals across all origin years.
        """
        if self.ultimate_ is None:
            raise ValueError("Call predict() before get_summary().")

        ibnr = self.get_ibnr()
        ult = self.get_ultimate()
        latest = self.get_latest()

        by_origin = pd.DataFrame(
            {
                "Latest": latest,
                "Mean Ultimate": ult.mean,
                "Mean IBNR": ibnr.mean,
                f"IBNR Lo{self.level}%": ibnr.lower,
                f"IBNR Hi{self.level}%": ibnr.upper,
                f"Ultimate Lo{self.level}%": ult.lower,
                f"Ultimate Hi{self.level}%": ult.upper,
            }
        )

        totals = pd.Series(
            {
                "Latest": latest.sum(),
                "Mean Ultimate": ult.mean.sum(),
                "Mean IBNR": ibnr.mean.sum(),
                f"IBNR Lo{self.level}% Total": ibnr.lower.sum(),
                f"IBNR Hi{self.level}% Total": ibnr.upper.sum(),
            }
        )

        return {"ByOrigin": by_origin, "Totals": totals}