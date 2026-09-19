"""A Bradley-Terry scale fitted to fractional outcomes.

A comparison shows Jev two texts and returns y, the probability that the first one ranks higher.
The model is a fractional logit:

    E[y] = sigmoid(theta[first] - theta[second] + gamma)

theta is the scale, one number per text, in logit units: a gap of 1.0 means Jev gives the higher text
about 73% in a head-to-head. gamma is the lean toward whichever text is shown first, the same term a
sports model calls home advantage; positions are randomised, so it is estimated rather than assumed
away. A light ridge pins the mean of theta at zero and keeps a text that wins everything finite.

Fitting is Newton's method with the linear solve done by conjugate gradients, so no n-by-n matrix is
built until the standard errors, and refitting after every round of comparisons stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RIDGE = 0.01
EXACT_LIMIT = 4000  # above this many texts the n-by-n inverse is skipped and standard errors are estimated by probing


@dataclass
class Fit:
    theta: np.ndarray
    gamma: float
    converged: bool


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(_log_sigmoid(x))


def _objective(params: np.ndarray, n: int, first, second, y, ridge: float) -> float:
    d = params[first] - params[second] + params[n]
    return float(np.sum(y * _log_sigmoid(d) + (1 - y) * _log_sigmoid(-d)) - 0.5 * ridge * np.sum(params ** 2))


def _solve(apply, rhs: np.ndarray, tol: float = 1e-8, max_iter: int = 500) -> np.ndarray:
    """Conjugate gradients for a symmetric positive definite system given only its matrix-vector product."""
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = r.copy()
    rr = float(r @ r)
    stop = tol * tol * max(rr, 1e-300)
    for _ in range(max_iter):
        if rr <= stop:
            break
        ap = apply(p)
        alpha = rr / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        rr_new = float(r @ r)
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def fit(n: int, first, second, y, *, start: Fit | None = None, ridge: float = RIDGE,
        tol: float = 1e-7, max_iter: int = 100) -> Fit:
    """Maximise the ridge-penalised quasi-likelihood. `first` and `second` index the texts in each comparison."""
    first, second, y = np.asarray(first, dtype=np.intp), np.asarray(second, dtype=np.intp), np.asarray(y, dtype=float)
    params = np.zeros(n + 1)
    if start is not None and len(start.theta) == n:
        params[:n], params[n] = np.nan_to_num(start.theta), start.gamma
    if len(y) == 0:
        return Fit(params[:n], 0.0, True)

    value = _objective(params, n, first, second, y, ridge)
    converged = False
    for _ in range(max_iter):
        s = _sigmoid(params[first] - params[second] + params[n])
        resid = y - s
        grad = np.empty(n + 1)
        grad[:n] = np.bincount(first, resid, n) - np.bincount(second, resid, n)
        grad[n] = resid.sum()
        grad -= ridge * params
        w = s * (1 - s)

        def hessian(v: np.ndarray) -> np.ndarray:
            u = w * (v[first] - v[second] + v[n])
            out = ridge * v
            out[:n] += np.bincount(first, u, n) - np.bincount(second, u, n)
            out[n] += u.sum()
            return out

        step = _solve(hessian, grad)
        # The objective is concave, so a full Newton step nearly always rises; halve it when it does not.
        scale = 1.0
        while scale > 1e-4:
            trial = params + scale * step
            trial_value = _objective(trial, n, first, second, y, ridge)
            if trial_value >= value:
                break
            scale /= 2
        else:
            # No step improves the fit. That is the optimum if the step on offer was already tiny.
            converged = float(np.max(np.abs(step))) < 1e-4
            break
        params, moved, value = trial, float(np.max(np.abs(scale * step))), trial_value
        if moved < tol:
            converged = True
            break
    return Fit(params[:n], float(params[n]), converged)


def information(n: int, first, second, fitted: Fit, ridge: float = RIDGE) -> np.ndarray:
    """The diagonal of the Hessian: how much the comparisons so far say about each text. Cheap."""
    first, second = np.asarray(first, dtype=np.intp), np.asarray(second, dtype=np.intp)
    if len(first) == 0:
        return np.full(n, ridge)
    s = _sigmoid(fitted.theta[first] - fitted.theta[second] + fitted.gamma)
    w = s * (1 - s)
    return np.bincount(first, w, n) + np.bincount(second, w, n) + ridge


def shortfall(n: int, first, second, y, fitted: Fit, bar: float) -> np.ndarray:
    """How much worse the comparisons would fit if each text were moved up to `bar`, one text at a time.

    This is the test --top uses to stop asking about a text. A symmetric interval is the wrong tool:
    a text that lost 0.03 to 0.97 against a middling opponent has a wide interval, because a lopsided
    result says little about exactly where it sits, yet it plainly does not sit at the top. The drop
    in quasi-log-likelihood sees that. Zero for a text already at or above the bar.
    """
    first, second, y = np.asarray(first, dtype=np.intp), np.asarray(second, dtype=np.intp), np.asarray(y, dtype=float)
    theta, gamma = fitted.theta, fitted.gamma

    def loglik(d: np.ndarray) -> np.ndarray:
        return y * _log_sigmoid(d) + (1 - y) * _log_sigmoid(-d)

    now = loglik(theta[first] - theta[second] + gamma)
    drop = (np.bincount(first, now - loglik(bar - theta[second] + gamma), n)
            + np.bincount(second, now - loglik(theta[first] - bar + gamma), n))
    return np.where(theta >= bar, 0.0, drop)


def _probed_errors(n: int, first, second, w, r2, ridge: float, probes: int = 96, seed: int = 0) -> np.ndarray:
    """The same sandwich without the n-by-n inverse, for large n.

    If u is random with covariance C, then v = H^-1 u has covariance H^-1 C H^-1, and the average of v**2
    over many draws is that matrix's diagonal. Drawing u = X'(sqrt(c) z) gives C = X' diag(c) X, so one
    set of solves with c = w recovers H^-1, which the leverages need, and a second with c = the scaled
    squared residuals recovers the sandwich. Each solve is a conjugate-gradient run. The estimate is
    unbiased, with a relative error of about sqrt(1 / (2 * probes)) in each standard error: 7% at 96.
    """
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(np.bincount(first, w, n + 1)[: n + 1] + np.bincount(second, w, n + 1)[: n + 1] + ridge)
    scale[n] = 1 / np.sqrt(w.sum() + ridge)

    def hessian(v: np.ndarray) -> np.ndarray:            # preconditioned: D^-1/2 H D^-1/2, whose diagonal is 1
        v = v * scale
        u = w * (v[first] - v[second] + v[n])
        out = ridge * v
        out[:n] += np.bincount(first, u, n) - np.bincount(second, u, n)
        out[n] += u.sum()
        return out * scale

    def draw(c: np.ndarray, with_ridge: bool) -> np.ndarray:
        z = np.sqrt(c) * rng.standard_normal(len(c))
        u = np.zeros(n + 1)
        u[:n] = np.bincount(first, z, n) - np.bincount(second, z, n)
        u[n] = z.sum()
        if with_ridge:
            u += np.sqrt(ridge) * rng.standard_normal(n + 1)
        return _solve(hessian, u * scale, tol=1e-5) * scale

    quadratic = np.zeros(len(w))
    for _ in range(probes // 3):
        v = draw(w, True)
        quadratic += (v[first] - v[second] + v[n]) ** 2
    leverage = np.clip(w * quadratic / (probes // 3), 0.0, 0.9)
    adjusted = r2 / (1 - leverage) ** 2
    variance = np.zeros(n)
    for _ in range(probes):
        variance += draw(adjusted, False)[:n] ** 2
    return np.sqrt(variance / probes)


def standard_errors(n: int, first, second, y, fitted: Fit, ridge: float = RIDGE) -> np.ndarray:
    """Robust (sandwich) standard errors for theta.

    y is a probability, not a coin flip, so the binomial variance a logit would assume is wrong. The
    sandwich uses the squared residuals in its place, which is the usual treatment of a fractional
    logit. Where Jev's answers line up on one scale the residuals are small and so are the errors;
    where they contradict each other the errors grow. With several thousand texts the n-by-n inverse is
    skipped and each text's error comes from its own diagonal terms, which ignores the uncertainty it
    inherits from its opponents.
    """
    first, second, y = np.asarray(first, dtype=np.intp), np.asarray(second, dtype=np.intp), np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.full(n, np.nan)
    s = _sigmoid(fitted.theta[first] - fitted.theta[second] + fitted.gamma)
    w, r2 = s * (1 - s), (y - s) ** 2

    if n > EXACT_LIMIT:
        return _probed_errors(n, first, second, w, r2, ridge)

    def accumulate(weights: np.ndarray) -> np.ndarray:
        m = np.zeros((n + 1, n + 1))
        np.add.at(m, (first, first), weights)
        np.add.at(m, (second, second), weights)
        np.add.at(m, (first, second), -weights)
        np.add.at(m, (second, first), -weights)
        np.add.at(m, (first, n), weights)
        np.add.at(m, (n, first), weights)
        np.add.at(m, (second, n), -weights)
        np.add.at(m, (n, second), -weights)
        m[n, n] = weights.sum()
        return m

    inv = np.linalg.inv(accumulate(w) + ridge * np.eye(n + 1))
    # HC3: a comparison that largely determines its own fitted value has a residual that understates
    # its error, so each squared residual is scaled up by its leverage.
    leverage = w * (inv[first, first] + inv[second, second] - 2 * inv[first, second]
                    + inv[n, n] + 2 * inv[first, n] - 2 * inv[second, n])
    cov = inv @ accumulate(r2 / (1 - np.clip(leverage, 0.0, 0.9)) ** 2) @ inv
    return np.sqrt(np.clip(np.diag(cov)[:n], 0.0, None))


def _largest_component(n: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Which texts belong to the largest group that the comparisons connect, as a mask."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(first.tolist(), second.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    compared = np.zeros(n, dtype=bool)
    compared[first] = compared[second] = True
    if not compared.any():
        return compared
    sizes = np.bincount(roots[compared], minlength=n)
    return compared & (roots == int(np.argmax(sizes)))


def reliability(n: int, first, second, y, *, seed: int = 0, ridge: float = RIDGE) -> float | None:
    """Split-half reliability, stepped up to full length with Spearman-Brown.

    The comparisons are dealt at random into two halves and a scale is fitted to each. If the two
    scales agree, the ranking does not hang on which pairs happened to be asked. With few comparisons
    per text a half can fall into pieces that no comparison joins, and the positions of two such pieces
    relative to each other are not data, only the ridge. So the two scales are compared on the texts
    that sit in the largest connected piece of both halves. None when that is too few to say.
    """
    first, second, y = np.asarray(first, dtype=np.intp), np.asarray(second, dtype=np.intp), np.asarray(y, dtype=float)
    if len(y) < 8 or n < 4:
        return None
    half = np.random.default_rng(seed).permutation(len(y)) % 2 == 0
    both = np.ones(n, dtype=bool)
    scales = []
    for mask in (half, ~half):
        both &= _largest_component(n, first[mask], second[mask])
        scales.append(fit(n, first[mask], second[mask], y[mask], ridge=ridge).theta)
    compared = np.zeros(n, dtype=bool)
    compared[first] = compared[second] = True
    if both.sum() < max(4, compared.sum() / 3):
        return None
    a, b = scales[0][both], scales[1][both]
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, 2 * r / (1 + r)) if r > -1 else 0.0
