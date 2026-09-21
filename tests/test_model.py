"""The scale itself: fitting, standard errors and the diagnostics. No network."""

import itertools

import numpy as np

from jsort.model import Fit, fit, information, place, reliability, shortfall, standard_errors


def all_pairs(theta, gamma=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pairs = list(itertools.permutations(range(len(theta)), 2))
    first, second = (np.array(c) for c in zip(*pairs))
    d = theta[first] - theta[second] + gamma + rng.standard_normal(len(pairs)) * noise
    return first, second, 1 / (1 + np.exp(-d))


def test_recovers_a_known_scale_and_the_position_lean():
    theta = np.array([-2.0, -1.0, -0.2, 0.3, 1.1, 1.8])
    first, second, y = all_pairs(theta, gamma=-0.25)
    got = fit(len(theta), first, second, y)
    assert got.converged
    assert np.allclose(got.theta, theta - theta.mean(), atol=0.03)   # the ridge shrinks it by about 1%
    assert abs(got.gamma + 0.25) < 0.01


def test_two_texts_asked_both_ways_separate_order_from_position():
    # The first-shown text is favoured by 0.4 logits; text 1 is truly 1.0 above text 0.
    y = 1 / (1 + np.exp(-np.array([-1.0 + 0.4, 1.0 + 0.4])))
    got = fit(2, [0, 1], [1, 0], y)
    assert got.theta[1] > got.theta[0]
    assert abs((got.theta[1] - got.theta[0]) - 1.0) < 0.05
    assert abs(got.gamma - 0.4) < 0.05


def test_certain_answers_stay_finite():
    got = fit(3, [0, 0, 1], [1, 2, 2], [1.0, 1.0, 1.0])
    assert np.all(np.isfinite(got.theta))
    assert got.theta[0] > got.theta[1] > got.theta[2]


def test_no_comparisons_is_a_flat_scale():
    got = fit(4, [], [], [])
    assert got.converged and np.all(got.theta == 0)
    assert np.all(np.isnan(standard_errors(4, [], [], [], got)))


def test_standard_errors_grow_with_contradiction_and_shrink_with_data():
    theta = np.linspace(-2, 2, 12)
    clean = all_pairs(theta)
    noisy = all_pairs(theta, noise=1.0, seed=1)
    se_clean = standard_errors(12, *clean, fit(12, *clean))
    se_noisy = standard_errors(12, *noisy, fit(12, *noisy))
    assert np.all(se_clean < 0.01)                       # answers that sit on one scale pin it down
    assert np.all(se_noisy > 5 * se_clean.max())
    half = tuple(a[::2] for a in noisy)
    assert standard_errors(12, *half, fit(12, *half)).mean() > se_noisy.mean()


def test_probed_standard_errors_match_the_exact_ones(monkeypatch):
    theta = np.random.default_rng(0).normal(0, 1.5, 60)
    first, second, y = all_pairs(theta, noise=0.8, seed=2)
    keep = np.random.default_rng(1).random(len(y)) < 0.15          # sparse, as a real run is
    data = (first[keep], second[keep], y[keep])
    fitted = fit(60, *data)
    exact = standard_errors(60, *data, fitted)
    monkeypatch.setattr("jsort.model.EXACT_LIMIT", 10)
    probed = standard_errors(60, *data, fitted)
    assert np.all(np.isfinite(probed)) and 0.93 < probed.mean() / exact.mean() < 1.07
    assert np.corrcoef(probed, exact)[0, 1] > 0.9


def test_reliability_tells_a_scale_from_noise():
    theta = np.random.default_rng(0).normal(0, 1.5, 30)
    first, second, y = all_pairs(theta, noise=0.5, seed=3)
    assert reliability(30, first, second, y) > 0.95
    coin = np.random.default_rng(4).uniform(0, 1, len(y))
    assert reliability(30, first, second, coin) < 0.5
    assert reliability(3, [0], [1], [0.7]) is None


def test_shortfall_separates_the_hopeless_from_the_contenders():
    theta = np.array([-3.0, 0.0, 2.9, 3.0])
    first, second, y = all_pairs(theta)
    fitted = fit(4, first, second, y)
    drop = shortfall(4, first, second, y, fitted, bar=float(fitted.theta[3]))
    assert drop[3] == 0
    assert drop[0] > drop[1] > drop[2] >= 0
    assert drop[0] > 4 and drop[2] < 0.5


def test_information_counts_close_matches_more_than_lopsided_ones():
    fitted = Fit(np.array([0.0, 0.1, 6.0]), 0.0, True)
    info = information(3, [0, 0], [1, 2], fitted)
    close_only = information(3, [0], [1], fitted)
    assert info[1] > info[2]
    assert info[0] - close_only[0] < 0.01


def test_reliability_ignores_pieces_that_no_comparison_joins():
    # Two groups of texts, compared thoroughly within but never across. Where one group sits relative
    # to the other is not data, so it must not count for or against the scale.
    rng = np.random.default_rng(5)
    theta = rng.normal(0, 1.5, 24)
    first, second, y = all_pairs(theta, noise=0.3, seed=6)
    within = (first < 14) == (second < 14)
    got = reliability(24, first[within], second[within], y[within])
    assert got is not None and got > 0.95
    lonely = (first < 3) & (second < 3)                       # too little joined up to say anything
    assert reliability(24, first[lonely], second[lonely], np.tile(y[lonely], 1)) is None


def test_fit_does_not_claim_convergence_it_did_not_reach():
    theta = np.linspace(-2, 2, 8)
    first, second, y = all_pairs(theta)
    assert fit(8, first, second, y).converged
    assert not fit(8, first, second, y, max_iter=1).converged


def test_placing_against_fixed_anchors_is_the_joint_fit_in_one_coordinate():
    # Hold every other text where the joint fit left it, and the lean too: the one-parameter optimum is the joint one.
    theta = np.linspace(-2.5, 2.5, 11)
    first, second, y = all_pairs(theta, gamma=-0.2, noise=0.4, seed=7)
    joint = fit(11, first, second, y)
    joint_se = standard_errors(11, first, second, y, joint)
    for text in (0, 4, 10):
        met = (first == text) | (second == text)
        leads = first[met] == text
        anchors = np.where(leads, joint.theta[second[met]], joint.theta[first[met]])
        score, se = place(anchors, leads, y[met], joint.gamma)
        assert abs(score - joint.theta[text]) < 1e-5
        assert 0.85 < se / joint_se[text] < 1.15  # and the error is the one the fit reports, to within a few percent


def test_placement_uses_the_lean_and_the_probabilities():
    anchors, lean = np.array([-1.0, 0.0, 1.0, 2.0]), 0.4
    leads = np.array([True, False, True, False])
    y = 1 / (1 + np.exp(-(np.where(leads, 1, -1) * (0.7 - anchors) + lean)))
    score, se = place(anchors, leads, y, lean)
    assert abs(score - 0.7) < 0.01 and se < 0.01           # answers that sit on the scale pin the text down
    assert abs(place(anchors, leads, y, 0.0)[0] - 0.7) < 0.05 < abs(place(anchors, ~leads, y, 0.0)[0] - 0.7)
    wins = (y > 0.5).astype(float)                           # the same answers as wins and losses say less
    assert abs(place(anchors, leads, wins, lean)[0] - 0.7) > 0.1


def test_a_text_beyond_every_anchor_is_placed_finitely():
    anchors = np.linspace(-3, 3, 8)
    leads = np.arange(8) % 2 == 0
    above, se = place(anchors, leads, np.where(leads, 1.0, 0.0), -0.1)       # beat every anchor, with certainty
    below, _ = place(anchors, leads, np.where(leads, 0.0, 1.0), -0.1)
    assert np.isfinite([above, below, se]).all() and 3 < above < 15 and -15 < below < -3
    assert all(np.isnan(place([], [], [], 0.0)))
