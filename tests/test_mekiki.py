import numpy as np
from mekiki.estimators import Estimators, max_conf, neg_entropy
from mekiki.shifts import FEATURE_SHIFTS, label_shift, noise, rotation


def probs(conf, n=100):
    p = np.full((n, 3), (1 - conf) / 2); p[:, 0] = conf; return p


def test_atc_threshold_hand_case():
    # source: 10 samples, accuracy 0.6 -> threshold at the 40th percentile of confidences
    P = np.zeros((10, 2)); conf = np.linspace(0.5, 0.95, 10); P[:, 0] = conf; P[:, 1] = 1 - conf
    y = np.array([0] * 6 + [1] * 4)          # argmax always 0 -> accuracy 0.6
    e = Estimators(P, y)
    assert e.acc_src == 0.6
    assert (max_conf(P) > e.t_conf).mean() == 0.6


def test_estimators_return_source_accuracy_when_target_equals_source():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1], 400); y = rng.integers(0, 3, 400)
    e = Estimators(P, y).estimate(P)
    assert abs(e["ATC (max confidence)"] - e["source accuracy (assume no shift)"]) < 0.01
    assert abs(e["difference of confidences"] - e["source accuracy (assume no shift)"]) < 1e-9


def test_low_confidence_target_lowers_confidence_based_estimates():
    P_src = probs(0.9); y = np.zeros(100, int)
    e = Estimators(P_src, y).estimate(probs(0.4))
    assert e["average confidence"] < 0.5 and e["difference of confidences"] < e["source accuracy (assume no shift)"]


def test_entropy_score_orders_sharp_above_flat():
    assert neg_entropy(probs(0.98))[0] > neg_entropy(probs(0.4))[0]


def test_shifts_change_data_within_range_and_label_shift_resamples_classes():
    X = np.random.default_rng(0).uniform(0, 16, (50, 64))
    for fam, (fn, sevs) in FEATURE_SHIFTS.items():
        Z = fn(X, np.random.default_rng(1), sevs[-1])
        assert Z.shape == X.shape and Z.min() >= 0 and Z.max() <= 16 and not np.allclose(Z, X), fam
    y = np.arange(50) % 10
    _, ys = label_shift(X, y, np.random.default_rng(0), 0.9)
    assert abs((ys < 5).mean() - 0.9) < 1e-9
    assert np.allclose(rotation(X, None, 0), X, atol=1e-6)
