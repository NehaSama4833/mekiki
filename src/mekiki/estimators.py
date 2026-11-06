"""Label-free accuracy estimators from model confidence. All fitted only on SOURCE validation data."""
from __future__ import annotations
import numpy as np


def max_conf(P): return P.max(1)
def neg_entropy(P): return (P * np.log(np.clip(P, 1e-12, 1))).sum(1)


class Estimators:
    def __init__(self, P_src_val, y_src_val):
        self.acc_src = float((P_src_val.argmax(1) == y_src_val).mean())
        self.conf_src = float(max_conf(P_src_val).mean())
        self.t_conf = self._atc_threshold(max_conf(P_src_val))
        self.t_ent = self._atc_threshold(neg_entropy(P_src_val))

    def _atc_threshold(self, s):
        """threshold t such that the fraction of source scores above t equals source accuracy."""
        return float(np.quantile(s, 1 - self.acc_src))

    def estimate(self, P):
        return {
            "source accuracy (assume no shift)": self.acc_src,
            "average confidence": float(max_conf(P).mean()),
            "difference of confidences": float(np.clip(self.acc_src - (self.conf_src - max_conf(P).mean()), 0, 1)),
            "ATC (max confidence)": float((max_conf(P) > self.t_conf).mean()),
            "ATC (negative entropy)": float((neg_entropy(P) > self.t_ent).mean()),
        }
