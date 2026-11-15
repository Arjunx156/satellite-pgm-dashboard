"""Section 11 of the notebook: Decision Network / influence diagram.

EU(a | E) = sum_s P(s | E) U(a, s),   a* = argmax_a EU(a | E).
The utilities are project decision-support assumptions, not real spacecraft commands.
"""

from fractions import Fraction

import pandas as pd

from . import config


def utility_table() -> pd.DataFrame:
    return pd.DataFrame([config.UTILITY[a] for a in config.ACTIONS],
                        index=config.ACTIONS, columns=config.STATES)


def expected_utilities(p_anomaly: float) -> dict:
    p_nominal = 1.0 - p_anomaly
    return {a: p_nominal * u_nom + p_anomaly * u_anom for a, (u_nom, u_anom) in config.UTILITY.items()}


def recommend(p_anomaly: float):
    eu = expected_utilities(p_anomaly)
    return max(eu, key=eu.get), eu


def action_regions():
    """Exact P(Anomaly) intervals where each action has the highest expected utility.

    Each EU is linear in p, so the regions are bounded by the pairwise crossing points.
    """
    cuts = {Fraction(0), Fraction(1)}
    acts = list(config.UTILITY.items())
    for i, (_, (n1, a1)) in enumerate(acts):
        for _, (n2, a2) in acts[i + 1:]:
            slope = (a1 - n1) - (a2 - n2)
            if slope:
                p = Fraction(n2 - n1, slope)
                if 0 < p < 1:
                    cuts.add(p)
    cuts = sorted(cuts)
    regions = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        best, _ = recommend(float((lo + hi) / 2))
        if regions and regions[-1][0] == best:
            regions[-1] = (best, regions[-1][1], hi)
        else:
            regions.append((best, lo, hi))
    return regions          # [(action, lower Fraction, upper Fraction), ...]


# Risk bands are the Decision Network's own action regions renamed, so they are derived from the
# utility table rather than invented. They are not validated spacecraft safety thresholds.
BAND_FOR_ACTION = {"Continue": "Low risk", "Monitor": "Moderate risk", "Safe Mode": "High risk"}


def risk_band(p_anomaly: float) -> str:
    return BAND_FOR_ACTION[recommend(p_anomaly)[0]]
