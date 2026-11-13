"""Naive baselines. Each takes the pairs frame and returns predicted km at the target date."""
import pandas as pd

from .series import KEY

SEG = ["vehicle_level", "vehicle_type"]


def b0_carry_forward(p: pd.DataFrame) -> pd.Series:
    """Last known reading, unchanged."""
    return p.km0


def b1_cumulative_rate(p: pd.DataFrame) -> pd.Series:
    """Last reading + the vehicle's average km/day since its first reading x gap (starter-like)."""
    return p.km0 + p.cum_rate.fillna(0) * p.gap


def b1b_recent_rate(p: pd.DataFrame) -> pd.Series:
    """Last reading + average km/day over the last 90 days (falls back to cumulative, then 0)."""
    return p.km0 + p.rate90.fillna(p.cum_rate).fillna(0) * p.gap


def fit_segment_rates(train_tl: pd.DataFrame) -> tuple[pd.Series, float]:
    """Pooled km/day per segment = sum(delta km) / sum(delta days) over consecutive observations."""
    g = train_tl.groupby(KEY)
    d = train_tl.assign(dkm=g.km.diff(), dday=g.day.diff()).dropna(subset=["dkm", "dday"])
    d = d[(d.dday > 0) & (d.dkm >= 0)]
    seg = d.groupby(SEG).apply(lambda s: s.dkm.sum() / s.dday.sum(), include_groups=False)
    return seg, d.dkm.sum() / d.dday.sum()


def b2_segment_rate(p: pd.DataFrame, seg_rates: pd.Series, global_rate: float) -> pd.Series:
    """Last reading + segment pooled rate x gap (cold-start-friendly)."""
    idx = pd.MultiIndex.from_frame(p[SEG])
    rate = pd.Series(seg_rates.reindex(idx).to_numpy(), index=p.index).fillna(global_rate)
    return p.km0 + rate * p.gap
