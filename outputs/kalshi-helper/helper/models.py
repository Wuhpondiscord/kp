"""Small distributional baseline. Inputs must already match settlement rules."""
from math import isfinite
from statistics import NormalDist


def weather_probability(mean, std, lower=None, upper=None):
    """P(lower <= X < upper). Caller encodes the reporting/rounding boundaries.

    Example: reported integer 70..74 F corresponds to latent [69.5,74.5)
    ONLY when the actual contract uses nearest-integer rounding.
    No automatic contract parsing or station identification is attempted.
    """
    mean, std = float(mean), float(std)
    if not isfinite(mean) or not isfinite(std) or std <= 0:
        raise ValueError("Weather mean must be finite; standard deviation must be positive")
    if lower is not None and not isfinite(float(lower)):
        raise ValueError("Use null for an unbounded lower endpoint")
    if upper is not None and not isfinite(float(upper)):
        raise ValueError("Use null for an unbounded upper endpoint")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("Lower endpoint must be below upper endpoint")
    distribution = NormalDist(mean, std)
    left = distribution.cdf(lower) if lower is not None else 0
    right = distribution.cdf(upper) if upper is not None else 1
    return max(0, min(1, right-left))


def weather_prediction(forecast):
    """Convert a reviewed forecast row into the common prediction interface."""
    for field in ("station", "rule_reference", "model_id"):
        if not forecast.get(field):
            raise ValueError(f"Weather row requires {field}")
    return dict(type="prediction", ticker=forecast["ticker"],
                available_at=forecast["available_at"], made_at=forecast["made_at"],
                features_available_at=forecast["features_available_at"],
                expires_at=forecast["expires_at"], model_id=forecast["model_id"],
                synthetic=bool(forecast.get("synthetic", False)),
                probability=weather_probability(forecast["mean"], forecast["std"],
                                                forecast.get("lower"), forecast.get("upper")),
                provenance=forecast)
