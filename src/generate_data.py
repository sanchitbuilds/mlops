"""Create a small synthetic login dataset (Experiment 1: data collection)."""
import numpy as np
import pandas as pd

def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "hour": rng.integers(0, 24, n),                 # hour of login (0-23)
        "failed_attempts": rng.poisson(0.7, n),         # wrong passwords before success
        "new_device": rng.integers(0, 2, n),            # 1 = device never seen before
        "foreign_country": (rng.random(n) < 0.15).astype(int),
        "vpn_used": (rng.random(n) < 0.2).astype(int),
        "hours_since_last_login": rng.exponential(24, n).round(1),
    })
    # Hidden rule that decides what is "suspicious" (+ noise)
    score = (
        2.0 * (df.failed_attempts >= 3)
        + 1.5 * df.new_device
        + 2.0 * df.foreign_country
        + 1.0 * df.vpn_used
        + 1.0 * df.hour.isin([0, 1, 2, 3, 4, 5])
        + 0.5 * (df.hours_since_last_login < 0.5)
        + rng.normal(0, 0.7, n)
    )
    df["is_anomaly"] = (score >= 3.5).astype(int)
    # Make some values missing so preprocessing has something to do
    for col in ["hours_since_last_login", "failed_attempts"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df

if __name__ == "__main__":
    df = make_data()
    df.to_csv("data/logins.csv", index=False)
    print(df.shape, "anomaly rate:", round(df.is_anomaly.mean(), 3))
