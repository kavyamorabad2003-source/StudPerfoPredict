"""Synthetic stand-in with the same columns as the UCI Student Performance dataset.

Only used when data/student-mat.csv is missing, so the app runs out of the box.
Replace it with the real dataset (see README.md) for real results.
"""
import numpy as np
import pandas as pd


def make_sample(n: int = 395, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)  # hidden "ability"

    def pick(options, p=None):
        return rng.choice(options, n, p=p)

    medu = np.clip(np.round(2.6 + 0.5 * a + rng.normal(0, 1, n)), 0, 4).astype(int)
    fedu = np.clip(np.round(0.6 * medu + 1.1 + rng.normal(0, 0.9, n)), 0, 4).astype(int)
    failures = np.clip(np.round(-0.5 * a + rng.normal(0.2, 0.7, n)), 0, 3).astype(int)
    studytime = np.clip(np.round(2.0 + 0.4 * a + rng.normal(0, 0.8, n)), 1, 4).astype(int)
    higher = np.where(rng.random(n) < 1 / (1 + np.exp(-(2.8 + 1.0 * a))), "yes", "no")
    absences = np.clip(np.round(rng.gamma(1.4, 4.0, n) - 0.8 * a), 0, 40).astype(int)
    absences[rng.random(n) < 0.3] = 0
    goout = np.clip(np.round(3 + rng.normal(0, 1.1, n)), 1, 5).astype(int)
    dalc = np.clip(np.round(1.4 + rng.exponential(0.6, n)), 1, 5).astype(int)
    walc = np.clip(np.round(dalc + rng.normal(0.9, 1, n)), 1, 5).astype(int)

    df = pd.DataFrame({
        "school": pick(["GP", "MS"], [0.88, 0.12]),
        "sex": pick(["F", "M"], [0.53, 0.47]),
        "age": np.clip(np.round(16.7 + failures * 0.5 + rng.normal(0, 1.1, n)), 15, 22).astype(int),
        "address": pick(["U", "R"], [0.78, 0.22]),
        "famsize": pick(["GT3", "LE3"], [0.71, 0.29]),
        "Pstatus": pick(["T", "A"], [0.9, 0.1]),
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": pick(["at_home", "health", "other", "services", "teacher"], [0.15, 0.09, 0.36, 0.26, 0.14]),
        "Fjob": pick(["at_home", "health", "other", "services", "teacher"], [0.05, 0.05, 0.55, 0.28, 0.07]),
        "reason": pick(["course", "home", "other", "reputation"], [0.37, 0.28, 0.09, 0.26]),
        "guardian": pick(["mother", "father", "other"], [0.69, 0.23, 0.08]),
        "traveltime": np.clip(np.round(1.4 + rng.exponential(0.4, n)), 1, 4).astype(int),
        "studytime": studytime,
        "failures": failures,
        "schoolsup": pick(["yes", "no"], [0.13, 0.87]),
        "famsup": pick(["yes", "no"], [0.61, 0.39]),
        "paid": pick(["yes", "no"], [0.46, 0.54]),
        "activities": pick(["yes", "no"]),
        "nursery": pick(["yes", "no"], [0.8, 0.2]),
        "higher": higher,
        "internet": pick(["yes", "no"], [0.83, 0.17]),
        "romantic": pick(["yes", "no"], [0.33, 0.67]),
        "famrel": np.clip(np.round(4 + rng.normal(0, 0.9, n)), 1, 5).astype(int),
        "freetime": np.clip(np.round(3.2 + rng.normal(0, 1, n)), 1, 5).astype(int),
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": np.clip(np.round(3.5 + rng.normal(0, 1.3, n)), 1, 5).astype(int),
        "absences": absences,
    })

    base = (10.8 + 2.9 * a + 0.5 * (studytime - 2) - 1.0 * failures
            + 0.8 * (higher == "yes") + 0.25 * (medu - 2.5) - 0.12 * (goout - 3)
            - 0.15 * (dalc - 1))
    g1 = np.clip(np.round(base + rng.normal(0, 1.4, n)), 3, 19).astype(int)
    g2 = np.clip(np.round(g1 + rng.normal(0, 1.3, n)), 0, 19).astype(int)
    g3 = np.clip(np.round(g2 + rng.normal(0.2, 1.3, n) - 0.03 * absences), 0, 20).astype(int)
    dropout = (g2 <= 9) & (rng.random(n) < 0.25)  # in the real data, G3 = 0 means dropped out
    g3[dropout] = 0
    df["G1"], df["G2"], df["G3"] = g1, g2, g3
    return df
