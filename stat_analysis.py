from scipy.stats import kruskal, mannwhitneyu
import pandas as pd
import itertools


def kruskal_test(df, feature, group_col="label"):
    """
    Perform Kruskal-Wallis test across all groups.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Feature column to analyze.
    group_col : str
        Column containing class labels.

    Returns
    -------
    dict
    """

    groups = [
        group[feature].dropna()
        for _, group in df.groupby(group_col)
    ]

    group_names = list(df[group_col].unique())

    stat, p = kruskal(*groups)

    return {
        "feature": feature,
        "groups": group_names,
        "H": stat,
        "p": p,
        "significant": p < 0.05,
    }


def pairwise_mannwhitney(df, feature, group_col="label"):
    """
    Perform all pairwise Mann-Whitney U tests.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
    group_col : str

    Returns
    -------
    pd.DataFrame
    """

    labels = sorted(df[group_col].dropna().unique())

    results = []

    for g1, g2 in itertools.combinations(labels, 2):

        vals1 = df[df[group_col] == g1][feature].dropna()
        vals2 = df[df[group_col] == g2][feature].dropna()

        if len(vals1) == 0 or len(vals2) == 0:
            continue

        U, p = mannwhitneyu(
            vals1,
            vals2,
            alternative="two-sided"
        )

        results.append({
            "feature": feature,
            "group1": g1,
            "group2": g2,
            "n1": len(vals1),
            "n2": len(vals2),
            "U": U,
            "p": p,
            "significant": p < 0.05,
        })

    return pd.DataFrame(results)