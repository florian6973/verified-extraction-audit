import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.special import kl_div
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.spatial import cKDTree
from scipy.special import digamma
import random
from scipy.stats import rankdata, skewnorm, kstest, t
from scipy.optimize import minimize
import argparse
import os
import joblib

# Try to import scipy's jf_skew_t (available in scipy >= 1.13.0)
try:
    from scipy.stats import jf_skew_t
    HAS_JF_SKEWT = True
except ImportError:
    HAS_JF_SKEWT = False

# Try to import external skewt package as fallback
try:
    from skewt import SkewT
    HAS_SKEWT = True
except ImportError:
    HAS_SKEWT = False


from tqdm import tqdm
# need load mia df

# Helper functions for plot labels
def format_pii_rate(pii_rate):
    """Convert PII rate to percentage string."""
    return f"{pii_rate*100:.0f}%"

def format_epoch_label(n_epochs):
    """Convert epoch number to overfitting label."""
    return "overfitting" if n_epochs == 10 else "no overfitting"

def format_pii_type_for_title(pii_type):
    """Convert PII type to display name for titles."""
    if pii_type == 'name-patient':
        return 'Name'
    elif pii_type == 'unit_no':
        return 'MRN'
    else:
        return pii_type

# metrics

# attacks

def check_no_prefix(df, col):
    # df = df.sort_values('value')
    no_prefix = all(not any(other.startswith(val) for other in df[col][i+1:])
                for i, val in enumerate(df[col]))
    # print(no_prefix)
    # assert no_prefix
    return no_prefix
def purge_prefixes(df):
    # col = 'value'
    col = 'list_tokens' # the only one that matters
    df = df.sort_values(col)
    no_prefix = check_no_prefix(df, col)
    # print(no_prefix)
    prefixes = {a for i, a in enumerate(df[col]) for b in df[col][i+1:] if b.startswith(a)}
    print(prefixes)
    assert len(prefixes) == 0

    df_no_prefix = df[~df[col].isin(prefixes)]
    # prefixes = {a for i, a in enumerate(df_no_prefix['value']) for b in df_no_prefix['value'][i+1:] if b.startswith(a)}
    # print(prefixes)
    no_prefix = check_no_prefix(df_no_prefix, col)

    assert no_prefix
    return df_no_prefix


def mi_per_name_knn_ross(M, S, k=5, names=None, return_bits=False, jitter=0.0, seed=None):
    """
    Per-name decomposition of the Ross (2014) KNN MI estimator for discrete M and continuous S.

    Parameters
    ----------
    M : array-like, shape (n,)
        Discrete labels (e.g., 0/1). Can be ints/strings; will be internally encoded.
    S : array-like, shape (n, d) or (n,)
        Continuous scores/features. If 1D, reshaped to (n,1).
    k : int
        k-th neighbor within SAME-CLASS used to set local radius r_i. Must be < min class size.
    names : array-like, optional, shape (n,)
        Names/IDs for each row; if provided, a structured output dict is also returned.
    return_bits : bool
        If True, converts nats -> bits.
    jitter : float
        If >0, adds Gaussian noise with std=jitter to S to break ties (useful if many duplicates).
    seed : int or None
        RNG seed for jitter.

    Returns
    -------
    c : np.ndarray, shape (n,)
        Per-sample contributions (nats or bits).
    I_total : float
        Mean of c (i.e., estimated MI).
    out_table : list of tuples (optional)
        Only if `names` is provided: [(name, c_i, label, m_i, N_xi), ...] sorted by c_i desc.
    """

    from npeet import entropy_estimators as ee
    S = np.log(S)
    # S = rankdata(S, axis=0) / len(S)
    # M = rankdata(M) / len(M)
    mi = ee.mi(S, M.reshape(-1,1))  # continuous (S) vs discrete (M)
    # print("Estimated MI:", mi, "nats")
    return None, mi, None
    # if names is None:
        # names = np.arange(len(M))
    # Inputs
    M = np.asarray(M)
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    n = S.shape[0]
    if len(M) != n:
        raise ValueError("M and S must have the same number of rows.")
    if names is not None and len(names) != n:
        raise ValueError("`names` length must match M/S length.")

    # Optional jitter for exact duplicates / ties
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        S = S + rng.normal(scale=jitter, size=S.shape)

    # Encode labels to consecutive ints
    uniq, inv = np.unique(M, return_inverse=True)  # inv[i] = class index of sample i
    label = inv
    class_counts = np.bincount(label)
    if np.any(class_counts <= 1):
        raise ValueError("Each class must have at least 2 samples.")
    # Ensure k is valid for every class (since we exclude self, max neighbor index is class_size-1)
    k_max = int(class_counts.min() - 1)
    if k > k_max:
        k = max(1, k_max)  # shrink if needed
        if k < 1:
            raise ValueError("k must be >=1 and < min class size.")

    # Build KD-tree on all S for the m_i counts (∞-norm as in KSG/Ross)
    tree_all = cKDTree(S)

    # Compute r_i: k-th neighbor distance WITHIN SAME CLASS (∞-norm) for each sample
    r = np.empty(n, dtype=float)
    for c in range(len(uniq)):
        idx = np.where(label == c)[0]
        S_c = S[idx]
        tree_c = cKDTree(S_c)
        # kth neighbor distance; query k+1 to skip self at index 0
        dists, _ = tree_c.query(S_c, k=k+1, p=np.inf)
        r[idx] = dists[:, k]

    # m_i: number of ALL points within radius r_i (inclusive), excluding self
    r_inclusive = np.nextafter(r, np.inf)  # include boundary reliably
    m = np.empty(n, dtype=int)
    for i in range(n):
        # Count neighbors within radius, exclude self
        m[i] = len(tree_all.query_ball_point(S[i], r_inclusive[i], p=np.inf)) - 1
        # Safety: m[i] should be >= k by construction (since class-nn distance <= all-nn distance)
        if m[i] < 1:
            m[i] = 1  # avoid digamma(0); shouldn't happen, but just in case

    # Assemble per-sample contributions: c_i = [ψ(n) - ψ(N_xi)] + [ψ(k) - ψ(m_i)]
    psi_n = digamma(n)
    psi_k = digamma(k)
    psi_Nx = digamma(class_counts[label])  # same for all in a class
    psi_m = digamma(m)

    c = (psi_n - psi_Nx) + (psi_k - psi_m)  # nats
    I_total = float(np.mean(c))
    if I_total < 0:
        I_total = 0.0  # clip tiny negatives from numerical noise
    if return_bits:
        c = c / np.log(2.0)
        I_total = I_total / np.log(2.0)

    # Optional per-name table
    out_table = None
    if names is not None:
        # helpful to inspect: (name, contribution, label, m_i, N_x_i)
        N_x_i = class_counts[label]
        rows = [(names[i], float(c[i]), M[i], int(m[i]), int(N_x_i[i])) for i in range(n)]
        rows.sort(key=lambda t: t[1], reverse=True)
        out_table = rows


    # print(f"Total I(M;S) KSG: {I_total:.6f} bits")
    if False:
        print("Top 5 names by contribution:")
        for row in out_table[:5]:
            print(row)
    if "Cqklzdhvuytq Fxjbdjkhsuaf" in names:
        # print(out_table[names.index("Cqklzdhvuytq Fxjbdjkhsuaf")])÷
        out_table_canary = [row for row in out_table if row[0] == "Cqklzdhvuytq Fxjbdjkhsuaf"]
        print(out_table_canary)
    # out_table_members = [row for row in out_table if row[2] == 1]
    # for row in out_table_members[:5]:
    #     print(row)
    return c, I_total, out_table

def attack(M, S, return_proba=False):
    if len(S.shape) == 1:
        S = S.reshape(-1, 1)

    S = np.log(S)

    # print(S.shape, M.shape)
    # X_train, X_test, y_train, y_test = train_test_split(S, M, test_size=0.9, random_state=42)

    n = S.shape[0]
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx, test_size=0.9, random_state=42, shuffle=True
    )

    X_train, X_test = S[idx_train], S[idx_test]
    y_train, y_test = M[idx_train], M[idx_test]


    # if S.shape[1] == 1:
    #     y_pred_proba = np.abs(X_test)/np.abs(X_test).sum()
    # else:
    if True:
        clf = LogisticRegression(random_state=42) #, C=0.0)
        clf.fit(X_train, y_train)
        # print(clf.coef_, clf.intercept_)
        # auc_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        # print("AUC train: ", auc_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    # print(auc)
    if return_proba:
        return auc, X_test, y_test, y_pred_proba, idx_test, clf
    else:
        return auc


def prepare_data(df, n_samples=None):
    df_val = df[df['group'] == 0]
    df_train = df[df['group'] == 1]
    if n_samples is None:
        n_samples = min(len(df_val), len(df_train))
    df_val = df_val.sample(n=min(len(df_val), len(df_train), n_samples))
    df_train = df_train.sample(n=min(len(df_val), len(df_train), n_samples))
    df = pd.concat([df_val, df_train])
    # print(len(df_val), len(df_train), len(df))

    # renormalize all columns
    # not need to renormalize because loss is just a score
    # for col in df.columns:
    #     if col.startswith('p_'):
    #         df[col] = df[col]/df[col].sum()

    return df

def get_prompts(pii_type):
    if pii_type == 'name-patient':
        return ['Name: ', 'Patient: ']
    elif pii_type == 'unit_no':
        return ['MRN: ', 'Patient #: ']
    elif pii_type == 'name-attending':
        return ['Name: ', 'Attending: ']
    else:
        raise ValueError(f"Invalid pii_type: {pii_type}")

def resample_from_skewt_fit(data, n_samples=10000, use_skewt=True, random_state=None):
    """
    Fit a skew t or skew normal distribution to data and resample from it.
    
    Parameters
    ----------
    data : array-like
        Data to fit distribution to
    n_samples : int, default 10000
        Number of samples to generate
    use_skewt : bool, default True
        If True, fit skew t distribution; otherwise fit skew normal
    random_state : int or None, default None
        Random seed for reproducibility
        
    Returns
    -------
    resampled_data : numpy.ndarray
        Resampled data from the fitted distribution
    """
    data = np.asarray(data)
    if len(data) < 3:
        # If too few points, return original data repeated
        return np.tile(data, (n_samples // len(data) + 1))[:n_samples]
    
    # Remove any invalid values before fitting
    data = data[data > 0]  # Only positive values since we'll take log
    if len(data) < 3:
        return np.tile(data, (n_samples // len(data) + 1))[:n_samples] if len(data) > 0 else np.ones(n_samples) * 1e-10
    
    log_data = np.log(data)
    # Get bounds for clipping to prevent overflow
    log_min = np.min(log_data)
    log_max = np.max(log_data)
    # Add some padding but prevent extreme values
    log_range = log_max - log_min
    clip_min = log_min - 5 * log_range if log_range > 0 else log_min - 50
    clip_max = log_max + 5 * log_range if log_range > 0 else log_max + 50
    # Also ensure reasonable absolute bounds to prevent overflow in exp
    clip_min = max(clip_min, -700)  # exp(-700) is very small but not zero
    clip_max = min(clip_max, 700)  # exp(700) is large but not infinite
    
    # Fit distribution
    if use_skewt:
        params, _, _, dist_type, skewt_impl = fit_skewnorm_distribution(data, use_skewt=True)
        if params is not None and dist_type == 'skewt':
            if skewt_impl == 'jf_skew_t' and HAS_JF_SKEWT:
                a, b, loc, scale = params
                # Sample from jf_skew_t
                resampled_log = jf_skew_t.rvs(a=a, b=b, loc=loc, scale=scale, size=n_samples, random_state=random_state)
                # Clip to prevent overflow
                resampled_log = np.clip(resampled_log, clip_min, clip_max)
            elif skewt_impl == 'external_skewt' and HAS_SKEWT:
                df, loc, scale, skew = params
                skewt_dist = SkewT()
                # Sample from external skewt (using rejection sampling or approximation)
                # For now, use a simple approach: sample from normal with skew adjustment
                try:
                    # Try with random_state if supported
                    try:
                        resampled_log = skewt_dist.rvs(df=df, loc=loc, scale=scale, skew=skew, size=n_samples, random_state=random_state)
                        # Clip to prevent overflow
                        resampled_log = np.clip(resampled_log, clip_min, clip_max)
                    except TypeError:
                        # If random_state not supported, set seed before calling
                        if random_state is not None:
                            np.random.seed(random_state)
                        resampled_log = skewt_dist.rvs(df=df, loc=loc, scale=scale, skew=skew, size=n_samples)
                        # Clip to prevent overflow
                        resampled_log = np.clip(resampled_log, clip_min, clip_max)
                except:
                    # Fallback to skew normal if rvs not available
                    params, _, _, _, _ = fit_skewnorm_distribution(data, use_skewt=False)
                    if params is not None:
                        a, loc, scale = params
                        resampled_log = skewnorm.rvs(a=a, loc=loc, scale=scale, size=n_samples, random_state=random_state)
                        resampled_log = np.clip(resampled_log, clip_min, clip_max)
                    else:
                        return np.tile(data, (n_samples // len(data) + 1))[:n_samples]
            else:
                # Fallback to skew normal
                params, _, _, _, _ = fit_skewnorm_distribution(data, use_skewt=False)
                if params is not None:
                    a, loc, scale = params
                    resampled_log = skewnorm.rvs(a=a, loc=loc, scale=scale, size=n_samples, random_state=random_state)
                    resampled_log = np.clip(resampled_log, clip_min, clip_max)
                else:
                    return np.tile(data, (n_samples // len(data) + 1))[:n_samples]
        else:
            # Fallback to skew normal
            params, _, _, _, _ = fit_skewnorm_distribution(data, use_skewt=False)
            if params is not None:
                a, loc, scale = params
                resampled_log = skewnorm.rvs(a=a, loc=loc, scale=scale, size=n_samples, random_state=random_state)
                resampled_log = np.clip(resampled_log, clip_min, clip_max)
            else:
                return np.tile(data, (n_samples // len(data) + 1))[:n_samples]
    else:
        # Fit skew normal
        params, _, _, _, _ = fit_skewnorm_distribution(data, use_skewt=False)
        if params is not None:
            a, loc, scale = params
            resampled_log = skewnorm.rvs(a=a, loc=loc, scale=scale, size=n_samples, random_state=random_state)
            resampled_log = np.clip(resampled_log, clip_min, clip_max)
        else:
            return np.tile(data, (n_samples // len(data) + 1))[:n_samples]
    
    # Check for invalid values before exp
    if not np.all(np.isfinite(resampled_log)):
        resampled_log = np.nan_to_num(resampled_log, nan=log_min, posinf=clip_max, neginf=clip_min)
        resampled_log = np.clip(resampled_log, clip_min, clip_max)
    
    # Convert back from log space to original space
    resampled_data = np.exp(resampled_log)
    
    # Check for invalid values after exp
    if not np.all(np.isfinite(resampled_data)):
        resampled_data = np.nan_to_num(resampled_data, nan=np.exp(log_min), posinf=np.exp(clip_max), neginf=np.exp(clip_min))
        # Ensure all values are positive and finite
        resampled_data = np.clip(resampled_data, np.exp(clip_min), np.exp(clip_max))
    
    return resampled_data

def compute_mi(df, pii_type, use_skewt=False):
    df['group'] = df.apply(lambda x: 0 if x['split'] == 'val' else 1, axis=1)
    # print(df)
    # print(df.shape)
    # print()

    # sample sample number of rows for each group
    

    # knobs
    # ratio pi between groups
    # prompts
    # number of datapoints

    models = ['p_pre', 'p_ft']
    prompts = get_prompts(pii_type)

    # print(df.columns)

    df_results = pd.DataFrame(columns=['model', 'prompt', 'I', 'I_std', 'auc', 'auc_std'])
    df_results_combined = pd.DataFrame(columns=['prompt', 'I', 'I_std', 'auc', 'auc_std'])

    factor = 1.96
    # individual mi
    for prompt in prompts:
        for model in models:
            df_temp = df[['value', 'group', model + '_' + prompt]].copy()
            col_name = model + '_' + prompt
            
            i_values = []
            auc_values = []
            np.random.seed(42)
            random.seed(42)
            # print(prompt, model)
            for i in range(100): # can boostrap at subsampling too
                df_temp_sub = prepare_data(df_temp)

                # print(df_temp_sub[col_name].isna().sum())
                # print(df_temp)
                # print(df_temp_sub)
                
                # For AUC: use original data
                auc = attack(df_temp_sub['group'].values, df_temp_sub[col_name].values)
                auc_values.append(auc)
                
                # For MI: resample if requested
                if use_skewt:
                    # Fit skew t for each group and resample
                    df_val_sub = df_temp_sub[df_temp_sub['group'] == 0]
                    df_train_sub = df_temp_sub[df_temp_sub['group'] == 1]
                    
                    # Resample from fitted distributions
                    val_resampled = resample_from_skewt_fit(df_val_sub[col_name].values, n_samples=10000, use_skewt=True, random_state=i)
                    train_resampled = resample_from_skewt_fit(df_train_sub[col_name].values, n_samples=10000, use_skewt=True, random_state=i)
                    
                    # Create resampled dataframe
                    group_resampled = np.concatenate([np.zeros(len(val_resampled)), np.ones(len(train_resampled))])
                    values_resampled = np.concatenate([val_resampled, train_resampled])
                    names_resampled = np.concatenate([
                        np.tile(df_val_sub['value'].values, (len(val_resampled) // len(df_val_sub) + 1))[:len(val_resampled)],
                        np.tile(df_train_sub['value'].values, (len(train_resampled) // len(df_train_sub) + 1))[:len(train_resampled)]
                    ])
                    
                    _, I, _ = mi_per_name_knn_ross(group_resampled, values_resampled, names=names_resampled, seed=i)
                else:
                    # Use original data for MI
                    _, I, _ = mi_per_name_knn_ross(df_temp_sub['group'].values, df_temp_sub[col_name].values, names=df_temp_sub['value'].values, seed=i)
                
                i_values.append(I)
            
            I = np.mean(i_values)
            I_std = np.std(i_values)*factor
            auc = np.mean(auc_values)
            auc_std = np.std(auc_values)*factor
            print(model, prompt, f"{I:.6f} bits ± {I_std:.6f} bits", f"{auc:.6f} ± {auc_std:.6f}")

            df_results.loc[len(df_results)] = [model, prompt, I, I_std, auc, auc_std]

    for prompt in prompts:
        df_temp = df[['value', 'group', 'p_pre_' + prompt, 'p_ft_' + prompt]].copy()
        pre_col = 'p_pre_' + prompt
        ft_col = 'p_ft_' + prompt
        
        i_values = []
        auc_values = []
        np.random.seed(42)
        random.seed(42)
        for i in range(100): # can boostrap at subsampling too
            df_temp_sub = prepare_data(df_temp)
            
            # For AUC: use original data
            auc = attack(df_temp_sub['group'].values, df_temp_sub[[pre_col, ft_col]].values)
            auc_values.append(auc)
            
            # For MI: resample if requested
            if use_skewt:
                # Fit skew t for each group and each column, then resample
                df_val_sub = df_temp_sub[df_temp_sub['group'] == 0]
                df_train_sub = df_temp_sub[df_temp_sub['group'] == 1]
                
                # Resample each column separately
                val_pre_resampled = resample_from_skewt_fit(df_val_sub[pre_col].values, n_samples=10000, use_skewt=True, random_state=i)
                val_ft_resampled = resample_from_skewt_fit(df_val_sub[ft_col].values, n_samples=10000, use_skewt=True, random_state=i)
                train_pre_resampled = resample_from_skewt_fit(df_train_sub[pre_col].values, n_samples=10000, use_skewt=True, random_state=i)
                train_ft_resampled = resample_from_skewt_fit(df_train_sub[ft_col].values, n_samples=10000, use_skewt=True, random_state=i)
                
                # Create resampled dataframe with both columns
                group_resampled = np.concatenate([np.zeros(len(val_pre_resampled)), np.ones(len(train_pre_resampled))])
                values_resampled = np.column_stack([
                    np.concatenate([val_pre_resampled, train_pre_resampled]),
                    np.concatenate([val_ft_resampled, train_ft_resampled])
                ])
                names_resampled = np.concatenate([
                    np.tile(df_val_sub['value'].values, (len(val_pre_resampled) // len(df_val_sub) + 1))[:len(val_pre_resampled)],
                    np.tile(df_train_sub['value'].values, (len(train_pre_resampled) // len(df_train_sub) + 1))[:len(train_pre_resampled)]
                ])
                
                _, I, _ = mi_per_name_knn_ross(group_resampled, values_resampled, names=names_resampled, seed=i)
            else:
                # Use original data for MI
                _, I, _ = mi_per_name_knn_ross(df_temp_sub['group'].values, df_temp_sub[[pre_col, ft_col]].values, names=df_temp_sub['value'].values, seed=i)
            
            i_values.append(I)
        
        I = np.mean(i_values)
        I_std = np.std(i_values)*factor
        auc = np.mean(auc_values)
        auc_std = np.std(auc_values)*factor
        print(prompt, f"{I:.6f} bits ± {I_std:.6f} bits", f"{auc:.6f} ± {auc_std:.6f}")
        df_results_combined.loc[len(df_results_combined)] = [prompt, I, I_std, auc, auc_std]

    i_values = []
    auc_values = []
    np.random.seed(42)
    random.seed(42)
    all_columns = [prefix + '_' + prompt for prefix in models for prompt in prompts]
    for i in range(100):
        df_temp_sub = prepare_data(df)
        
        # For AUC: use original data
        auc, X_test, y_test, y_pred_proba, idx_test, clf = attack(df_temp_sub['group'].values, df_temp_sub[all_columns].values, return_proba=True)
        auc_values.append(auc)
        auc_last = auc
        
        # For MI: resample if requested
        if use_skewt:
            # Fit skew t for each group and each column, then resample
            df_val_sub = df_temp_sub[df_temp_sub['group'] == 0]
            df_train_sub = df_temp_sub[df_temp_sub['group'] == 1]
            
            # Resample each column separately
            val_resampled_cols = []
            train_resampled_cols = []
            for col in all_columns:
                val_resampled = resample_from_skewt_fit(df_val_sub[col].values, n_samples=10000, use_skewt=True, random_state=i)
                train_resampled = resample_from_skewt_fit(df_train_sub[col].values, n_samples=10000, use_skewt=True, random_state=i)
                val_resampled_cols.append(val_resampled)
                train_resampled_cols.append(train_resampled)
            
            # Stack all columns
            val_all_resampled = np.column_stack(val_resampled_cols)
            train_all_resampled = np.column_stack(train_resampled_cols)
            
            # Create resampled dataframe
            group_resampled = np.concatenate([np.zeros(len(val_all_resampled)), np.ones(len(train_all_resampled))])
            values_resampled = np.vstack([val_all_resampled, train_all_resampled])
            names_resampled = np.concatenate([
                np.tile(df_val_sub['value'].values, (len(val_all_resampled) // len(df_val_sub) + 1))[:len(val_all_resampled)],
                np.tile(df_train_sub['value'].values, (len(train_all_resampled) // len(df_train_sub) + 1))[:len(train_all_resampled)]
            ])
            
            _, I, _ = mi_per_name_knn_ross(group_resampled, values_resampled, names=names_resampled, seed=i)
        else:
            # Use original data for MI
            _, I, _ = mi_per_name_knn_ross(df_temp_sub['group'].values, df_temp_sub[all_columns].values, names=df_temp_sub['value'].values, seed=i)
        
        i_values.append(I)
    I = np.mean(i_values)
    I_std = np.std(i_values)*factor
    auc = np.mean(auc_values)
    auc_std = np.std(auc_values)*factor
    print("All", f"{I:.6f} bits ± {I_std:.6f} bits", f"{auc:.6f} ± {auc_std:.6f}")

    # print(auc_last, X_test, y_test, y_pred_proba, idx_test, df_temp_sub)
    df_temp_sub["y_pred_proba"] = np.nan
    df_temp_sub.iloc[idx_test, df_temp_sub.columns.get_loc("y_pred_proba")] = y_pred_proba
    # print(df_temp_sub[['value', 'group', 'y_pred_proba']])
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # Example: threshold that maximizes F1
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_i = np.argmax(f1)
    best_thr = thresholds[best_i-1] if best_i > 0 else 0.0  # curve has one extra point

    print(f"best_thr≈{best_thr:.4f}  precision={precisions[best_i]:.4f}  recall={recalls[best_i]:.4f}  f1={f1[best_i]:.4f}")
    

    # fct to compute mi 

    # fct to compute AUC attack

    return df_results, df_results_combined, I, I_std, auc, auc_std, df_temp_sub, clf

def fit_skewnorm_distribution(data, use_skewt=False):
    """
    Fit a skewed normal or skewed t distribution to the data.
    
    Parameters
    ----------
    data : array-like
        Data to fit the distribution to
    use_skewt : bool, default False
        If True, fit a skewed t distribution instead of skewed normal.
        Requires the 'skewt' package to be installed.
        
    Returns
    -------
    params : tuple
        For skewnorm: (a, loc, scale) parameters
        For skewt: (df, loc, scale, skew) parameters (if skewt package available)
    ks_stat : float
        Kolmogorov-Smirnov test statistic
    ks_pvalue : float
        Kolmogorov-Smirnov test p-value
    dist_type : str
        'skewnorm' or 'skewt' indicating which distribution was fitted
    skewt_impl : str or None
        If dist_type is 'skewt', indicates implementation: 'jf_skew_t' or 'external_skewt'
        None for skewnorm
    """
    if len(data) < 3:
        return None, None, None, None, None
    
    data = np.asarray(data)
    log_data = np.log(data)
    
    if use_skewt:
        # Try scipy's jf_skew_t first (available in scipy >= 1.13.0)
        if HAS_JF_SKEWT:
            try:
                mean_init = np.mean(log_data)
                std_init = np.std(log_data)
                # jf_skew_t has parameters: a, b, loc, scale
                # a and b control skewness and degrees of freedom
                # Initial: a=b gives symmetric t-distribution
                a_init = b_init = 5.0  # Start with symmetric t-distribution
                
                # Check for numerical issues
                if not np.isfinite(mean_init) or not np.isfinite(std_init) or std_init <= 0:
                    raise ValueError(f"Invalid initial parameters: mean={mean_init}, std={std_init}")
                
                def neg_log_likelihood_jf_skewt(params):
                    a, b, loc, scale = params
                    if a <= 0 or b <= 0 or scale <= 0:
                        return np.inf
                    try:
                        log_likelihood = np.sum(jf_skew_t.logpdf(log_data, a=a, b=b, loc=loc, scale=scale))
                        if not np.isfinite(log_likelihood):
                            return np.inf
                        return -log_likelihood
                    except:
                        return np.inf
                
                # Try multiple optimization methods for better convergence
                methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']
                result = None
                for method in methods:
                    try:
                        if method == 'L-BFGS-B':
                            # Use bounds for L-BFGS-B
                            bounds = [(0.1, 50.0), (0.1, 50.0), (None, None), (0.01, None)]
                            result = minimize(neg_log_likelihood_jf_skewt, [a_init, b_init, mean_init, std_init], 
                                           method=method, bounds=bounds, options={'maxiter': 1000})
                        else:
                            result = minimize(neg_log_likelihood_jf_skewt, [a_init, b_init, mean_init, std_init], 
                                           method=method, options={'maxiter': 1000})
                        if result.success:
                            break
                    except:
                        continue
                
                if result is not None and result.success:
                    a, b, loc, scale = result.x
                    # Ensure valid parameters
                    a = max(a, 0.1)
                    b = max(b, 0.1)
                    scale = max(scale, 0.01)
                    # Perform KS test
                    ks_stat, ks_pvalue = kstest(log_data, 
                                               lambda x: jf_skew_t.cdf(x, a=a, b=b, loc=loc, scale=scale))
                    return (a, b, loc, scale), ks_stat, ks_pvalue, 'skewt', 'jf_skew_t'
                else:
                    reason = result.message if result is not None else "No result returned"
                    raise ValueError(f"Optimization failed: {reason}")
            except Exception as e:
                print(f"Warning: jf_skew_t fitting failed: {e}. Trying external skewt package...")
        
        # Fallback to external skewt package if available
        if HAS_SKEWT:
            try:
                skewt_dist = SkewT()
                mean_init = np.mean(log_data)
                std_init = np.std(log_data)
                
                # Check for numerical issues
                if not np.isfinite(mean_init) or not np.isfinite(std_init) or std_init <= 0:
                    raise ValueError(f"Invalid initial parameters: mean={mean_init}, std={std_init}")
                
                def neg_log_likelihood_skewt(params):
                    df, loc, scale, skew = params
                    if df <= 1 or scale <= 0:
                        return np.inf
                    try:
                        log_likelihood = np.sum(skewt_dist.logpdf(log_data, df=df, loc=loc, scale=scale, skew=skew))
                        if not np.isfinite(log_likelihood):
                            return np.inf
                        return -log_likelihood
                    except:
                        return np.inf
                
                # Try multiple optimization methods for better convergence
                methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']
                result = None
                for method in methods:
                    try:
                        if method == 'L-BFGS-B':
                            # Use bounds for L-BFGS-B
                            bounds = [(1.1, 50.0), (None, None), (0.01, None), (-10.0, 10.0)]
                            result = minimize(neg_log_likelihood_skewt, [5.0, mean_init, std_init, 0.0], 
                                           method=method, bounds=bounds, options={'maxiter': 1000})
                        else:
                            result = minimize(neg_log_likelihood_skewt, [5.0, mean_init, std_init, 0.0], 
                                           method=method, options={'maxiter': 1000})
                        if result.success:
                            break
                    except:
                        continue
                
                if result is not None and result.success:
                    df, loc, scale, skew = result.x
                    df = max(df, 1.1)
                    ks_stat, ks_pvalue = kstest(log_data, 
                                               lambda x: skewt_dist.cdf(x, df=df, loc=loc, scale=scale, skew=skew))
                    return (df, loc, scale, skew), ks_stat, ks_pvalue, 'skewt', 'external_skewt'
                else:
                    reason = result.message if result is not None else "No result returned"
                    raise ValueError(f"Optimization failed: {reason}")
            except Exception as e:
                print(f"Warning: External skewt fitting failed: {e}. Falling back to skew normal.")
        
        if not HAS_JF_SKEWT and not HAS_SKEWT:
            raise ImportError("Skew t distribution requires either scipy >= 1.13.0 (for jf_skew_t) "
                            "or the 'skewt-scipy' package. Install with: pip install skewt-scipy")
        
        # If both methods failed, fall back to skewnorm
        # Only print warning if we actually tried both methods
        if HAS_JF_SKEWT or HAS_SKEWT:
            print(f"Warning: Skew t fitting failed for dataset (n={len(data)}). Falling back to skew normal.")
        use_skewt = False
    
    # Fit skewed normal distribution
    mean_init = np.mean(log_data)
    std_init = np.std(log_data)
    skew_init = 0.0  # Start with no skew
    
    def neg_log_likelihood(params):
        a, loc, scale = params
        if scale <= 0:
            return np.inf
        try:
            log_likelihood = np.sum(skewnorm.logpdf(log_data, a=a, loc=loc, scale=scale))
            return -log_likelihood
        except:
            return np.inf
    
    # Fit using MLE
    try:
        result = minimize(neg_log_likelihood, [skew_init, mean_init, std_init], 
                         method='Nelder-Mead', options={'maxiter': 1000})
        if result.success:
            a, loc, scale = result.x
            # Perform KS test
            ks_stat, ks_pvalue = kstest(log_data, 
                                       lambda x: skewnorm.cdf(x, a=a, loc=loc, scale=scale))
            return (a, loc, scale), ks_stat, ks_pvalue, 'skewnorm', None
        else:
            return None, None, None, None, None
    except:
        return None, None, None, None, None

def plot_distribution_histograms(df_wide, pii_type, fit_skewnorm=False, use_skewt=False):
    """
    Plot histograms of the probability distributions, creating one plot per prompt.
    Each plot shows 4 subplots:
    1. Validation data only
    2. Training data only  
    3. Pre-trained model (val + train combined)
    4. Fine-tuned model (val + train combined)
    
    Parameters
    ----------
    df_wide : pandas.DataFrame
        DataFrame with probability distributions
    pii_type : str
        Type of PII being analyzed
    fit_skewnorm : bool, default False
        If True, fit distributions to training data for each prompt and model,
        plot the density curves above histograms, and include KS test results in the legend.
    use_skewt : bool, default False
        If True and fit_skewnorm is True, fit skewed t distribution instead of skewed normal.
        Requires the 'skewt' package to be installed.
    """
    # Get unique prompts from the data by extracting from column names
    prompt_columns = [col for col in df_wide.columns if col.startswith('p_pre_') or col.startswith('p_ft_')]
    prompts = []
    for col in prompt_columns:
        if col.startswith('p_pre_'):
            prompt = col.replace('p_pre_', '')
        elif col.startswith('p_ft_'):
            prompt = col.replace('p_ft_', '')
        if prompt not in prompts:
            prompts.append(prompt)
    
    # Split data by split (val vs train)
    df_val = df_wide[df_wide['split'] == 'val']
    df_train = df_wide[df_wide['split'] == 'train']
    
    figures = []
    
    for prompt in prompts:
        # print(f"Prompt: {prompt}")
        # Create 2x2 subplots for this prompt
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get columns for this prompt
        pre_col = f'p_pre_{prompt}'
        ft_col = f'p_ft_{prompt}'
        prob_columns = [pre_col, ft_col]
        
        # Clean up prompt name for display
        clean_prompt = prompt.strip().rstrip(':').strip()
        
        # Colors for each distribution
        colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
        labels = [f'Pre-trained ({clean_prompt})', f'Fine-tuned ({clean_prompt})']
        
        # Subplot 1: Validation data only
        ax = axes[0]
        for i, (col, color, label) in enumerate(zip(prob_columns, colors, labels)):
            data = df_val[col].dropna()
            if len(data) > 0:
                label_with_count = f'{label} (n={len(data)})'
                ax.hist(np.log(data), bins=50, alpha=0.6, color=color, edgecolor='black', linewidth=0.5, 
                       density=True, label=label_with_count)
        ax.set_xlabel('Log probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Validation Data Only - {clean_prompt} Prompt')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Subplot 2: Training data only
        ax = axes[1]
        for i, (col, color, label) in enumerate(zip(prob_columns, colors, labels)):
            data = df_train[col].dropna()
            if len(data) > 0:
                log_data = np.log(data)
                label_with_count = f'{label} (n={len(data)})'
                ax.hist(log_data, bins=50, alpha=0.6, color=color, edgecolor='black', linewidth=0.5, 
                       density=True, label=label_with_count)
                
                # Fit distribution if requested
                if fit_skewnorm:
                    params, ks_stat, ks_pvalue, dist_type, skewt_impl = fit_skewnorm_distribution(data, use_skewt=use_skewt)
                    if params is not None:
                        # Plot density curve
                        x_range = np.linspace(log_data.min(), log_data.max(), 200)
                        if dist_type == 'skewt':
                            if skewt_impl == 'jf_skew_t' and HAS_JF_SKEWT:
                                a, b, loc, scale = params
                                density = jf_skew_t.pdf(x_range, a=a, b=b, loc=loc, scale=scale)
                                dist_label = 'SkewT (JF)'
                            elif skewt_impl == 'external_skewt' and HAS_SKEWT:
                                df, loc, scale, skew = params
                                skewt_dist = SkewT()
                                density = skewt_dist.pdf(x_range, df=df, loc=loc, scale=scale, skew=skew)
                                dist_label = 'SkewT'
                            else:
                                # Fallback to skewnorm if skewt implementation not available
                                a, loc, scale = params if len(params) == 3 else (params[0], params[2], params[3])
                                density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                                dist_label = 'SkewNorm'
                        else:
                            # skewnorm
                            a, loc, scale = params
                            density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                            dist_label = 'SkewNorm'
                        ax.plot(x_range, density, color=color, linewidth=2, linestyle='--',
                               label=f'{label} {dist_label} fit (KS: D={ks_stat:.3f}, p={ks_pvalue:.3f})')
        ax.set_xlabel('Log probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Training Data Only - {clean_prompt} Prompt')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Subplot 3: Pre-trained model (val + train combined)
        ax = axes[2]
        col = pre_col
        color = '#1f77b4'  # Blue
        label = f'Pre-trained ({clean_prompt})'
        
        # Get validation and training data separately for this column
        data_val = df_val[col].dropna()
        data_train = df_train[col].dropna()
        
        if len(data_val) > 0:
            label_with_count = f'{label} - Val (n={len(data_val)})'
            ax.hist(np.log(data_val), bins=50, alpha=0.6, color=color, edgecolor='black', linewidth=0.5, 
                   density=True, label=label_with_count, linestyle='--', hatch='///')
        
        if len(data_train) > 0:
            train_color = '#00CED1'  # Dark Turquoise
            log_data_train = np.log(data_train)
            label_with_count = f'{label} - Train (n={len(data_train)})'
            ax.hist(log_data_train, bins=50, alpha=0.6, color=train_color, edgecolor='black', linewidth=0.5, 
                   density=True, label=label_with_count, hatch='')
            
            # Fit distribution if requested
            if fit_skewnorm:
                params, ks_stat, ks_pvalue, dist_type, skewt_impl = fit_skewnorm_distribution(data_train, use_skewt=use_skewt)
                if params is not None:
                    # Plot density curve
                    x_range = np.linspace(log_data_train.min(), log_data_train.max(), 200)
                    if dist_type == 'skewt':
                        if skewt_impl == 'jf_skew_t' and HAS_JF_SKEWT:
                            a, b, loc, scale = params
                            density = jf_skew_t.pdf(x_range, a=a, b=b, loc=loc, scale=scale)
                            dist_label = 'SkewT (JF)'
                        elif skewt_impl == 'external_skewt' and HAS_SKEWT:
                            df, loc, scale, skew = params
                            skewt_dist = SkewT()
                            density = skewt_dist.pdf(x_range, df=df, loc=loc, scale=scale, skew=skew)
                            dist_label = 'SkewT'
                        else:
                            # Fallback to skewnorm if skewt implementation not available
                            a, loc, scale = params if len(params) == 3 else (params[0], params[2], params[3])
                            density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                            dist_label = 'SkewNorm'
                    else:
                        # skewnorm
                        a, loc, scale = params
                        density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                        dist_label = 'SkewNorm'
                    ax.plot(x_range, density, color=train_color, linewidth=2, linestyle='--',
                           label=f'{label} - Train {dist_label} (KS: D={ks_stat:.3f}, p={ks_pvalue:.3f})')
        ax.set_xlabel('Log probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Pre-trained Model (Val + Train) - {clean_prompt} Prompt')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Subplot 4: Fine-tuned model (val + train combined)
        ax = axes[3]
        col = ft_col
        color = '#ff7f0e'  # Orange
        label = f'Fine-tuned ({clean_prompt})'
        
        # Get validation and training data separately for this column
        data_val = df_val[col].dropna()
        data_train = df_train[col].dropna()
        
        if len(data_val) > 0:
            label_with_count = f'{label} - Val (n={len(data_val)})'
            ax.hist(np.log(data_val), bins=50, alpha=0.6, color=color, edgecolor='black', linewidth=0.5, 
                   density=True, label=label_with_count, hatch='///')
        
        if len(data_train) > 0:
            train_color = '#32CD32'  # Lime Green
            log_data_train = np.log(data_train)
            label_with_count = f'{label} - Train (n={len(data_train)})'
            ax.hist(log_data_train, bins=50, alpha=0.6, color=train_color, edgecolor='black', linewidth=0.5, 
                   density=True, label=label_with_count, hatch='')
            
            # Fit distribution if requested
            if fit_skewnorm:
                params, ks_stat, ks_pvalue, dist_type, skewt_impl = fit_skewnorm_distribution(data_train, use_skewt=use_skewt)
                if params is not None:
                    # Plot density curve
                    x_range = np.linspace(log_data_train.min(), log_data_train.max(), 200)
                    if dist_type == 'skewt':
                        if skewt_impl == 'jf_skew_t' and HAS_JF_SKEWT:
                            a, b, loc, scale = params
                            density = jf_skew_t.pdf(x_range, a=a, b=b, loc=loc, scale=scale)
                            dist_label = 'SkewT (JF)'
                        elif skewt_impl == 'external_skewt' and HAS_SKEWT:
                            df, loc, scale, skew = params
                            skewt_dist = SkewT()
                            density = skewt_dist.pdf(x_range, df=df, loc=loc, scale=scale, skew=skew)
                            dist_label = 'SkewT'
                        else:
                            # Fallback to skewnorm if skewt implementation not available
                            a, loc, scale = params if len(params) == 3 else (params[0], params[2], params[3])
                            density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                            dist_label = 'SkewNorm'
                    else:
                        # skewnorm
                        a, loc, scale = params
                        density = skewnorm.pdf(x_range, a=a, loc=loc, scale=scale)
                        dist_label = 'SkewNorm'
                    ax.plot(x_range, density, color=train_color, linewidth=2, linestyle='--',
                           label=f'{label} - Train {dist_label} (KS: D={ks_stat:.3f}, p={ks_pvalue:.3f})')
        ax.set_xlabel('Log probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Fine-tuned Model (Val + Train) - {clean_prompt} Prompt')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        # print(f"Saving figure for prompt: {clean_prompt}, {prompt}")
        figures.append((fig, clean_prompt))
    
    return figures

if __name__ == "__main__":
    if True:
        # path_pretrained = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/ll_all_output_True.csv'
        # path_pretrained_mrn = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/ll_all_output_True-with_mrn.csv'
        # path_finetuned = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/ll_all_output_False.csv'
        # path_finetuned_mrn = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/ll_all_output_False-with_mrn.csv'
        # df_pretrained = pd.read_csv(path_pretrained)
        # df_pretrained_mrn = pd.read_csv(path_pretrained_mrn)
        # df_pretrained = pd.concat([df_pretrained, df_pretrained_mrn])

        # df_finetuned = pd.read_csv(path_finetuned)
        # df_finetuned_mrn = pd.read_csv(path_finetuned_mrn)
        # df_finetuned = pd.concat([df_finetuned, df_finetuned_mrn])

        # replace with new header
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate overall probability plots')
        # parser.add_argument('--pii_type', type=str, default='name', choices=['name', 'mrn'],
        #                     help='PII type to analyze (default: name)')
        parser.add_argument('--model', type=str, default='1B', choices=['1B', '8B'],
                            help='Model size to use (default: 1B)')
        parser.add_argument('--dataset_size', type=int, default=1, choices=[1,10,100],
                            help='Dataset size to use (default: 1)')
        parser.add_argument('--plots_dir', type=str, 
                            default='/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration',
                            help='Directory to save plots (default: pipeline/plots)')
        args = parser.parse_args()

        # pii_type = args.pii_type
        model = args.model
        dataset_size = args.dataset_size
        plots_dir = args.plots_dir

        # pii_type_suffix = pii_type
        model_suffix = model

        if dataset_size == 1:
            if model == '8B':
                qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_8B.csv'
                pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_8B.csv'
            elif model == '1B':
                qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_batch.csv' # 1B_batch soon
                pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_batch.csv' # 1B_batch soon
                # qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True.csv' # 1B_batch soon
                # pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False.csv' # 1B_batch soon

                # qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_batch.csv' # 1B_batch soon
                # pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_batch.csv' # 1B_batch soon
            else:
                raise ValueError(f"Model {model} not supported")
        elif dataset_size == 10:
            if model == '1B':
                qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_10_batch.csv'
                pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_10_batch.csv'
            elif model == '8B':
                qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_8B_10_batch.csv'
                pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_8B_10_batch.csv'
            else:
                raise ValueError(f"Model {model} not supported")
        elif dataset_size == 100:
            if model == '1B':
                qi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_100_batch.csv'
                pi_paths = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_100_batch.csv'
            else:
                raise ValueError(f"Model {model} not supported")
        else:
            raise ValueError(f"Dataset size {dataset_size} not supported")

        os.makedirs(plots_dir, exist_ok=True)

        df_pretrained = pd.read_csv(qi_paths)
        df_finetuned = pd.read_csv(pi_paths)

        df_pretrained_short = df_pretrained[['pii_type','split', 'prompt', 'value', 'll', 'list_tokens']]

        # remove prefixes for probability? when renormalizing

        # bimodal distribution before: attending names and patient names
        # ll

        # print(df_pretrained)
        # print(df_finetuned)
        # exit()

        # print(df_finetuned['pii_type'].unique())
        # exit()

        # avoid MRN issues?
        df_pretrained['value'] = df_pretrained['value'].astype(str)
        df_finetuned['value'] = df_finetuned['value'].astype(str)

        all_results = pd.DataFrame(columns=['pii_type', 'dataset_size', 'pii_rate', 'n_epochs', 'model', 'prompt', 'I', 'I_std', 'auc', 'auc_std'])
        all_results_combined = pd.DataFrame(columns=['pii_type', 'dataset_size', 'pii_rate', 'n_epochs', 'prompt', 'I', 'I_std', 'auc', 'auc_std'])
        all_results_all = pd.DataFrame(columns=['pii_type', 'dataset_size', 'pii_rate', 'n_epochs', 'I', 'I_std', 'auc', 'auc_std'])
        for pii_type in df_finetuned['pii_type'].unique():
            # if pii_type == "unit_no":
            #     continue
            # if 'name' in pii_type:
                # continue
            pii_type_suffix = pii_type
            for index, df_g in df_finetuned[df_finetuned['pii_type'] == pii_type].groupby(['dataset_size', 'pii_rate', 'n_epochs']):
                if index[0] <= 0:
                    continue

                # if index[1] != 0.1 or index[2] != 10:
                #     continue

                df_finetuned_filtered = df_finetuned[df_finetuned['pii_type'] == pii_type]
                df_finetuned_filtered_val = df_finetuned_filtered[df_finetuned_filtered['split'] == 'val']
                df_finetuned_filtered_val = df_finetuned_filtered_val.drop_duplicates(subset=['prompt','value'])

                df_pretrained_short_filtered = df_pretrained_short[df_pretrained_short['pii_type'] == pii_type]
                df_g = df_g[df_g['split'] == 'train']

                df_g.to_csv("tmp_train_finetuned.csv", index=False)
                df_finetuned_filtered_val.to_csv("tmp_val_finetuned.csv", index=False)
                df_pretrained_short_filtered.to_csv("tmp_pretrained_short.csv", index=False)

                print(pii_type, index, len(df_g), len(df_finetuned_filtered_val), len(df_pretrained_short_filtered))
                print(df_g['ll'].isna().sum())
                print(df_finetuned_filtered_val['ll'].isna().sum())
                print(df_pretrained_short_filtered['ll'].isna().sum())

                # why converted to int automatically?
                df_g['value'] = df_g['value'].astype(str)
                df_finetuned_filtered_val['value'] = df_finetuned_filtered_val['value'].astype(str)
                df_pretrained_short_filtered['value'] = df_pretrained_short_filtered['value'].astype(str)


                # Collect both datasets efficiently without iterating row-by-row
                df_analysis = pd.concat([
                    df_g[['split', 'prompt', 'value', 'll']],
                    df_finetuned_filtered_val[['split', 'prompt', 'value', 'll']]
                ], ignore_index=True)


                # print(df_analysis['ll_x'].isna().sum())

                # print(df_analysis[df_analysis['split'] == 'val'].shape)
                # print(df_analysis[df_analysis['split'] == 'train'].shape)
                # print(df_pretrained_short_filtered[df_pretrained_short_filtered['split'] == 'val'][['split','prompt', 'value']].iloc[0]['value'].replace(' ', 'X'))
                # print(df_analysis[df_analysis['split'] == 'val'][['split','prompt', 'value']].iloc[0]['value'].replace(' ', 'X'))

                df_analysis = df_analysis.merge(df_pretrained_short_filtered, on=['split','prompt','value'], how='inner')

                # print(df_analysis[df_analysis['split'] == 'val'].shape)
                # print(df_analysis[df_analysis['split'] == 'train'].shape)
                # exit()
                df_analysis['p_ft'] = np.exp(df_analysis['ll_x'])
                df_analysis['p_pre'] = np.exp(df_analysis['ll_y'])

                df_analysis = df_analysis[['split', 'prompt', 'value', 'p_ft', 'p_pre', 'list_tokens']]
                
                # print(df_analysis['p_pre'].isna().sum())
                # print(df_analysis['p_ft'].isna().sum())

                # exit()

                # print(len(df_analysis))

                # print(df_analysis)
                # input()
                # continue

                df_wide = df_analysis.pivot(
                    index=['value', 'split', 'list_tokens'],  # what stays fixed
                    columns='prompt',                         # what becomes columns
                    values=['p_pre', 'p_ft']                  # what gets spread across
                )

                # Flatten the multi-index columns
                df_wide.columns = [f"{col1}_{col2}" for col1, col2 in df_wide.columns]
                df_wide = df_wide.reset_index()

                # df_wide = purge_prefixes(df_wide)
                # print(df_wide)

                # input()

                # compute_mi(df_wide)
                df_results, df_results_combined, I, I_std, auc, auc_std, df_temp_sub, clf = compute_mi(df_wide, pii_type, use_skewt=False)#, use_skewt=True)
                all_results_all.loc[len(all_results_all)] = [pii_type, index[0], index[1], index[2], I, I_std, auc, auc_std]

                df_temp_sub.to_csv(os.path.join(plots_dir, f'df_temp_sub_{pii_type_suffix}_{model_suffix}_{dataset_size}_{index[1]}_{index[2]}.csv'), index=False)

                joblib.dump(clf, os.path.join(plots_dir, f'clf_{pii_type_suffix}_{model_suffix}_{dataset_size}_{index[1]}_{index[2]}.pkl'))

                # Create histogram plots for this configuration (one per prompt)
                figures = plot_distribution_histograms(df_wide, pii_type, fit_skewnorm=False, use_skewt=False)#, fit_skewnorm=True, use_skewt=True)
                for fig, prompt_name in figures:
                    hist_filename = os.path.join(plots_dir, f'histograms_{pii_type_suffix}_{model_suffix}_{dataset_size}_{index[1]}_{index[2]}_{prompt_name}.png')
                    # print(f"Histogram saved to: {hist_filename}")
                    fig.savefig(hist_filename, dpi=300, bbox_inches='tight')
                    # fig.close()
                    print(f"Histogram saved to: {hist_filename}")
                
                df_results['pii_type'] = pii_type
                df_results['dataset_size'] = index[0]
                df_results['pii_rate'] = index[1]
                df_results['n_epochs'] = index[2]
                # reorder columns
                df_results = df_results[['pii_type', 'dataset_size', 'pii_rate', 'n_epochs', 'model', 'prompt', 'I', 'I_std', 'auc', 'auc_std']]
                if len(all_results) == 0:
                    all_results = df_results
                else:   
                    all_results = pd.concat([all_results, df_results], ignore_index=True)

                df_results_combined['pii_type'] = pii_type
                df_results_combined['dataset_size'] = index[0]
                df_results_combined['pii_rate'] = index[1]
                df_results_combined['n_epochs'] = index[2]
                df_results_combined = df_results_combined[['pii_type', 'dataset_size', 'pii_rate', 'n_epochs', 'prompt', 'I', 'I_std', 'auc', 'auc_std']]
                if len(all_results_combined) == 0:
                    all_results_combined = df_results_combined
                else:
                    all_results_combined = pd.concat([all_results_combined, df_results_combined], ignore_index=True)
            # break

                # for idx, row in df_g.iterrows():
                    # df_analysis.loc[len(df_analysis)] = ['ft', row['split'], row['prompt'], row['value'], row['ll']]


            # train_pretrained
            # val_pretrainedj
            # train_finetuned
            # val_finetuned


        all_results.to_csv(os.path.join(plots_dir, f'mia_results_all_{model}_{dataset_size}.csv'), index=False)
        all_results_combined.to_csv(os.path.join(plots_dir, f'mia_results_combined_all_{model}_{dataset_size}.csv'), index=False)
        all_results_all.to_csv(os.path.join(plots_dir, f'mia_results_all_all_{model}_{dataset_size}.csv'), index=False)
        
    all_results = pd.read_csv(os.path.join(plots_dir, f'mia_results_all_{model}_{dataset_size}.csv'))
    all_results_combined = pd.read_csv(os.path.join(plots_dir, f'mia_results_combined_all_{model}_{dataset_size}.csv'))
    all_results_all = pd.read_csv(os.path.join(plots_dir, f'mia_results_all_all_{model}_{dataset_size}.csv'))
    # visualization: for each configuration
    # ignore dataset size for now
    # plot as line with error bars, the MI as a function of pii rate. one curve for each model and each number of epochs, one subplot for each prompt and pii_type
    
    # Create visualization
    def create_mi_visualization(df_results):
        """
        Create visualization plots for MI analysis.
        Plots MI as a function of pii_rate with error bars.
        One curve for each model and number of epochs.
        One subplot for each prompt and pii_type.
        """
        # Get unique values for subplot organization
        prompts = df_results['prompt'].unique()
        pii_types = df_results['pii_type'].unique()
        models = df_results['model'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(len(prompts), len(pii_types), figsize=(10*len(pii_types), 4*len(prompts)))
        if len(prompts) == 1 and len(pii_types) == 1:
            axes = np.array([[axes]])
        elif len(prompts) == 1:
            axes = axes.reshape(1, -1)
        elif len(pii_types) == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for different models
        model_colors = {'p_ft': '#1f77b4', 'p_pre': '#ff7f0e'}  # Blue for fine-tuned, Orange for pre-trained
        # Line styles for different epochs
        epoch_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Cycle through styles
        
        for i, prompt in enumerate(prompts):
            for j, pii_type in enumerate(pii_types):
                ax = axes[i, j]
                
                # Filter data for this prompt and pii_type
                data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot curves for each model and epoch combination
                for model in models:
                    for k, epoch in enumerate(epochs):
                        model_epoch_data = data[(data['model'] == model) & (data['n_epochs'] == epoch)]
                        
                        if len(model_epoch_data) == 0:
                            continue
                        
                        # Sort by pii_rate for proper line plotting
                        model_epoch_data = model_epoch_data.sort_values('pii_rate')
                        
                        # Create label for legend - show epoch and model separately
                        model_display = 'Fine-tuned' if model == 'p_ft' else 'Pre-trained'
                        label = f"{model_display} ({format_epoch_label(epoch)})"
                        
                        # Plot line with error bars
                        ax.errorbar(
                            model_epoch_data['pii_rate'],
                            model_epoch_data['I'],
                            yerr=model_epoch_data['I_std'],
                            label=label,
                            color=model_colors[model],
                            linestyle=epoch_linestyles[k % len(epoch_linestyles)],
                            marker='o',
                            capsize=3,
                            capthick=1,
                            markersize=4
                        )
                
                # Customize subplot
                ax.set_xscale('log')
                # ax.set_yscale('log')
                ax.set_xlabel('PII Rate (%)')
                ax.set_ylabel('Mutual Information (bits)')
                pii_type_display = format_pii_type_for_title(pii_type)
                ax.set_title(f'Prompt: {prompt}\nPII Type: {pii_type_display}')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_auc_visualization(df_results):
        """
        Create visualization plots for AUC analysis.
        Plots AUC as a function of pii_rate with error bars.
        One curve for each model and number of epochs.
        One subplot for each prompt and pii_type.
        """
        # Get unique values for subplot organization
        prompts = df_results['prompt'].unique()
        pii_types = df_results['pii_type'].unique()
        models = df_results['model'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(len(prompts), len(pii_types), figsize=(10*len(pii_types), 4*len(prompts)))
        if len(prompts) == 1 and len(pii_types) == 1:
            axes = np.array([[axes]])
        elif len(prompts) == 1:
            axes = axes.reshape(1, -1)
        elif len(pii_types) == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for different models
        model_colors = {'p_ft': '#1f77b4', 'p_pre': '#ff7f0e'}  # Blue for fine-tuned, Orange for pre-trained
        # Line styles for different epochs
        epoch_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Cycle through styles
        
        for i, prompt in enumerate(prompts):
            for j, pii_type in enumerate(pii_types):
                ax = axes[i, j]
                
                # Filter data for this prompt and pii_type
                data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot curves for each model and epoch combination
                for model in models:
                    for k, epoch in enumerate(epochs):
                        model_epoch_data = data[(data['model'] == model) & (data['n_epochs'] == epoch)]
                        
                        if len(model_epoch_data) == 0:
                            continue
                        
                        # Sort by pii_rate for proper line plotting
                        model_epoch_data = model_epoch_data.sort_values('pii_rate')
                        
                        # Create label for legend - show epoch and model separately
                        model_display = 'Fine-tuned' if model == 'p_ft' else 'Pre-trained'
                        label = f"{model_display} ({format_epoch_label(epoch)})"
                        
                        # Plot line with error bars
                        ax.errorbar(
                            model_epoch_data['pii_rate'],
                            model_epoch_data['auc'],
                            yerr=model_epoch_data['auc_std'],
                            label=label,
                            color=model_colors[model],
                            linestyle=epoch_linestyles[k % len(epoch_linestyles)],
                            marker='o',
                            capsize=3,
                            capthick=1,
                            markersize=4
                        )
                
                # Customize subplot
                ax.set_xscale('log')
                ax.set_xlabel('PII Rate (%)')
                ax.set_ylabel('AUC')
                pii_type_display = format_pii_type_for_title(pii_type)
                ax.set_title(f'Prompt: {prompt}\nPII Type: {pii_type_display}')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_combined_visualization(df_results, use_shared_y_scale=True):
        """
        Create visualization plots for combined MI analysis.
        Plots MI as a function of pii_rate with error bars.
        One curve for each number of epochs.
        One subplot for each prompt and pii_type.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        # Get unique values for subplot organization
        prompts = df_results['prompt'].unique()
        pii_types = df_results['pii_type'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(len(prompts), len(pii_types), figsize=(10*len(pii_types), 4*len(prompts)))
        if len(prompts) == 1 and len(pii_types) == 1:
            axes = np.array([[axes]])
        elif len(prompts) == 1:
            axes = axes.reshape(1, -1)
        elif len(pii_types) == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for different epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Calculate shared y-scale if requested
        if use_shared_y_scale:
            all_values = []
            all_errors = []
            for i, prompt in enumerate(prompts):
                for j, pii_type in enumerate(pii_types):
                    data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                    for k, epoch in enumerate(epochs):
                        epoch_data = data[data['n_epochs'] == epoch]
                        if len(epoch_data) > 0:
                            all_values.append(epoch_data['I'].iloc[0])
                            all_errors.append(epoch_data['I_std'].iloc[0])
            
            # Calculate shared y-limits with some padding
            if all_values:
                y_min = min(all_values) - max(all_errors) - 0.1
                y_max = max(all_values) + max(all_errors) + 0.1
                # Ensure y_min is not negative for MI values
                y_min = max(0, y_min)
            else:
                y_min, y_max = 0, 1
        
        for i, prompt in enumerate(prompts):
            for j, pii_type in enumerate(pii_types):
                ax = axes[i, j]
                
                # Filter data for this prompt and pii_type
                data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot curves for each epoch
                for k, epoch in enumerate(epochs):
                    epoch_data = data[data['n_epochs'] == epoch]
                    
                    if len(epoch_data) == 0:
                        continue
                    
                    # Sort by pii_rate for proper line plotting
                    epoch_data = epoch_data.sort_values('pii_rate')
                    
                    # Create label for legend
                    label = f"Joint over models ({format_epoch_label(epoch)})"
                    
                    # Plot line with error bars
                    ax.errorbar(
                        epoch_data['pii_rate'],
                        epoch_data['I'],
                        yerr=epoch_data['I_std'],
                        label=label,
                        color=colors[k],
                        linestyle='-',
                        marker='o',
                        capsize=3,
                        capthick=1,
                        markersize=4
                    )
            
            # Customize subplot
            ax.set_xscale('log')
            ax.set_xlabel('PII Rate (%)')
            ax.set_ylabel('Mutual Information (bits)')
            pii_type_display = format_pii_type_for_title(pii_type)
            ax.set_title(f'Prompt: {prompt}\nPII Type: {pii_type_display}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Apply shared y-scale if requested
            if use_shared_y_scale:
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig
    
    def create_combined_auc_visualization(df_results, use_shared_y_scale=True):
        """
        Create visualization plots for combined AUC analysis.
        Plots AUC as a function of pii_rate with error bars.
        One curve for each number of epochs.
        One subplot for each prompt and pii_type.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        # Get unique values for subplot organization
        prompts = df_results['prompt'].unique()
        pii_types = df_results['pii_type'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(len(prompts), len(pii_types), figsize=(10*len(pii_types), 4*len(prompts)))
        if len(prompts) == 1 and len(pii_types) == 1:
            axes = np.array([[axes]])
        elif len(prompts) == 1:
            axes = axes.reshape(1, -1)
        elif len(pii_types) == 1:
            axes = axes.reshape(-1, 1)
        
        # Color map for different epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Calculate shared y-scale if requested
        if use_shared_y_scale:
            all_values = []
            all_errors = []
            for i, prompt in enumerate(prompts):
                for j, pii_type in enumerate(pii_types):
                    data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                    for k, epoch in enumerate(epochs):
                        epoch_data = data[data['n_epochs'] == epoch]
                        if len(epoch_data) > 0:
                            all_values.append(epoch_data['auc'].iloc[0])
                            all_errors.append(epoch_data['auc_std'].iloc[0])
            
            # Calculate shared y-limits with some padding
            if all_values:
                y_min = min(all_values) - max(all_errors) - 0.01
                y_max = min(1.0, max(all_values) + max(all_errors) + 0.01)  # AUC max is 1.0
                # Ensure y_min is reasonable for AUC values
                y_min = max(0.4, y_min)  # AUC should be > 0.5 for meaningful results
            else:
                y_min, y_max = 0.4, 1.0
        
        for i, prompt in enumerate(prompts):
            for j, pii_type in enumerate(pii_types):
                ax = axes[i, j]
                
                # Filter data for this prompt and pii_type
                data = df_results[(df_results['prompt'] == prompt) & (df_results['pii_type'] == pii_type)]
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot curves for each epoch
                for k, epoch in enumerate(epochs):
                    epoch_data = data[data['n_epochs'] == epoch]
                    
                    if len(epoch_data) == 0:
                        continue
                    
                    # Sort by pii_rate for proper line plotting
                    epoch_data = epoch_data.sort_values('pii_rate')
                    
                    # Create label for legend
                    label = f"Joint over models ({format_epoch_label(epoch)})"
                    
                    # Plot line with error bars
                    ax.errorbar(
                        epoch_data['pii_rate'],
                        epoch_data['auc'],
                        yerr=epoch_data['auc_std'],
                        label=label,
                        color=colors[k],
                        linestyle='-',
                        marker='o',
                        capsize=3,
                        capthick=1,
                        markersize=4
                    )
            
            # Customize subplot
            ax.set_xscale('log')
            ax.set_xlabel('PII Rate (%)')
            ax.set_ylabel('AUC')
            pii_type_display = format_pii_type_for_title(pii_type)
            ax.set_title(f'Prompt: {prompt}\nPII Type: {pii_type_display}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Apply shared y-scale if requested
            if use_shared_y_scale:
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig
    
    def create_all_visualization(df_results, use_shared_y_scale=True):
        """
        Create visualization plots for overall combined results (all prompts and models).
        Plots MI as a function of pii_rate with error bars.
        One curve for each number of epochs.
        One subplot for each pii_type.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        # Get unique values for subplot organization
        pii_types = df_results['pii_type'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(1, len(pii_types), figsize=(10*len(pii_types), 6))
        if len(pii_types) == 1:
            axes = np.array([axes])
        
        # Color map for different epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Calculate shared y-scale if requested
        if use_shared_y_scale:
            all_values = []
            all_errors = []
            for pii_type in pii_types:
                data = df_results[df_results['pii_type'] == pii_type]
                for k, epoch in enumerate(epochs):
                    epoch_data = data[data['n_epochs'] == epoch]
                    if len(epoch_data) > 0:
                        all_values.append(epoch_data['I'].iloc[0])
                        all_errors.append(epoch_data['I_std'].iloc[0])
            
            # Calculate shared y-limits with some padding
            if all_values:
                y_min = min(all_values) - max(all_errors) - 0.1
                y_max = max(all_values) + max(all_errors) + 0.1
                # Ensure y_min is not negative for MI values
                y_min = max(0, y_min)
            else:
                y_min, y_max = 0, 1
        
        for j, pii_type in enumerate(pii_types):
            ax = axes[j]
            
            # Filter data for this pii_type
            data = df_results[df_results['pii_type'] == pii_type]
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot curves for each epoch
            for k, epoch in enumerate(epochs):
                epoch_data = data[data['n_epochs'] == epoch]
                
                if len(epoch_data) == 0:
                    continue
                
                # Sort by pii_rate for proper line plotting
                epoch_data = epoch_data.sort_values('pii_rate')
                
                # Create label for legend
                label = f"Joint over models and prompts ({format_epoch_label(epoch)})"
                
                # Plot line with error bars
                ax.errorbar(
                    epoch_data['pii_rate'],
                    epoch_data['I'],
                    yerr=epoch_data['I_std'],
                    label=label,
                    color=colors[k],
                    linestyle='-',
                    marker='o',
                    capsize=3,
                    capthick=1,
                    markersize=4
                )
            
            # Customize subplot
            ax.set_xscale('log')
            ax.set_xlabel('PII Rate (%)')
            ax.set_ylabel('Mutual Information (bits)')
            pii_type_display = format_pii_type_for_title(pii_type)
            ax.set_title(f'PII Type: {pii_type_display}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Apply shared y-scale if requested
            if use_shared_y_scale:
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig
    
    def create_all_auc_visualization(df_results, use_shared_y_scale=True):
        """
        Create visualization plots for overall combined AUC results (all prompts and models).
        Plots AUC as a function of pii_rate with error bars.
        One curve for each number of epochs.
        One subplot for each pii_type.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        # Get unique values for subplot organization
        pii_types = df_results['pii_type'].unique()
        epochs = sorted(df_results['n_epochs'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(1, len(pii_types), figsize=(10*len(pii_types), 6))
        if len(pii_types) == 1:
            axes = np.array([axes])
        
        # Color map for different epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Calculate shared y-scale if requested
        if use_shared_y_scale:
            all_values = []
            all_errors = []
            for pii_type in pii_types:
                data = df_results[df_results['pii_type'] == pii_type]
                for k, epoch in enumerate(epochs):
                    epoch_data = data[data['n_epochs'] == epoch]
                    if len(epoch_data) > 0:
                        all_values.append(epoch_data['auc'].iloc[0])
                        all_errors.append(epoch_data['auc_std'].iloc[0])
            
            # Calculate shared y-limits with some padding
            if all_values:
                y_min = min(all_values) - max(all_errors) - 0.01
                y_max = min(1.0, max(all_values) + max(all_errors) + 0.01)  # AUC max is 1.0
                # Ensure y_min is reasonable for AUC values
                y_min = max(0.4, y_min)  # AUC should be > 0.5 for meaningful results
            else:
                y_min, y_max = 0.4, 1.0
        
        for j, pii_type in enumerate(pii_types):
            ax = axes[j]
            
            # Filter data for this pii_type
            data = df_results[df_results['pii_type'] == pii_type]
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot curves for each epoch
            for k, epoch in enumerate(epochs):
                epoch_data = data[data['n_epochs'] == epoch]
                
                if len(epoch_data) == 0:
                    continue
                
                # Sort by pii_rate for proper line plotting
                epoch_data = epoch_data.sort_values('pii_rate')
                
                # Create label for legend
                label = f"Joint over models and prompts ({format_epoch_label(epoch)})"
                
                # Plot line with error bars
                ax.errorbar(
                    epoch_data['pii_rate'],
                    epoch_data['auc'],
                    yerr=epoch_data['auc_std'],
                    label=label,
                    color=colors[k],
                    linestyle='-',
                    marker='o',
                    capsize=3,
                    capthick=1,
                    markersize=4
                )
            
            # Customize subplot
            ax.set_xscale('log')
            ax.set_xlabel('PII Rate (%)')
            ax.set_ylabel('AUC')
            pii_type_display = format_pii_type_for_title(pii_type)
            ax.set_title(f'PII Type: {pii_type_display}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Apply shared y-scale if requested
            if use_shared_y_scale:
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_visualization(all_results, all_results_combined, all_results_all, use_shared_y_scale=True):
        """
        Create comprehensive visualization for each PII type.
        For each PII type: subplots for each (pii_rate, n_epochs) combination.
        Each subplot shows all MI estimates side by side with error bars.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        pii_types = all_results['pii_type'].unique()
        
        for pii_type in pii_types:
            # Filter data for this PII type
            df_individual = all_results[all_results['pii_type'] == pii_type]
            df_combined = all_results_combined[all_results_combined['pii_type'] == pii_type]
            df_all = all_results_all[all_results_all['pii_type'] == pii_type]
            
            # Get unique combinations of pii_rate and n_epochs
            combinations = df_individual[['pii_rate', 'n_epochs']].drop_duplicates().sort_values(['pii_rate', 'n_epochs'])
            
            if len(combinations) == 0:
                continue
                
            # Create subplots
            n_combinations = len(combinations)
            n_cols = min(4, n_combinations)  # Max 4 columns
            n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_combinations == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            else:
                axes = axes.flatten()
            
            # Colors for different estimates
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            # Calculate shared y-scale if requested
            if use_shared_y_scale:
                all_values = []
                all_errors = []
                for _, row in combinations.iterrows():
                    pii_rate = row['pii_rate']
                    n_epochs = row['n_epochs']
                    
                    # Get data for this combination
                    individual_data = df_individual[(df_individual['pii_rate'] == pii_rate) & (df_individual['n_epochs'] == n_epochs)]
                    combined_data = df_combined[(df_combined['pii_rate'] == pii_rate) & (df_combined['n_epochs'] == n_epochs)]
                    all_data = df_all[(df_all['pii_rate'] == pii_rate) & (df_all['n_epochs'] == n_epochs)]
                    
                    # Collect all values and errors
                    for _, model_row in individual_data.iterrows():
                        all_values.append(model_row['I'])
                        all_errors.append(model_row['I_std'])
                    
                    for _, combined_row in combined_data.iterrows():
                        all_values.append(combined_row['I'])
                        all_errors.append(combined_row['I_std'])
                    
                    if len(all_data) > 0:
                        all_values.append(all_data.iloc[0]['I'])
                        all_errors.append(all_data.iloc[0]['I_std'])
                
                # Calculate shared y-limits with some padding
                if all_values:
                    y_min = min(all_values) - max(all_errors) - 0.1
                    y_max = max(all_values) + max(all_errors) + 0.1
                    # Ensure y_min is not negative for MI values
                    y_min = max(0, y_min)
                else:
                    y_min, y_max = 0, 1
            
            for idx, (_, row) in enumerate(combinations.iterrows()):
                ax = axes[idx] if n_combinations > 1 else axes[0]
                
                pii_rate = row['pii_rate']
                n_epochs = row['n_epochs']
                
                # Get data for this combination
                individual_data = df_individual[(df_individual['pii_rate'] == pii_rate) & (df_individual['n_epochs'] == n_epochs)]
                combined_data = df_combined[(df_combined['pii_rate'] == pii_rate) & (df_combined['n_epochs'] == n_epochs)]
                all_data = df_all[(df_all['pii_rate'] == pii_rate) & (df_all['n_epochs'] == n_epochs)]
                
                # Prepare data for plotting
                labels = []
                values = []
                errors = []
                color_list = []
                
                # Individual model results
                individual_count = 0
                for _, model_row in individual_data.iterrows():
                    model = model_row['model']
                    prompt = model_row['prompt']
                    labels.append(f"{model}_{prompt.strip()}")
                    values.append(model_row['I'])
                    errors.append(model_row['I_std'])
                    color_list.append(colors[len(labels)-1])
                    individual_count += 1
                
                # Combined results
                combined_count = 0
                for _, combined_row in combined_data.iterrows():
                    prompt = combined_row['prompt']
                    labels.append(f"Joint over models_{prompt.strip()}")
                    values.append(combined_row['I'])
                    errors.append(combined_row['I_std'])
                    color_list.append(colors[len(labels)-1])
                    combined_count += 1
                
                # All results
                all_count = 0
                if len(all_data) > 0:
                    labels.append("Joint over models and prompts")
                    values.append(all_data.iloc[0]['I'])
                    errors.append(all_data.iloc[0]['I_std'])
                    color_list.append(colors[len(labels)-1])
                    all_count += 1
                
                # Create bar plot
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, values, yerr=errors, capsize=5, color=color_list, alpha=0.7, edgecolor='black')
                
                # Add vertical separator lines
                if individual_count > 0 and combined_count > 0:
                    # Line between individual and combined
                    separator_x = individual_count - 0.5
                    ax.axvline(x=separator_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                if combined_count > 0 and all_count > 0:
                    # Line between combined and all
                    separator_x = individual_count + combined_count - 0.5
                    ax.axvline(x=separator_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # Customize subplot
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Mutual Information (bits)')
                ax.set_title(f'PII Rate: {format_pii_rate(pii_rate)}, {format_epoch_label(n_epochs)}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Apply shared y-scale if requested
                if use_shared_y_scale:
                    ax.set_ylim(y_min, y_max)
                
                # Add value labels on bars
                for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Hide empty subplots
            for idx in range(n_combinations, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(plots_dir, f'comprehensive_{pii_type}_{model_suffix}_{dataset_size}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comprehensive visualization saved to: {filename}")
    
    def create_comprehensive_auc_visualization(all_results, all_results_combined, all_results_all, use_shared_y_scale=True):
        """
        Create comprehensive AUC visualization for each PII type.
        For each PII type: subplots for each (pii_rate, n_epochs) combination.
        Each subplot shows all AUC estimates side by side with error bars.
        
        Parameters:
        -----------
        use_shared_y_scale : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        pii_types = all_results['pii_type'].unique()
        
        for pii_type in pii_types:
            # Filter data for this PII type
            df_individual = all_results[all_results['pii_type'] == pii_type]
            df_combined = all_results_combined[all_results_combined['pii_type'] == pii_type]
            df_all = all_results_all[all_results_all['pii_type'] == pii_type]
            
            # Get unique combinations of pii_rate and n_epochs
            combinations = df_individual[['pii_rate', 'n_epochs']].drop_duplicates().sort_values(['pii_rate', 'n_epochs'])
            
            if len(combinations) == 0:
                continue
                
            # Create subplots
            n_combinations = len(combinations)
            n_cols = min(4, n_combinations)  # Max 4 columns
            n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_combinations == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            else:
                axes = axes.flatten()
            
            # Colors for different estimates
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            # Calculate shared y-scale if requested
            if use_shared_y_scale:
                all_values = []
                all_errors = []
                for _, row in combinations.iterrows():
                    pii_rate = row['pii_rate']
                    n_epochs = row['n_epochs']
                    
                    # Get data for this combination
                    individual_data = df_individual[(df_individual['pii_rate'] == pii_rate) & (df_individual['n_epochs'] == n_epochs)]
                    combined_data = df_combined[(df_combined['pii_rate'] == pii_rate) & (df_combined['n_epochs'] == n_epochs)]
                    all_data = df_all[(df_all['pii_rate'] == pii_rate) & (df_all['n_epochs'] == n_epochs)]
                    
                    # Collect all values and errors
                    for _, model_row in individual_data.iterrows():
                        all_values.append(model_row['auc'])
                        all_errors.append(model_row['auc_std'])
                    
                    for _, combined_row in combined_data.iterrows():
                        all_values.append(combined_row['auc'])
                        all_errors.append(combined_row['auc_std'])
                    
                    if len(all_data) > 0:
                        all_values.append(all_data.iloc[0]['auc'])
                        all_errors.append(all_data.iloc[0]['auc_std'])
                
                # Calculate shared y-limits with some padding
                if all_values:
                    y_min = min(all_values) - max(all_errors) - 0.01
                    y_max = min(1.0, max(all_values) + max(all_errors) + 0.01)  # AUC max is 1.0
                    # Ensure y_min is reasonable for AUC values
                    y_min = max(0.4, y_min)  # AUC should be > 0.5 for meaningful results
                else:
                    y_min, y_max = 0.4, 1.0
            
            for idx, (_, row) in enumerate(combinations.iterrows()):
                ax = axes[idx] if n_combinations > 1 else axes[0]
                
                pii_rate = row['pii_rate']
                n_epochs = row['n_epochs']
                
                # Get data for this combination
                individual_data = df_individual[(df_individual['pii_rate'] == pii_rate) & (df_individual['n_epochs'] == n_epochs)]
                combined_data = df_combined[(df_combined['pii_rate'] == pii_rate) & (df_combined['n_epochs'] == n_epochs)]
                all_data = df_all[(df_all['pii_rate'] == pii_rate) & (df_all['n_epochs'] == n_epochs)]
                
                # Prepare data for plotting
                labels = []
                values = []
                errors = []
                color_list = []
                
                # Individual model results
                individual_count = 0
                for _, model_row in individual_data.iterrows():
                    model = model_row['model']
                    prompt = model_row['prompt']
                    labels.append(f"{model}_{prompt.strip()}")
                    values.append(model_row['auc'])
                    errors.append(model_row['auc_std'])
                    color_list.append(colors[len(labels)-1])
                    individual_count += 1
                
                # Combined results
                combined_count = 0
                for _, combined_row in combined_data.iterrows():
                    prompt = combined_row['prompt']
                    labels.append(f"Joint over models_{prompt.strip()}")
                    values.append(combined_row['auc'])
                    errors.append(combined_row['auc_std'])
                    color_list.append(colors[len(labels)-1])
                    combined_count += 1
                
                # All results
                all_count = 0
                if len(all_data) > 0:
                    labels.append("Joint over models and prompts")
                    values.append(all_data.iloc[0]['auc'])
                    errors.append(all_data.iloc[0]['auc_std'])
                    color_list.append(colors[len(labels)-1])
                    all_count += 1
                
                # Create bar plot
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, values, yerr=errors, capsize=5, color=color_list, alpha=0.7, edgecolor='black')
                
                # Add vertical separator lines
                if individual_count > 0 and combined_count > 0:
                    # Line between individual and combined
                    separator_x = individual_count - 0.5
                    ax.axvline(x=separator_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                if combined_count > 0 and all_count > 0:
                    # Line between combined and all
                    separator_x = individual_count + combined_count - 0.5
                    ax.axvline(x=separator_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # Customize subplot
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('AUC')
                ax.set_title(f'PII Rate: {format_pii_rate(pii_rate)}, {format_epoch_label(n_epochs)}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Apply shared y-scale if requested
                if use_shared_y_scale:
                    ax.set_ylim(y_min, y_max)
                
                # Add value labels on bars
                for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Hide empty subplots
            for idx in range(n_combinations, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(plots_dir, f'comprehensive_auc_{pii_type}_{model_suffix}_{dataset_size}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comprehensive AUC visualization saved to: {filename}")
    
    def create_auc_ci_plots(all_results_all, model_suffix, dataset_size, plots_dir):
        """
        Create plots with subplots for overfit/no-overfit showing AUC with confidence intervals.
        Similar to overall_proba.py plots, but for AUC from joint over models and prompts.
        
        Parameters:
        -----------
        all_results_all : DataFrame
            DataFrame containing joint over models and prompts results with columns:
            pii_type, dataset_size, pii_rate, n_epochs, auc, auc_std
        model_suffix : str
            Model suffix for filename
        dataset_size : int
            Dataset size for filename
        plots_dir : str
            Directory to save plots
        """
        # Convert std to confidence intervals (assuming normal distribution for bootstrap std)
        # For 95% CI: mean ± 1.96 * std
        z_score = 1.96  # For 95% confidence interval
        
        # Get unique PII types
        pii_types = all_results_all['pii_type'].unique()
        
        for pii_type in pii_types:
            # Filter data for this PII type
            df_pii = all_results_all[all_results_all['pii_type'] == pii_type].copy()
            
            if len(df_pii) == 0:
                continue
            
            # Add epoch_label column
            df_pii['epoch_label'] = df_pii['n_epochs'].map({2: 'no overfit', 3: 'no overfit', 10: 'overfit'})
            df_pii['pii_rate_percent'] = df_pii['pii_rate'] * 100
            
            # Calculate confidence intervals
            df_pii['ci_lower'] = df_pii['auc'] - z_score * df_pii['auc_std']
            df_pii['ci_upper'] = df_pii['auc'] + z_score * df_pii['auc_std']
            
            # Group by pii_rate and epoch_label
            df_grouped = df_pii.groupby(['pii_rate', 'epoch_label']).agg({
                'auc': 'mean',  # In case there are multiple entries
                'ci_lower': 'mean',
                'ci_upper': 'mean',
                'pii_rate_percent': 'first'
            }).reset_index()
            
            # === Plot: AUC with Confidence Intervals ===
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Subplot 1: No overfit
            ax1 = axes[0]
            df_no_overfit = df_grouped[df_grouped['epoch_label'] == 'no overfit'].sort_values('pii_rate')
            if len(df_no_overfit) > 0:
                x_pos = range(len(df_no_overfit))
                means = df_no_overfit['auc'].values
                ci_lower = df_no_overfit['ci_lower'].values
                ci_upper = df_no_overfit['ci_upper'].values
                
                ax1.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], 
                            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                            color='blue', label='Mean with 95% CI')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([f"{p:.0f}%" for p in df_no_overfit['pii_rate_percent'].values])
                ax1.set_xlabel('PII Rate (%)', fontsize=14)
                ax1.set_ylabel('AUC', fontsize=14)
                ax1.set_title('No Overfit Models', fontsize=16, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=12)
                ax1.set_ylim([0.4, 1.0])  # AUC range
            
            # Subplot 2: Overfit
            ax2 = axes[1]
            df_overfit = df_grouped[df_grouped['epoch_label'] == 'overfit'].sort_values('pii_rate')
            if len(df_overfit) > 0:
                x_pos = range(len(df_overfit))
                means = df_overfit['auc'].values
                ci_lower = df_overfit['ci_lower'].values
                ci_upper = df_overfit['ci_upper'].values
                
                ax2.errorbar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], 
                            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                            color='red', label='Mean with 95% CI')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([f"{p:.0f}%" for p in df_overfit['pii_rate_percent'].values])
                ax2.set_xlabel('PII Rate (%)', fontsize=14)
                ax2.set_ylabel('AUC', fontsize=14)
                ax2.set_title('Overfit Models', fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=12)
                ax2.set_ylim([0.4, 1.0])  # AUC range
            
            pii_type_display = format_pii_type_for_title(pii_type)
            plt.suptitle(f'AUC (Joint over models and prompts) with 95% Confidence Intervals - {pii_type_display}', 
                        fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(plots_dir, f'auc_ci_joint_{pii_type}_{model_suffix}_{dataset_size}.png')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"AUC CI plot saved to: {filename}")
    
    def create_correlation_plot(all_results, all_results_combined, all_results_all):
        """
        Create correlation plot showing AUC vs MI for individual, combined, and all results.
        Each point represents one configuration (pii_type, dataset_size, pii_rate, n_epochs).
        Different colors and markers for different categories.
        Includes error bars for both MI (x-axis) and AUC (y-axis) dimensions.
        """
        # Prepare data for plotting
        categories = []
        mi_values = []
        auc_values = []
        mi_errors = []
        auc_errors = []
        labels = []
        colors = []
        markers = []
        
        # Individual results
        if len(all_results) > 0:
            for _, row in all_results.iterrows():
                categories.append('Individual')
                mi_values.append(row['I'])
                auc_values.append(row['auc'])
                mi_errors.append(row['I_std'])
                auc_errors.append(row['auc_std'])
                labels.append(f"{row['model']}_{row['prompt'].strip()}")
                colors.append('#1f77b4')  # Blue
                markers.append('o')
        
        # Combined results
        if len(all_results_combined) > 0:
            for _, row in all_results_combined.iterrows():
                categories.append('Combined')
                mi_values.append(row['I'])
                auc_values.append(row['auc'])
                mi_errors.append(row['I_std'])
                auc_errors.append(row['auc_std'])
                labels.append(f"Joint over models_{row['prompt'].strip()}")
                colors.append('#ff7f0e')  # Orange
                markers.append('s')
        
        # All results
        if len(all_results_all) > 0:
            for _, row in all_results_all.iterrows():
                categories.append('All')
                mi_values.append(row['I'])
                auc_values.append(row['auc'])
                mi_errors.append(row['I_std'])
                auc_errors.append(row['auc_std'])
                labels.append("Joint over models and prompts")
                colors.append('#2ca02c')  # Green
                markers.append('^')
        
        if len(mi_values) == 0:
            print("No data available for correlation plot.")
            return
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot points for each category with error bars
        individual_mask = [cat == 'Individual' for cat in categories]
        combined_mask = [cat == 'Combined' for cat in categories]
        all_mask = [cat == 'All' for cat in categories]
        
        if any(individual_mask):
            individual_mi = [mi for i, mi in enumerate(mi_values) if individual_mask[i]]
            individual_auc = [auc for i, auc in enumerate(auc_values) if individual_mask[i]]
            individual_mi_err = [err for i, err in enumerate(mi_errors) if individual_mask[i]]
            individual_auc_err = [err for i, err in enumerate(auc_errors) if individual_mask[i]]
            
            ax.errorbar(individual_mi, individual_auc, 
                       xerr=individual_mi_err, yerr=individual_auc_err,
                       fmt='o', c='#1f77b4', markersize=6, 
                       alpha=0.7, label='Individual Models and Prompts', 
                       capsize=3, capthick=1, elinewidth=1)
        
        if any(combined_mask):
            combined_mi = [mi for i, mi in enumerate(mi_values) if combined_mask[i]]
            combined_auc = [auc for i, auc in enumerate(auc_values) if combined_mask[i]]
            combined_mi_err = [err for i, err in enumerate(mi_errors) if combined_mask[i]]
            combined_auc_err = [err for i, err in enumerate(auc_errors) if combined_mask[i]]
            
            ax.errorbar(combined_mi, combined_auc, 
                       xerr=combined_mi_err, yerr=combined_auc_err,
                       fmt='s', c='#ff7f0e', markersize=6, 
                       alpha=0.7, label='Joint over models', 
                       capsize=3, capthick=1, elinewidth=1)
        
        if any(all_mask):
            all_mi = [mi for i, mi in enumerate(mi_values) if all_mask[i]]
            all_auc = [auc for i, auc in enumerate(auc_values) if all_mask[i]]
            all_mi_err = [err for i, err in enumerate(mi_errors) if all_mask[i]]
            all_auc_err = [err for i, err in enumerate(auc_errors) if all_mask[i]]
            
            ax.errorbar(all_mi, all_auc, 
                       xerr=all_mi_err, yerr=all_auc_err,
                       fmt='^', c='#2ca02c', markersize=8, 
                       alpha=0.7, label='Joint over models and prompts', 
                       capsize=3, capthick=1, elinewidth=1)
        
        # Add correlation line for all points
        if len(mi_values) > 1:
            correlation = np.corrcoef(mi_values, auc_values)[0, 1]
            z = np.polyfit(mi_values, auc_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(mi_values), max(mi_values), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                   label=f'Correlation Line (r={correlation:.3f})')
        
        # Customize the plot
        ax.set_xlabel('Mutual Information (bits)', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Correlation: AUC vs Mutual Information\n(Individual, Joint over models, and Joint over models and prompts)\nError bars show 95% CI', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add text box with correlation statistics
        if len(mi_values) > 1:
            correlation = np.corrcoef(mi_values, auc_values)[0, 1]
            r_squared = correlation ** 2
            textstr = f'Correlation: {correlation:.3f}\nR²: {r_squared:.3f}\nN: {len(mi_values)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        # Set axis limits with padding to account for error bars
        mi_min = min(mi_values) - max(mi_errors) - 0.1
        mi_max = max(mi_values) + max(mi_errors) + 0.1
        auc_min = max(-0.05, min(auc_values) - max(auc_errors) - 0.05)
        auc_max = min(1.05, max(auc_values) + max(auc_errors) + 0.05)
        
        ax.set_xlim(mi_min, mi_max)
        ax.set_ylim(auc_min, auc_max)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(plots_dir, f'correlation_plot_all_{model}_{dataset_size}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation plot saved to: {filename}")
    
    def create_mi_difference_plot(all_results, share_y_axis=True):
        """
        Create bar plot showing I_ft - I_pre for each prompt with error bars.
        Shows the improvement in MI from pre-trained to fine-tuned models.
        Creates one plot with subplots for each configuration (pii_type, dataset_size, pii_rate, n_epochs).
        
        Parameters:
        -----------
        share_y_axis : bool, default True
            If True, all subplots will use the same y-axis scale for easy comparison.
            If False, each subplot will auto-scale to its data range.
        """
        if len(all_results) == 0:
            print("No individual results available for MI difference plot.")
            return
        
        # Get unique configurations
        configs = all_results[['pii_type', 'dataset_size', 'pii_rate', 'n_epochs']].drop_duplicates().sort_values(['pii_type', 'dataset_size', 'pii_rate', 'n_epochs'])
        
        n_configs = len(configs)
        if n_configs == 0:
            print("No configurations found for MI difference plot.")
            return
        
        # Calculate subplot layout
        n_cols = min(4, n_configs)  # Max 4 columns
        n_rows = (n_configs + n_cols - 1) // n_cols  # Ceiling division
        
        # Calculate shared y-axis limits if requested
        if share_y_axis:
            all_differences = []
            all_errors = []
            for _, config in configs.iterrows():
                pii_type = config['pii_type']
                dataset_size = config['dataset_size']
                pii_rate = config['pii_rate']
                n_epochs = config['n_epochs']
                
                # Filter data for this configuration
                data = all_results[
                    (all_results['pii_type'] == pii_type) &
                    (all_results['dataset_size'] == dataset_size) &
                    (all_results['pii_rate'] == pii_rate) &
                    (all_results['n_epochs'] == n_epochs)
                ]
                
                if len(data) == 0:
                    continue
                
                # Get unique prompts for this configuration
                prompts = data['prompt'].unique()
                
                for prompt in prompts:
                    prompt_data = data[data['prompt'] == prompt]
                    
                    if len(prompt_data) == 0:
                        continue
                    
                    # Get pre-trained and fine-tuned results
                    pre_data = prompt_data[prompt_data['model'] == 'p_pre']
                    ft_data = prompt_data[prompt_data['model'] == 'p_ft']
                    
                    if len(pre_data) == 0 or len(ft_data) == 0:
                        continue
                    
                    # Calculate difference (I_ft - I_pre)
                    I_pre = pre_data.iloc[0]['I']
                    I_ft = ft_data.iloc[0]['I']
                    I_pre_std = pre_data.iloc[0]['I_std']
                    I_ft_std = ft_data.iloc[0]['I_std']
                    
                    difference = I_ft - I_pre
                    error = np.sqrt(I_ft_std**2 + I_pre_std**2)
                    
                    all_differences.append(difference)
                    all_errors.append(error)
            
            # Calculate shared y-limits with some padding
            if all_differences:
                y_min = min(all_differences) - max(all_errors) - 0.1
                y_max = max(all_differences) + max(all_errors) + 0.1
            else:
                y_min, y_max = -0.5, 0.5
        else:
            y_min, y_max = None, None
        
        # Create the main figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_configs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.flatten()
        
        # Process each configuration
        for idx, (_, config) in enumerate(configs.iterrows()):
            ax = axes[idx] if n_configs > 1 else axes[0]
            
            pii_type = config['pii_type']
            dataset_size = config['dataset_size']
            pii_rate = config['pii_rate']
            n_epochs = config['n_epochs']
            
            # Filter data for this configuration
            data = all_results[
                (all_results['pii_type'] == pii_type) &
                (all_results['dataset_size'] == dataset_size) &
                (all_results['pii_rate'] == pii_rate) &
                (all_results['n_epochs'] == n_epochs)
            ]
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get unique prompts for this configuration
            prompts = data['prompt'].unique()
            
            # Calculate differences for each prompt
            differences = []
            errors = []
            prompt_labels = []
            
            for prompt in prompts:
                prompt_data = data[data['prompt'] == prompt]
                
                if len(prompt_data) == 0:
                    continue
                
                # Get pre-trained and fine-tuned results
                pre_data = prompt_data[prompt_data['model'] == 'p_pre']
                ft_data = prompt_data[prompt_data['model'] == 'p_ft']
                
                if len(pre_data) == 0 or len(ft_data) == 0:
                    continue
                
                # Calculate difference (I_ft - I_pre)
                I_pre = pre_data.iloc[0]['I']
                I_ft = ft_data.iloc[0]['I']
                I_pre_std = pre_data.iloc[0]['I_std']
                I_ft_std = ft_data.iloc[0]['I_std']
                
                difference = I_ft - I_pre
                # Error propagation: sqrt(std_ft^2 + std_pre^2)
                error = np.sqrt(I_ft_std**2 + I_pre_std**2)
                
                differences.append(difference)
                errors.append(error)
                prompt_labels.append(prompt.strip().rstrip(':').strip())
            
            if len(differences) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create bar plot
            x_pos = np.arange(len(prompt_labels))
            colors = ['#2ca02c' if diff > 0 else '#d62728' for diff in differences]  # Green for positive, red for negative
            
            bars = ax.bar(x_pos, differences, yerr=errors, capsize=3, 
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Customize subplot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(prompt_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('MI Difference (bits)', fontsize=10)
            pii_type_display = format_pii_type_for_title(pii_type)
            ax.set_title(f'{pii_type_display}\nSize: {dataset_size}, Rate: {format_pii_rate(pii_rate)}, {format_epoch_label(n_epochs)}', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # Add value labels on bars (smaller font for subplots)
            for i, (bar, diff, err) in enumerate(zip(bars, differences, errors)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01 if height >= 0 else height - err - 0.01,
                       f'{diff:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            # Apply shared y-axis limits if requested
            if share_y_axis and y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
        
        # Hide empty subplots
        for idx in range(n_configs, len(axes)):
            axes[idx].set_visible(False)
        
        # Add a single legend for the entire figure
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ca02c', alpha=0.7, label='Improvement'),
                         Patch(facecolor='#d62728', alpha=0.7, label='Degradation')]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        # Add overall title
        fig.suptitle('MI Improvement: Fine-tuned vs Pre-trained Models', fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # Make room for the suptitle and legend
        
        # Save the single plot with all subplots
        filename = os.path.join(plots_dir, f'mi_difference_all_configurations_all_{model}_{dataset_size}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"MI difference plot (all configurations) saved to: {filename}")
    
    # Create and save the visualization
    if len(all_results) > 0:
        # Create MI visualization
        fig = create_mi_visualization(all_results)
        plt.savefig(os.path.join(plots_dir, f'mi_visualization_all_{model}_{dataset_size}.png'), 
                   dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        print("MI Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/mi_visualization.png")
        
        # Create AUC visualization
        fig_auc = create_auc_visualization(all_results)
        plt.savefig(os.path.join(plots_dir, f'auc_visualization_all_{model}_{dataset_size}.png'), 
                   dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        print("AUC Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/auc_visualization.png")
        
        # Create combined visualizations
        if len(all_results_combined) > 0:
            # Create combined MI visualization
            fig_combined_mi = create_combined_visualization(all_results_combined, use_shared_y_scale=True)
            plt.savefig(os.path.join(plots_dir, f'mi_combined_visualization_all_{model}_{dataset_size}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("Joint over models MI Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/mi_combined_visualization.png")
            
            # Create combined AUC visualization
            fig_combined_auc = create_combined_auc_visualization(all_results_combined, use_shared_y_scale=True)
            plt.savefig(os.path.join(plots_dir, f'auc_combined_visualization_all_{model}_{dataset_size}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("Joint over models AUC Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/auc_combined_visualization.png")
        
        # Create all visualizations
        if len(all_results_all) > 0:
            # Create all MI visualization
            fig_all_mi = create_all_visualization(all_results_all, use_shared_y_scale=True)
            plt.savefig(os.path.join(plots_dir, f'mi_all_visualization_all_{model}_{dataset_size}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("Joint over models and prompts MI Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/mi_all_visualization.png")
            
            # Create all AUC visualization
            fig_all_auc = create_all_auc_visualization(all_results_all, use_shared_y_scale=True)
            plt.savefig(os.path.join(plots_dir, f'auc_all_visualization_all_{model}_{dataset_size}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("Joint over models and prompts AUC Visualization saved to: /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/mia/auc_all_visualization.png")
        
        # Create comprehensive visualizations
        create_comprehensive_visualization(all_results, all_results_combined, all_results_all, use_shared_y_scale=True)
        
        # Create comprehensive AUC visualizations
        create_comprehensive_auc_visualization(all_results, all_results_combined, all_results_all, use_shared_y_scale=True)
        
        # Create AUC CI plots (joint over models and prompts)
        if len(all_results_all) > 0:
            create_auc_ci_plots(all_results_all, model_suffix, dataset_size, plots_dir)
        
        # Create correlation plot
        create_correlation_plot(all_results, all_results_combined, all_results_all)
        
        # Create MI difference plot
        create_mi_difference_plot(all_results, share_y_axis=True)
    else:
        print("No results to visualize.")