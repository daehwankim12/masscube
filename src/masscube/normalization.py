# Author: Hauxu Yu

# A module for normalization
# There are two types of normalization:
# 1. Sample normalization - to normalize samples with different total amounts/concentrations.
# 2. Signal normalization - to address the signal drifts in the mass spectrometry data.

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

"""
Sample normalization
====================

Aims:

Sample normalization is to normalize samples with different total amounts/concentrations, 
such as urine samples, fecal samples, and tissue samples that are unweighted.

Modules:
    1. Select high-quality features for normalization using QC and blank samples.
    2. Find the reference sample.
    3. Find the normalization factors.
    4. Normalize samples by factors.
"""


def sample_normalization(feature_table, sample_metadata=None, method='pqn', feature_selection=True):
    """
    A normalization function that takes a feature table as input and returns a normalized feature table.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_metadata : pd.DataFrame
        DataFrame containing sample metadata. See params module for details.
    method : str
        The method to find the normalization factors.
        'pqn': probabilistic quotient normalization.
        'total_intensity': total intensity normalization.
        'median_intensity': median intensity normalization.
        'quantile': quantile normalization.
        'mdfc': maximum density fold change normalization (https://doi.org/10.1093/bioinformatics/btac355).
    feature_selection : bool
        Whether to select high-quality features for normalization. High-quality features have
        relative standard deviation (RSD) less than 25% in QC samples and average intensity in QC+biological samples
        greater than 2 fold of the average intensity in blank samples.

    Returns
    -------
    pandas DataFrame
        Normalized feature table.
    """

    if sample_metadata is None:
        print("\tSample normalization failed: sample metadata is required.")
        return feature_table

    data = feature_table[sample_metadata.iloc[:, 0]].values

    if feature_selection:
        hq_data = high_quality_feature_selection(data, is_qc=sample_metadata['is_qc'], is_blank=sample_metadata['is_blank'])
    else:
        hq_data = data
    
    # drop blank samples for normalization
    data_to_norm = data[:, ~sample_metadata['is_blank']]
    hq_data_to_norm = hq_data[:, ~sample_metadata['is_blank']]

    # STEP 3: find the normalization factors
    v = find_normalization_factors(hq_data_to_norm, method=method)

    # STEP 4: normalize samples by factors
    data_to_norm = sample_normalization_by_factors(data_to_norm, v)
    data[:, ~sample_metadata['is_blank']] = data_to_norm

    # STEP 5: update the feature table
    feature_table[sample_metadata.iloc[:, 0]] = data

    return feature_table


def find_normalization_factors(array, method='pqn'):
    """ 
    A function to find the normalization factors for a data frame.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    method : str
        The method to find the normalization factors.
        'pqn': probabilistic quotient normalization.

    Returns
    -------
    numpy array
        Normalization factor.
    """

    # find the reference sample
    ref_idx = find_reference_sample(array)
    ref_arr = array[:, ref_idx]

    factors = []
    if method == 'pqn':
        for i in range(array.shape[1]):
            a = array[:, i]
            common = np.logical_and(a > 0, ref_arr > 0)
            factors.append(np.median(a[common] / ref_arr[common]))
    return np.array(factors)
    

def sample_normalization_by_factors(array, v):
    """
    A function to normalize a data frame by a vector.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    v : numpy array
        The normalization factor.
    """

    # change all zeros to ones
    v[v == 0] = 1

    return np.array(array / v, dtype=np.int64)


def find_reference_sample(array, method='median_intensity'):
    """
    A function to find the reference sample for normalization.
    Note, samples are in columns and features are in rows.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    method : str
        The method to find the reference sample. 
        'number': the reference sample has the most detected features.
        'total_intensity': the reference sample has the highest total intensity.
        'median_intensity': the reference sample has the highest median intensity.

    Returns
    -------
    int
        The index of the reference sample.
    """

    if method == 'number':
        # find the reference sample with the most detected features
        return np.argmax(np.count_nonzero(array, axis=0))
    elif method == 'total_intensity':
        # find the reference sample with the highest total intensity
        return np.argmax(np.sum(array, axis=0))
    elif method == 'median_intensity':
        # find the reference sample with the highest median intensity
        return np.argmax(np.median(array, axis=0))


def high_quality_feature_selection(array, is_qc=None, is_blank=None, blank_ratio_tol=0.5, qc_rsd_tol=0.25):
    """
    Select high-quality features based on provided criteria for normalization.
    High-quality features have (default):
        1. relative standard deviation (RSD) less than 25% in QC samples and 
        2. average intensity in QC and biological samples greater than 0.5 fold of 
        the average intensity in blank samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized. Samples are in columns and features are in rows.
    is_qc : numpy array
        Boolean array indicating whether a sample is a quality control sample.
    is_blank : numpy array
        Boolean array indicating whether a sample is a blank sample.
    blank_ratio_tol : float
        The tolerance of the ratio of the average intensity in blank samples to the average intensity in QC and biological samples.
    qc_rsd_tol : float
        The tolerance of the relative standard deviation (RSD) in QC samples.

    Returns
    -------
    numpy array
        High-quality features. Features are in rows and samples are in columns.
    numpy array
        The index of the selected features.
    """

    # 1. filter features based on blank samples
    if is_blank is not None:
        blank_avg = np.mean(array[:, is_blank], axis=1)
        sample_ave = np.mean(array[:, ~is_blank], axis=1)
        sample_ave[sample_ave == 0] = 1     # avoid division by zero
        blank_pass = blank_avg / sample_ave < blank_ratio_tol
    else:
        blank_pass = np.ones(array.shape[0], dtype=bool)

    # 2. filter features based on QC samples (3 QC samples are required)
    if is_qc is not None and np.sum(is_qc) > 2:
        sd = np.std(array[:, is_qc], axis=1, ddof=1) 
        mean = np.mean(array[:, is_qc], axis=1)
        rsd = np.array([s/m if m != 0 else 99 for s, m in zip(sd, mean)])
        qc_pass = rsd < qc_rsd_tol
    else:
        qc_pass = np.ones(array.shape[0], dtype=bool)

    idxes = np.logical_and(blank_pass, qc_pass)

    return array[idxes]


"""
Signal normalization
====================

Provides
    1. Feature-wise normalization based on timestamp.
    2. Standard-free QC-based normalization.
"""

def signal_normalization(feature_table, sample_metadata, method='lowess', batch_idx=None):
    """
    A function to normalize MS signal drifts based on analytical order.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_metadata : pd.DataFrame
        DataFrame containing sample metadata. See params module for details.
    method : str
        The method to find the normalization factors.
        'lowess': locally weighted scatterplot smoothing.
    batch_idx : list
        Not used now. The index of the batches. It should have the same length as the number of samples.
        e.g., [1, 1,..., 2, 2,..., 3, 3,...]
    
    Returns
    -------
    pandas DataFrame
        Normalized feature table.
    """

    if np.sum(sample_metadata['is_qc']) < 3:
        print("\tSignal normalization failed: at least three QC samples are required.")
        return feature_table

    data = feature_table[sample_metadata.iloc[:, 0]].values

    if method == 'lowess':
        data_corr = lowess_normalization(array=data, is_qc=sample_metadata['is_qc'], 
                                         analytical_order=sample_metadata['analytical_order'], batch_idx=batch_idx)
    
    # STEP 3: update the feature table
    feature_table[sample_metadata.iloc[:, 0]] = data_corr

    return feature_table


def lowess_normalization(array, is_qc, analytical_order, batch_idx=None):
    """
    A function to normalize samples using quality control samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized. Samples are in columns and features are in rows.
    is_qc : numpy array
        Boolean array indicating whether a sample is a quality control sample.
    analytical_order : numpy array or list
        The order of the samples.
    batch_idx : numpy array or list
        The index of the batches. It should have the same length as the number of samples.
        e.g., [1, 1,..., 2, 2,..., 3, 3,...]

    Returns
    -------
    numpy array
        Normalized data.
    """

    # 1. reorder the data array
    array = array[:, np.argsort(analytical_order)]
    is_qc = is_qc[np.argsort(analytical_order)]
    qc_idx = np.where(is_qc)[0]

    # 2. feature-wise normalization
    data_corr = []
    corrected_arr = []
    for int_arr in array:
        # build loess model using qc samples
        qc_arr = int_arr[qc_idx]
        # only keep the positive values
        qc_idx_tmp = qc_idx[qc_arr > 0]
        qc_arr = qc_arr[qc_arr > 0]

        if len(qc_arr) > 2:
            model = lowess(qc_arr, qc_idx_tmp, frac=0.09, it=0)
            x_new = np.arange(len(int_arr))
            y_new = np.interp(x_new, model[:, 0], model[:, 1])
            y_new[y_new < 0] = np.min(y_new[y_new > 0])
            int_arr_corr = int_arr / y_new * np.min(y_new)
            corrected_arr.append(True)
        else:
            int_arr_corr = int_arr
            corrected_arr.append(False)

        data_corr.append(int_arr_corr)

    data_corr = np.array(data_corr)
    # sort the data back to the original order
    data_corr = data_corr[:, np.argsort(analytical_order)]

    return data_corr