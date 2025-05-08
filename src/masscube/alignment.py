# Author: Huaxu Yu

# A module to align features (characterized by unique m/z and retention time) from different files. 

# imports
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import pickle

from .raw_data_utils import read_raw_file_to_obj
from .params import Params
from .utils_functions import convert_signals_to_string, extract_signals_from_string

"""
Classes
------------------------------------------------------------------------------------------------------------------------
"""


class AlignedFeature:
    """
    A class to model a feature in mass spectrometry data. Generally, a feature is defined as 
    a unique pair of m/z and retention time.
    """

    __slots__ = (
        # individual files
        'feature_id_arr', 'mz_arr', 'rt_arr', 'scan_idx_arr', 'peak_height_arr',
        'peak_area_arr', 'top_average_arr', 'ms2_seq', 'length_arr',
        'gaussian_similarity_arr', 'noise_score_arr', 'asymmetry_factor_arr',
        'sse_arr', 'is_segmented_arr',

        # summary
        'id', 'feature_group_id', 'mz', 'rt', 'reference_file', 'reference_scan_idx',
        'highest_intensity', 'ms2', 'ms2_reference_file', 'gaussian_similarity',
        'noise_score', 'asymmetry_factor', 'detection_rate', 'detection_rate_gap_filled',
        'charge_state', 'is_isotope', 'isotope_signals', 'is_in_source_fragment',
        'adduct_type', 'annotation_algorithm', 'search_mode', 'similarity', 'annotation',
        'formula', 'matched_peak_number', 'smiles', 'inchikey', 'matched_precursor_mz',
        'matched_retention_time', 'matched_adduct_type', 'matched_ms2'
    )

    def __init__(self, file_number=1):
        """
        Define the attributes of a aligned feature.

        Parameters
        ----------
        file_number: int
            The number of files.
        """

        # individual files
        self.feature_id_arr = -np.ones(file_number,
                                       dtype=int)  # feature ID from individual files (-1 if not detected or gap filled)
        self.mz_arr = np.zeros(file_number)  # m/z
        self.rt_arr = np.zeros(file_number)  # retention time
        self.scan_idx_arr = np.zeros(file_number, dtype=int)  # scan index of the peak apex
        self.peak_height_arr = np.zeros(file_number)  # peak height
        self.peak_area_arr = np.zeros(file_number)  # peak area
        self.top_average_arr = np.zeros(file_number)  # average of the highest three intensities
        self.ms2_seq = []  # representative MS2 spectrum from each file (default: highest total intensity)
        self.length_arr = np.zeros(file_number, dtype=int)  # length (i.e. non-zero scans in the peak)
        self.gaussian_similarity_arr = np.zeros(file_number)  # Gaussian similarity
        self.noise_score_arr = np.zeros(file_number)  # noise score
        self.asymmetry_factor_arr = np.zeros(file_number)  # asymmetry factor
        self.sse_arr = np.zeros(file_number)  # squared error to the smoothed curve
        self.is_segmented_arr = np.zeros(file_number, dtype=bool)  # whether the peak is segmented

        # summary
        self.id = None  # index of the feature
        self.feature_group_id = None  # feature group ID
        self.mz = 0.0  # m/z
        self.rt = 0.0  # retention time
        self.reference_file = None  # the reference file with the highest peak height
        self.reference_scan_idx = None  # the scan index of the peak apex from the reference file
        self.highest_intensity = 0.0  # the highest peak height from individual files (which is the reference file)
        self.ms2 = None  # representative MS2 spectrum
        self.ms2_reference_file = None  # the reference file for the representative MS2 spectrum
        self.gaussian_similarity = 0.0  # Gaussian similarity from the reference file
        self.noise_score = 0.0  # noise level from the reference file
        self.asymmetry_factor = 0.0  # asymmetry factor from the reference file
        self.detection_rate = 0.0  # number of detected files / total number of files (blank not included)
        self.detection_rate_gap_filled = 0.0  # number of detected files after gap filling / total number of files (blank not included)
        self.charge_state = 1  # charge state
        self.is_isotope = False  # whether it is an isotope
        self.isotope_signals = None  # isotope signals [[m/z, intensity], ...]
        self.is_in_source_fragment = False  # whether it is an in-source fragment
        self.adduct_type = None  # adduct type

        self.annotation_algorithm = None  # annotation algorithm. Not used now.
        self.search_mode = None  # 'identity search', 'fuzzy search', or 'mzrt_search'
        self.similarity = None  # similarity score (0-1)
        self.annotation = None  # name of annotated compound
        self.formula = None  # molecular formula
        self.matched_peak_number = None  # number of matched peaks
        self.smiles = None  # SMILES
        self.inchikey = None  # InChIKey
        self.matched_precursor_mz = None  # matched precursor m/z
        self.matched_retention_time = None  # matched retention time
        self.matched_adduct_type = None  # matched adduct type
        self.matched_ms2 = None  # matched ms2 spectra


"""
Functions
------------------------------------------------------------------------------------------------------------------------
"""


def feature_alignment(path: str, params: Params):
    """
    Align the features from multiple processed single files as .txt format.

    Parameters
    ----------
    path: str
        The path to the feature tables.
    params: Params object
        The parameters for alignment including sample names and sample groups.

    Returns
    -------
    features: list
        A list of AlignedFeature objects.
    """

    # STEP 1: preparation
    features = []
    names = [os.path.join(path, f + ".txt") for f in params.sample_names]

    file_exists = np.array([os.path.exists(name) for name in names])
    if not np.all(file_exists):
        not_found_indices = np.where(~file_exists)[0]
        not_found_files = [params.sample_names[i] for i in not_found_indices]

        for f in not_found_files:
            params.problematic_files[f] = "file does not exist"

        # Use boolean indexing for faster filtering
        names = np.array(names)[file_exists].tolist()
        params.sample_metadata = params.sample_metadata[~params.sample_metadata.iloc[:, 1].isin(not_found_files)]
        params.sample_names = list(params.sample_metadata.iloc[:, 0])

    # find anchors for retention time correction
    rt_cor_functions = {}
    if params.correct_rt:
        intensities = np.array([
            pd.read_csv(n, sep="\t", low_memory=False, usecols=["peak_height"])["peak_height"].sum()
            for n in names
        ])

        # Direct median calculation
        median_idx = np.argsort(intensities)[len(intensities) // 2]
        anchor_selection_name = names[median_idx]
        mz_ref, rt_ref = rt_anchor_selection(anchor_selection_name, num=100)

    # STEP 2: read individual feature tables and align features
    for i, file_name in enumerate(tqdm(params.sample_names)):

        if i >= len(names) or not os.path.exists(names[i]):
            continue

        # read feature table
        current_table = pd.read_csv(names[i], low_memory=False, sep="\t")
        mask = current_table["MS2"].notna() | (current_table["total_scans"] > params.scan_number_cutoff)
        current_table = current_table[mask]

        # sort current table by peak height from high to low
        current_table.sort_values(by="peak_height", ascending=False, inplace=True)
        current_table.reset_index(drop=True, inplace=True)
        tmp_table = current_table[current_table["peak_height"] > params.ms1_abs_int_tol * 5]

        available_features = np.ones(len(current_table), dtype=bool)

        # Retention time correction
        if params.correct_rt:
            _, model = retention_time_correction(mz_ref, rt_ref, tmp_table["m/z"].values, tmp_table["RT"].values,
                                                 rt_tol=params.rt_tol_rt_correction)
            if model is not None:
                current_table.loc[:, "RT"] = model(current_table["RT"].values)
            rt_cor_functions[file_name] = model

        # Pre-compute arrays for frequent access
        mz_values = current_table["m/z"].values
        rt_values = current_table["RT"].values
        peak_heights = current_table["peak_height"].values

        for f in features:
            # Calculate all distances at once using numpy operations
            mz_diff = np.abs(f.mz - mz_values)
            rt_diff = np.abs(f.rt - rt_values)

            # Combined mask for all conditions
            match_mask = (mz_diff < params.mz_tol_alignment) & (rt_diff < params.rt_tol_alignment) & available_features
            matching_indices = np.where(match_mask)[0]

            if len(matching_indices) > 0:
                p = matching_indices[0]
                _assign_value_to_feature(f=f, df=current_table, i=i, p=p, file_name=file_name)
                available_features[p] = False
                # check if this file can be the reference file
                if peak_heights[p] > f.highest_intensity:
                    _assign_reference_values(f=f, df=current_table, i=i, p=p, file_name=file_name)

        # if an feature is not detected in the previous files, add it to the features
        new_feature_indices = np.where(available_features)[0]
        for j in new_feature_indices:
            f = AlignedFeature(file_number=len(params.sample_names))
            _assign_value_to_feature(f=f, df=current_table, i=i, p=j, file_name=file_name)
            _assign_reference_values(f=f, df=current_table, i=i, p=j, file_name=file_name)
            features.append(f)

        # summarize (calculate the average mz and rt) and reorder the features
        features.sort(key=lambda x: x.highest_intensity, reverse=True)

    # save the retention time correction models
    if params.correct_rt:
        with open(os.path.join(params.project_file_dir, "rt_correction_models.pkl"), "wb") as f:
            pickle.dump(rt_cor_functions, f)

    # choose the best ms2
    for f in features:
        if not f.ms2_seq:
            continue
        ms2_data = []
        for file_name, ms2 in f.ms2_seq:
            signals = extract_signals_from_string(ms2)
            if len(signals) > 0:
                total_intensity = np.sum(signals[:, 1])
                ms2_data.append((file_name, signals, total_intensity))
        if ms2_data:
            best_entry = max(ms2_data, key=lambda x: x[2])
            f.ms2_reference_file = best_entry[0]
            f.ms2 = convert_signals_to_string(best_entry[1])

    # STEP 3: calculate the detection rate and drop features using the detection rate cutoff
    is_blank = params.sample_metadata['is_blank'].values
    non_blank_mask = ~is_blank
    non_blank_count = np.sum(non_blank_mask)  # Calculate once instead of repeatedly

    # Calculate and filter in one pass
    filtered_features = []
    for f in features:
        f.detection_rate = np.sum(f.feature_id_arr[non_blank_mask] != -1) / non_blank_count
        if f.detection_rate > params.detection_rate_cutoff:
            filtered_features.append(f)

    features = filtered_features

    # STEP 4: clean features by merging the features with almost the same m/z and retention time
    if params.merge_features:
        features = merge_features(features, params)

    # STEP 5: gap filling
    print("\tFilling gaps...")
    if params.fill_gaps:
        features = gap_filling(features, params)

    # STEP 6: index the features
    features.sort(key=lambda x: x.highest_intensity, reverse=True)
    for i, f in enumerate(features):
        f.id = i

    return features


def gap_filling(features, params: Params):
    """
    Fill the gaps for aligned features.

    Parameters
    ----------------------------------------------------------
    features: list
        The aligned features.
    params: Params object
        The parameters for gap filling.

    Returns
    ----------------------------------------------------------
    features: list
        The aligned features with filled gaps.
    """

    # fill the gaps by forced peak picking (local maximum)
    if params.gap_filling_method == 'local_maximum':

        # if retention time correction is applied, read the model
        rt_cor_functions = None
        if params.correct_rt:
            rt_model_path = os.path.join(params.project_dir, "rt_correction_models.pkl")
            if os.path.exists(rt_model_path):
                with open(rt_model_path, "rb") as f:
                    rt_cor_functions = pickle.load(f)

        # Pre-identify which features need gap filling for each sample
        gap_filling_needed = {}
        for i, file_name in enumerate(params.sample_names):
            if gap_features := [
                j for j, f in enumerate(features) if f.feature_id_arr[i] == -1
            ]:
                gap_filling_needed[i] = gap_features

        # Process only samples that have gaps to fill
        for i, file_name in enumerate(tqdm(params.sample_names)):
            if i not in gap_filling_needed or not gap_filling_needed[i]:
                continue

            fn = os.path.join(params.tmp_file_dir, file_name + ".mzpkl")
            if not os.path.exists(fn):
                continue

            d = read_raw_file_to_obj(fn)

            # Apply retention time correction once per file if needed
            if rt_cor_functions is not None and file_name in rt_cor_functions:
                f_rt = rt_cor_functions[file_name]
                if f_rt is not None:
                    d.correct_retention_time(f_rt)

            for feat_idx in gap_filling_needed[i]:
                f = features[feat_idx]
                # Extract EIC and calculate metrics
                eic_time_arr, eic_signals, _ = d.get_eic_data(f.mz, f.rt, params.mz_tol_alignment,
                                                              params.gap_filling_rt_window)

                if len(eic_signals) > 0:
                    # Use NumPy vectorized operations for calculations
                    f.peak_height_arr[i] = np.max(eic_signals[:, 1])
                    f.peak_area_arr[i] = int(np.trapz(y=eic_signals[:, 1], x=eic_time_arr))
                    f.top_average_arr[i] = np.mean(np.sort(eic_signals[:, 1])[-3:])

    # calculate the detection rate after gap filling (blank samples are not included)
    non_blank_mask = ~params.sample_metadata['is_blank']
    non_blank_count = np.sum(non_blank_mask)

    if non_blank_count > 0:
        for f in features:
            f.detection_rate_gap_filled = np.sum(f.peak_height_arr[non_blank_mask] > 0) / non_blank_count

    return features


def merge_features(features: list, params: Params):
    """
    Clean features by merging features with almost the same m/z and retention time.

    Parameters
    ----------
    features: list
        A list of AlignedFeature objects.
    params: Params object
        The parameters for feature cleaning.

    Returns
    -------
    features: list
        A list of cleaned AlignedFeature objects.
    """
    if not features:
        return []

    features = sorted(features, key=lambda x: x.mz)
    features_cleaned = []

    def merge_feature_group(group):
        if len(group) == 1:
            return group[0]

        group.sort(key=lambda x: x.highest_intensity, reverse=True)
        merged_f = group[0]

        peak_arrays = np.array([f.peak_height_arr for f in group])
        area_arrays = np.array([f.peak_area_arr for f in group])
        top_avg_arrays = np.array([f.top_average_arr for f in group])

        merged_f.peak_height_arr = np.max(peak_arrays, axis=0)
        merged_f.peak_area_arr = np.max(area_arrays, axis=0)
        merged_f.top_average_arr = np.max(top_avg_arrays, axis=0)

        return merged_f

    current_group = [features[0]]

    for i in range(1, len(features)):
        if features[i].mz - features[i - 1].mz < params.mz_tol_merge_features:
            current_group.append(features[i])
        else:
            if len(current_group) == 1:
                features_cleaned.append(current_group[0])
            else:
                current_group.sort(key=lambda x: x.rt)
                rt_group = [current_group[0]]

                for j in range(1, len(current_group)):
                    if current_group[j].rt - current_group[j - 1].rt < params.rt_tol_merge_features:
                        rt_group.append(current_group[j])
                    else:
                        features_cleaned.append(merge_feature_group(rt_group))
                        rt_group = [current_group[j]]

                if rt_group:
                    features_cleaned.append(merge_feature_group(rt_group))

            current_group = [features[i]]

    if len(current_group) == 1:
        features_cleaned.append(current_group[0])
    elif len(current_group) > 1:
        current_group.sort(key=lambda x: x.rt)
        rt_group = [current_group[0]]

        for j in range(1, len(current_group)):
            if current_group[j].rt - current_group[j - 1].rt < params.rt_tol_merge_features:
                rt_group.append(current_group[j])
            else:
                features_cleaned.append(merge_feature_group(rt_group))
                rt_group = [current_group[j]]

        if rt_group:
            features_cleaned.append(merge_feature_group(rt_group))

    features_cleaned.sort(key=lambda x: x.highest_intensity, reverse=True)

    return features_cleaned


def convert_features_to_df(features, sample_names, quant_method="peak_height"):
    """
    Convert the aligned features to a DataFrame.

    Parameters
    ----------
    features : list
        list of features
    sample_names : list
        list of sample names
    quant_method : str
        quantification method, "peak_height", "peak_area" or "top_average"

    Returns
    -------
    feature_table : pd.DataFrame
        feature DataFrame
    """

    results = []
    sample_names = sample_names if isinstance(sample_names, list) else list(sample_names)
    columns = ["group_ID", "feature_ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment",
               "Gaussian_similarity", "noise_score",
               "asymmetry_factor", "detection_rate", "detection_rate_gap_filled", "alignment_reference_file", "charge",
               "isotopes", "MS2_reference_file", "MS2", "matched_MS2",
               "search_mode", "annotation", "formula", "similarity", "matched_peak_number", "SMILES",
               "InChIKey"] + sample_names

    if not features:
        return pd.DataFrame(columns=columns)

    for f in features:
        if quant_method == "peak_height":
            quant = list(f.peak_height_arr)
        elif quant_method == "peak_area":
            quant = list(f.peak_area_arr)
        elif quant_method == "top_average":
            quant = list(f.top_average_arr)

        quant = [int(x) for x in quant]

        results.append([f.feature_group_id, f.id, f.mz, f.rt, f.adduct_type, f.is_isotope, f.is_in_source_fragment,
                        f.gaussian_similarity, f.noise_score,
                        f.asymmetry_factor, f.detection_rate, f.detection_rate_gap_filled, f.reference_file,
                        f.charge_state, f.isotope_signals, f.ms2_reference_file,
                        f.ms2, f.matched_ms2, f.search_mode, f.annotation, f.formula, f.similarity,
                        f.matched_peak_number, f.smiles, f.inchikey] + quant)

    feature_table = pd.DataFrame(results, columns=columns)

    return feature_table


def output_feature_to_msp(feature_table, output_path):
    """
    A function to output MS2 spectra to MSP format.

    Parameters
    ----------
    feature_table : pandas.DataFrame
        A DataFrame containing MS2 spectra.
    output_path : str
        The path to the output MSP file.
    """

    # check the output path to make sure it is a .msp file and it exists
    if not output_path.lower().endswith(".msp"):
        raise ValueError("The output path must be a .msp file.")

    pattern = re.compile(r"\d+\.\d+")

    feature_ids = feature_table['feature_ID'].values
    mzs = feature_table['m/z'].values
    adducts = feature_table['adduct'].values
    rts = feature_table['RT'].values
    ms2s = feature_table['MS2'].values
    annotations = feature_table['annotation'].values
    search_modes = feature_table['search_mode'].values
    formulas = feature_table['formula'].values
    inchikeys = feature_table['InChIKey'].values
    smiles = feature_table['SMILES'].values

    buffer_size = min(1000, len(feature_table))
    buffer = []

    with open(output_path, "w") as f:
        for i in range(len(feature_table)):
            output_lines = [f"ID: {feature_ids[i]}"]

            is_ms2_missing = pd.isna(ms2s[i])
            if is_ms2_missing:
                output_lines.extend([
                    "NAME: Unknown",
                    f"PRECURSORMZ: {mzs[i]}",
                    f"PRECURSORTYPE: {adducts[i]}",
                    f"RETENTIONTIME: {rts[i]}",
                    "Num Peaks: 0",
                    ""
                ])
            else:
                name = "Unknown" if pd.isna(annotations[i]) else str(annotations[i])
                output_lines.extend([
                    f"NAME: {name}",
                    f"PRECURSORMZ: {mzs[i]}",
                    f"PRECURSORTYPE: {adducts[i]}",
                    f"RETENTIONTIME: {rts[i]}",
                    f"SEARCHMODE: {search_modes[i]}",
                    f"FORMULA: {formulas[i]}",
                    f"INCHIKEY: {inchikeys[i]}",
                    f"SMILES: {smiles[i]}"
                ])
                peaks = pattern.findall(ms2s[i])
                num_peaks = len(peaks) // 2
                output_lines.append(f"Num Peaks: {num_peaks}")
                output_lines.extend(
                    f"{peaks[2 * j]}\t{peaks[2 * j + 1]}"
                    for j in range(num_peaks)
                )
            output_lines.append("")
            buffer.append('\n'.join(output_lines))

            if len(buffer) >= buffer_size or i == len(feature_table) - 1:
                f.write('\n'.join(buffer) + '\n')
                buffer = []


def output_feature_table(feature_table, output_path):
    """
    Output the aligned feature table.

    Parameters
    ----------------------------------------------------------
    feature_table: DataFrame
        The aligned feature table.
    output_path: str
        The path to save the aligned feature table.
    """
    columns_to_round = {
        "m/z": 4,
        "RT": 3,
        'detection_rate': 2,
        'detection_rate_gap_filled': 2
    }

    # keep four digits for the m/z column and three digits for the RT column
    for col, decimals in columns_to_round.items():
        if col in feature_table.columns:
            feature_table[col] = feature_table[col].round(decimals)
    feature_table['similarity'] = pd.to_numeric(feature_table['similarity'], errors='coerce').round(4)

    feature_table.to_csv(output_path, index=False, sep="\t")


def retention_time_correction(mz_ref, rt_ref, mz_arr, rt_arr, mz_tol=0.01, rt_tol=0.5, mode='linear_interpolation',
                              rt_max=None):
    """
    To correct retention times for feature alignment.

    There are three steps:
    1. Find the selected anchors in the given data.
    2. Create a model to correct retention times.
    3. Correct retention times.
    
    Parameters
    ----------
    mz_ref: np.array
        The m/z values of the selected anchors from another reference file.
    rt_ref: np.array
        The retention times of the selected anchors from another reference file.
    mz_arr: np.array
        Feature m/z values in the current file.
    rt_arr: np.array
        Feature retention times in the current file.
    mz_tol: float
        The m/z tolerance for selecting anchors.
    rt_tol: float
        The retention time tolerance for selecting anchors.
    mode: str
        The mode for retention time correction. Not used now.
        'linear_interpolation': linear interpolation for retention time correction.
    rt_max: float
        End of the retention time range.
    return_model: bool
        Whether to return the model for retention time correction.
    
    Returns
    -------
    rt_arr: np.array
        The corrected retention times.
    f: interp1d
        The model for retention time correction.
    """
    if len(mz_ref) == 0 or len(mz_arr) == 0:
        return rt_arr, None

    rt_matched = []
    idx_matched = []

    for i in range(len(mz_ref)):
        mz_mask = np.abs(mz_arr - mz_ref[i]) < mz_tol
        rt_mask = np.abs(rt_arr - rt_ref[i]) < rt_tol
        v = np.where(mz_mask & rt_mask)[0]
        if len(v) == 1:
            rt_matched.append(rt_arr[v[0]])
            idx_matched.append(i)
    rt_ref = rt_ref[idx_matched]

    if len(idx_matched) < 5:
        return rt_arr, None

    # remove outliers
    v = rt_ref - np.array(rt_matched)
    k = np.abs(v - np.mean(v)) < np.std(v)
    rt_ref = rt_ref[k]
    rt_matched = np.array(rt_matched)[k]

    if len(rt_matched) < 5:
        return rt_arr, None

    x = [0]
    y = [0]
    for i in range(len(rt_matched)):
        if rt_matched[i] - x[-1] > 0.1:
            x.append(rt_matched[i])
            y.append(rt_ref[i])

    try:
        f = interp1d(x, y, fill_value='extrapolate')
        return f(rt_arr), f
    except ValueError:
        # 보간에 실패한 경우(예: 포인트 부족)
        return rt_arr, None


def rt_anchor_selection(data_path, num=50, noise_score_tol=0.1, mz_tol=0.01):
    """
    Retention time anchors have unique m/z values and low noise scores. From all candidate features, 
    the top *num* features with the highest peak heights are selected as anchors.

    Parameters
    ----------
    data_path : str
        Absolute directory to the feature tables.
    num : int
        The number of anchors to be selected.
    noise_tol : float
        The noise level for the anchors. Suggestions: 0.3 or lower.
    mz_tol : float
        The m/z tolerance for selecting anchors.

    Returns
    -------
    anchors: list
        A list of anchors (dict) for retention time correction.
    """

    try:
        df = pd.read_csv(data_path, sep="\t", usecols=["m/z", "RT", "noise_score", "peak_height"])
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading file {data_path}: {e}")
        return np.array([]), np.array([])

    if df.empty:
        return np.array([]), np.array([])

    # sort by m/z
    df = df.sort_values(by="m/z")
    mzs = df["m/z"].values
    rts = df["RT"].values
    noise_scores = df["noise_score"].values
    peak_heights = df["peak_height"].values

    diff = np.diff(mzs)

    indices = np.arange(1, len(mzs) - 1)

    prev_diff_mask = diff[indices - 1] > mz_tol
    next_diff_mask = diff[indices] > mz_tol
    noise_mask = noise_scores[indices] < noise_score_tol

    valid_mask = prev_diff_mask & next_diff_mask & noise_mask
    candidates = indices[valid_mask]

    if len(candidates) == 0:
        return np.array([]), np.array([])

    candidate_heights = peak_heights[candidates]

    num_select = min(num, len(candidates))

    top_indices = np.argsort(candidate_heights)[-num_select:][::-1]
    final_candidates = candidates[top_indices]

    valid_mzs = mzs[final_candidates]
    valid_rts = rts[final_candidates]

    return valid_mzs, valid_rts


"""
Internal Functions
------------------------------------------------------------------------------------------------------------------------
"""

def split_to_train_test(array, interval=0.1):
    """
    Split the selected anchors into training and testing sets.

    Parameters
    ----------
    array: numpy.ndarray
        The retention times of the selected anchors.
    interval: float
        The time interval for splitting the anchors.

    Returns
    -------
    train_idx: list
        The indices of the training set.
    test_idx: list
        The indices of the testing set.
    """

    train_idx = [0, len(array) - 1]
    for i in range(1, len(array)):
        if array[i] - array[train_idx[-1]] > interval:
            train_idx.append(i)
    train_idx.sort()
    test_idx = [i for i in range(len(array)) if i not in train_idx]

    return train_idx, test_idx


def _assign_value_to_feature(f, df, i, p, file_name):
    """
    Assign the values from individual files to the aligned feature.

    Parameters
    ----------
    f: AlignedFeature
        The aligned feature.
    df: DataFrame
        The feature table from the individual file.
    i: int
        The file index among all files to be aligned.
    p: int
        The row index of the feature in the current individual file.
    file_name: str
        The name of the current file.
    """

    f.feature_id_arr[i] = df.loc[p, "feature_ID"]
    f.mz_arr[i] = df.loc[p, "m/z"]
    f.rt_arr[i] = df.loc[p, "RT"]
    f.peak_height_arr[i] = df.loc[p, "peak_height"]
    f.peak_area_arr[i] = df.loc[p, "peak_area"]
    f.top_average_arr[i] = df.loc[p, "top_average"]
    f.length_arr[i] = df.loc[p, "total_scans"]
    f.gaussian_similarity_arr[i] = df.loc[p, "Gaussian_similarity"]
    f.noise_score_arr[i] = df.loc[p, "noise_score"]
    f.asymmetry_factor_arr[i] = df.loc[p, "asymmetry_factor"]
    f.scan_idx_arr[i] = df.loc[p, "scan_idx"]
    if pd.notna(df.loc[p, "MS2"]):
        f.ms2_seq.append([file_name, df.loc[p, "MS2"]])


def _assign_reference_values(f, df, i, p, file_name):
    """
    Assign the reference values to the aligned feature.

    Parameters
    ----------
    f: AlignedFeature
        The aligned feature.
    df: DataFrame
        The feature table from the individual file.
    i: int
        The file index among all files to be aligned.
    p: int
        The row index of the feature in the current individual file.
    file_name: str
        The name of the reference file
    """

    f.mz = df.loc[p, "m/z"]
    f.rt = df.loc[p, "RT"]
    f.reference_file = file_name
    f.reference_scan_idx = df.loc[p, "scan_idx"]
    f.highest_intensity = df.loc[p, "peak_height"]
    f.gaussian_similarity = df.loc[p, "Gaussian_similarity"]
    f.noise_score = df.loc[p, "noise_score"]
    f.asymmetry_factor = df.loc[p, "asymmetry_factor"]
