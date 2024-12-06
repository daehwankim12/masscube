# Author: Hauxu Yu

# A module to group features (unique m/z and reteniton time) originated from the 
# same compound. It will annotate isotopes, in-source fragments and adducts.

# Development plan: annotation of multimers are to be performed in the future.

# imports
import numpy as np
import os
from tqdm import tqdm

from .params import Params
from .raw_data_utils import read_raw_file_to_obj
from .utils_functions import extract_signals_from_string


def group_features_after_alignment(features, params: Params):
    """
    For untargeted metabolomics workflow only. Project directory is required to reload the raw data.
    Group features after alignment based on the reference file. This function requires
    to reload the raw data to examine the scan-to-scan correlation between features.

    Parameters
    ----------
    features: list
        A list of AlignedFeature objects.
    params: Params object
        A Params object that contains the parameters for feature grouping.
    """

    # sort features by maximum intensity from high to low
    features.sort(key=lambda x: x.highest_intensity, reverse=True)
    mz_arr = np.array([f.mz for f in features])
    rt_arr = np.array([f.rt for f in features])
    
    to_anno = np.ones(len(features), dtype=bool)
    feature_group_id = 1

    # find isotopes, in-source fragments, and adducts for each feature
    for i, f in enumerate(tqdm(features)):
        
        if not to_anno[i]:
            continue

        to_anno[i] = False
        f.feature_group_id = feature_group_id
        fn = os.path.join(params.tmp_file_dir, f.reference_file + ".mzpkl")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Reference file {fn} not found for feature grouping.")
        
        # load the reference file
        d = read_raw_file_to_obj(fn)
        _, eic_a, _ = d.get_eic_data(f.mz, f.rt, mz_tol=params.mz_tol_ms1, rt_tol=0.25)
        k = np.argmin(np.abs(d.ms1_time_arr-f.rt))

        f.isotope_signals = find_isotope_signals(mz=f.mz, signals=d.scans[k].signals, mz_tol=params.mz_tol_feature_grouping)
        if f.adduct_type is None:
            f.adduct_type = "[M+H]+" if params.ion_mode.lower() == "positive" else "[M-H]-"
        for s in f.isotope_signals:
            w1 = np.abs(mz_arr - s[0]) < params.mz_tol_feature_grouping
            w2 = np.abs(rt_arr - f.rt) < params.rt_tol_feature_grouping
            w = np.where(w1 & w2 & to_anno)[0]
            if len(w) > 0:
                for wi in w:
                    to_anno[wi] = False
                    features[wi].feature_group_id = f.feature_group_id
                    features[wi].is_isotope = True
                    features[wi].adduct_type = f.adduct_type + "_isotope"
        # prepare m/z list to search in-source fragments and adducts
        search_dict = generate_search_dict(f, f.adduct_type, params.ion_mode)
        # find the features that are in the same group
        for key in search_dict.keys():
            v1 = np.abs(mz_arr - search_dict[key]) < params.mz_tol_feature_grouping
            v2 = np.abs(rt_arr - f.rt) < params.rt_tol_feature_grouping
            v = np.where(v1 & v2 & to_anno)[0]
            if len(v) > 0:
                # find the one with the lowest RT difference
                vi = v[np.argmin(np.abs(rt_arr[v] - f.rt))]
                # check which has the largest scan-to-scan correlation
                # create local EICs for this two m/z values
                _, eic_b, _ = d.get_eic_data(mz_arr[vi], f.rt, mz_tol=params.mz_tol_ms1, rt_tol=0.25)
                if scan_to_scan_cor_intensity(eic_a[:, 1], eic_b[:, 1]) > params.scan_scan_cor_tol:
                    to_anno[vi] = False
                    features[vi].feature_group_id = f.feature_group_id
                    features[vi].adduct_type = key
                    if "ISF" in key:
                        features[vi].is_in_source_fragment = True
                        features[vi].adduct_type = "ISF"
                    # find isotopes for this ion and do not use scan-to-scan correlation
                    features[vi].isotope_signals = find_isotope_signals(features[vi].mz, d.scans[k].signals, mz_tol=params.mz_tol_feature_grouping)
                    # retrieve the isotope signals from the feature list
                    for s in features[vi].isotope_signals:
                        w1 = np.abs(mz_arr - s[0]) < params.mz_tol_feature_grouping
                        w2 = np.abs(rt_arr - f.rt) < params.rt_tol_feature_grouping
                        w = np.where(w1 & w2 & to_anno)[0]
                        if len(w) > 0:
                            for wi in w:
                                to_anno[wi] = False
                                features[wi].feature_group_id = f.feature_group_id
                                features[wi].is_isotope = True
                                features[wi].adduct_type = features[vi].adduct_type + "_isotope"
        feature_group_id += 1


def group_features_single_file(d):
    """
    Group features from a single file based on the m/z, retention time, MS2 and scan-to-scan correlation.

    Parameters
    ----------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    # sort features by maximum intensity from high to low
    d.features.sort(key=lambda x: x.peak_height, reverse=True)
    mz_arr = np.array([f.mz for f in d.features])
    rt_arr = np.array([f.rt for f in d.features])

    to_anno = np.ones(len(d.features), dtype=bool)
    feature_group_id = 1

    # find isotopes, in-source fragments, and adducts for each feature
    for i, f in enumerate(tqdm(d.features)):

        if not to_anno[i]:
            continue

        to_anno[i] = False
        f.feature_group_id = feature_group_id

        _, eic_a, _ = d.get_eic_data(f.mz, f.rt, mz_tol=d.params.mz_tol_ms1, rt_tol=0.25)
        k = np.argmin(np.abs(d.ms1_time_arr-f.rt))

        f.isotope_signals = find_isotope_signals(mz=f.mz, signals=d.scans[k].signals, mz_tol=d.params.mz_tol_feature_grouping)
        if f.adduct_type is None:
            f.adduct_type = "[M+H]+" if d.params.ion_mode.lower() == "positive" else "[M-H]-"
        for s in f.isotope_signals:
            w1 = np.abs(mz_arr - s[0]) < d.params.mz_tol_feature_grouping
            w2 = np.abs(rt_arr - f.rt) < d.params.rt_tol_feature_grouping
            w = np.where(w1 & w2 & to_anno)[0]
            if len(w) > 0:
                for wi in w:
                    to_anno[wi] = False
                    d.features[wi].feature_group_id = feature_group_id
                    d.features[wi].is_isotope = True
                    d.features[wi].adduct_type = f.adduct_type + "_isotope"
        # prepare m/z list to search in-source fragments and adducts
        search_dict = generate_search_dict(f, f.adduct_type, d.params.ion_mode)
        # find the features that are in the same group
        for key in search_dict.keys():
            v1 = np.abs(mz_arr - search_dict[key]) < d.params.mz_tol_feature_grouping
            v2 = np.abs(rt_arr - f.rt) < d.params.rt_tol_feature_grouping
            v = np.where(v1 & v2 & to_anno)[0]
            if len(v) > 0:
                # find the one with the lowest RT difference
                vi = v[np.argmin(np.abs(rt_arr[v] - f.rt))]
                # check which has the largest scan-to-scan correlation
                # create local EICs for this two m/z values
                _, eic_b, _ = d.get_eic_data(mz_arr[vi], f.rt, mz_tol=d.params.mz_tol_ms1, rt_tol=0.25)
                if scan_to_scan_cor_intensity(eic_a[:, 1], eic_b[:, 1]) > d.params.scan_scan_cor_tol:
                    to_anno[vi] = False
                    d.features[vi].feature_group_id = feature_group_id
                    d.features[vi].adduct_type = key
                    if "ISF" in key:
                        d.features[vi].is_in_source_fragment = True
                        d.features[vi].adduct_type = "ISF"
                    # find isotopes for this ion and do not use scan-to-scan correlation
                    d.features[vi].isotope_signals = find_isotope_signals(d.features[vi].mz, d.scans[k].signals, mz_tol=d.params.mz_tol_feature_grouping)
                    # retrieve the isotope signals from the feature list
                    for s in d.features[vi].isotope_signals:
                        w1 = np.abs(mz_arr - s[0]) < d.params.mz_tol_feature_grouping
                        w2 = np.abs(rt_arr - f.rt) < d.params.rt_tol_feature_grouping
                        w = np.where(w1 & w2 & to_anno)[0]
                        if len(w) > 0:
                            for wi in w:
                                to_anno[wi] = False
                                d.features[wi].feature_group_id = feature_group_id
                                d.features[wi].is_isotope = True
                                d.features[wi].adduct_type = d.features[vi].adduct_type + "_isotope"
        feature_group_id += 1


def generate_search_dict(feature, adduct_form, ion_mode):
    """
    Generate a search dictionary for the feature grouping.

    Parameters
    ----------
    feature: Feature object
        The feature object to be grouped.
    adduct_form: str
        The adduct form of the feature.
    ion_mode: str
        The ionization mode. "positive" or "negative".

    Returns
    -------
    dict
        A dictionary that contains the possible adducts and in-source fragments.
    """

    search_dict = {}
    if ion_mode.lower() == "positive":
        dic = ADDUCT_POS
    elif ion_mode.lower() == "negative":
        dic = ADDUCT_NEG
    
    if adduct_form not in dic.keys():
        return search_dict

    base_mz = feature.mz - dic[adduct_form][0]

    # possible adducts
    for key in dic.keys():
        if key != adduct_form:
            search_dict[key] = (base_mz*dic[key][1] + dic[key][0]) / dic[key][2]
    
    # possible in-source fragments
    if feature.ms2 is not None:
        if type(feature.ms2) == str:
            ms2 = extract_signals_from_string(feature.ms2)
        else:
            ms2 = feature.ms2.signals
        for i, p in enumerate(ms2):
            search_dict[f'ISF_{i+1}'] = p[0]
    
    return search_dict


def find_isotope_signals(mz, signals, mz_tol=0.015, charge_state=1, num=5):
    """
    Find isotope patterns from the MS1 signals.

    Parameters
    ----------
    mz: float
        The m/z value of the feature.
    signals: np.array
        The MS1 signals as [[m/z, intensity], ...]
    mz_tol: float
        The m/z tolerance to find isotopes.
    charge_state: int
        The charge state of the feature.
    num: int
        The maximum number of isotopes to be found.

    Returns
    -------
    numpy.array
        The m/z and intensity of the isotopes.
    """
    
    diff = signals[:, 0] - mz
    isotope_signals = []
    if charge_state == 1:
        for i in range(1, num+1):
            v = np.abs(diff - 1.003355*i) < mz_tol
            if np.sum(v) > 0:
                tmp = signals[v]
                tmp = [np.mean(tmp[:, 0]), np.sum(tmp[:, 1])]
                isotope_signals.append(tmp)
    
    return np.array(isotope_signals)


"""
Helper functions and constants
==============================
"""

def scan_to_scan_cor_intensity(a, b):
    """
    Calculate the scan-to-scan correlation between two features using Pearson correlation.

    Parameters
    ----------
    a: np.array
        Intensity array of the first m/z
    b: np.array
        Intensity array of the second m/z
    
    Returns
    -------
    float
        The scan-to-scan correlation between the two features.
    """

    v = (a>0) & (b>0)
    if np.sum(v) < 5:
        return 1.0

    return np.corrcoef(a[v], b[v])[0, 1]


def scan_to_scan_correlation(feature_a, feature_b):
    """
    Calculate the scan-to-scan correlation between two features using Pearson correlation.

    Parameters
    ----------
    feature_a: Feature object
        The first feature object.
    feature_b: Feature object
        The second feature object.
    
    Returns
    -------
    float
        The scan-to-scan correlation between the two features.
    """

    # find the common scans in the two rois
    common_idx_a = np.nonzero(np.in1d(feature_a.scan_idx_seq, feature_b.scan_idx_seq))[0]
    common_idx_b = np.nonzero(np.in1d(feature_b.scan_idx_seq, feature_a.scan_idx_seq))[0]

    # if the number of common scans is less than 5, return 1
    if len(common_idx_a) < 5:
        return 1.0

    # find the intensities of the common scans in the two rois
    int1 = feature_a.signals[common_idx_a,1]
    int2 = feature_b.signals[common_idx_b,1]

    # if all values are same in either feature, return 1
    # this is to avoid the case where the common scans are all zeros
    if np.all(int1 == int1[0]) or np.all(int2 == int2[0]):
        return 1.0
    
    return np.corrcoef(int1, int2)[0, 1]


def get_charge_state(mz_seq, valid_charge_states=[1,2]):
    """
    A function to determine the charge state using the m/z sequence of isotopes.

    Parameters
    ----------
    mz_seq: list
        A list of m/z values of isotopes.
    valid_charge_states: list
        A list of valid charge states.

    Returns
    -------
    int
        The charge state of the isotopes.
    """

    # if there is only one isotope, return 1
    if len(mz_seq) < 2:
        return 1
    else:
        for charge in valid_charge_states:
            if abs(mz_seq[1] - mz_seq[0] - 1.003355/charge) < 0.01:
                return charge
    return 1


ADDUCT_POS = {
    '[M+H]+': (1.007276, 1, 1),
    '[M+NH4]+': (18.033826, 1, 1),
    '[M]+': (0, 1, 1),
    '[M+H+CH3OH]+': (33.03349, 1, 1),
    '[M+Na]+': (22.989221, 1, 1),
    '[M+K]+': (38.963158, 1, 1),
    # '[M+Li]+': (6.014574, 1, 1),
    # '[M+Ag]+': (106.904548, 1, 1),
    '[M+H+CH3CN]+': (42.033826, 1, 1),
    '[M-H+2Na]+': (44.971165, 1, 1),
    '[M+H-H2O]+': (-17.003288, 1, 1),
    '[M+H-2H2O]+': (-35.01385291583, 1, 1),
    '[M+H-3H2O]+': (-53.02441759986, 1, 1),
    '[M+H+HAc]+': (61.02841, 1, 1),
    '[M+H+HFA]+': (47.01276, 1, 1),
    # '[M+Ca]2+': (39.961493, 1, 2),
    # '[M+Fe]2+': (55.93384, 1, 2),
    '[2M+H]+': (1.007276, 2, 1),
    '[2M+NH4]+': (18.033826, 2, 1),
    '[2M+Na]+': (22.989221, 2, 1),
    '[2M+H-H2O]+': (-17.003288, 2, 1),
    '[3M+H]+': (1.007276, 3, 1),
    '[3M+NH4]+': (18.033826, 3, 1),
    '[3M+Na]+': (22.989221, 3, 1),
    '[3M+H-H2O]+': (-17.003288, 3, 1),
    '[M+2H]2+': (0.503638, 1, 2),
    '[M+3H]3+': (0.335759, 1, 3),
}

ADDUCT_NEG = {
    '[M-H]-': (-1.007276, 1, 1),
    '[M+Cl]-': (34.969401, 1, 1),
    # '[M+Br]-': (78.918886, 1, 1),
    '[M+FA]-': (44.998203, 1, 1),
    '[M+Ac]-': (59.013853, 1, 1),
    '[M-H-H2O]-': (-19.017841, 1, 1),
    '[M-H+Cl]2-': (33.962124, 1, 2),
    '[2M-H]-': (-1.007276, 2, 1),
    '[2M+Cl]-': (34.969401, 2, 1),
    # '[2M+Br]-': (78.918886, 2, 1),
    '[2M+FA]-': (44.998203, 2, 1),
    '[2M+Ac]-': (59.013853, 2, 1),
    '[2M-H-H2O]-': (-19.017841, 2, 1),
    '[3M-H]-': (-1.007276, 3, 1),
    '[3M+Cl]-': (34.969401, 3, 1),
    # '[3M+Br]-': (78.918886, 3, 1),
    '[3M+FA]-': (44.998203, 3, 1),
    '[3M+Ac]-': (59.013853, 3, 1),
    '[3M-H-H2O]-': (-19.017841, 3, 1),
    '[M-2H]2-': (-0.503638, 1, 2),
    '[M-3H]3-': (-0.335759, 1, 3),
}


'''
Old functions, not used in the current version

def annotate_isotope(d, mz_tol=0.015, rt_tol=0.1, valid_intensity_ratio_range=[0.001, 1.2], charge_state_range=[1,2]):
    """
    A function to annotate isotopes in the MS data.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object.
    mz_tol: float
        The m/z tolerance to find isotopes.
    rt_tol: float
        The retention time tolerance to find isotopes.
    valid_intensity_ratio_range: list
        The valid intensity ratio range between isotopes.
    charge_state_range: list (not used)
        The charge state range of the isotopes. [lower, upper]
    """

    # rank the rois (d.rois) in each file by m/z
    d.rois.sort(key=lambda x: x.mz)

    for r in d.rois:

        if r.is_isotope:
            continue

        r.isotope_int_seq = [r.peak_height]
        r.isotope_mz_seq = [r.mz]

        last_mz = r.mz

        # check if the currect ion is double charged
        r.charge_state = 1
        target_mz = r.mz + 1.003355/2
        v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]
        if len(v) > 0:
            # an isotope can't have intensity 1.2 fold or higher than M0 or 1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < valid_intensity_ratio_range[1]*r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > 0.01*r.peak_height]
            if len(v) > 0:
                r.charge_state = 2

        isotope_id_seq = []
        target_mz = r.mz + 1.003355/r.charge_state
        total_int = r.peak_height

        # find roi using isotope list
        i = 0
        while i < 5:   # maximum 5 isotopes
            # if isotpoe is not found in two daltons, stop searching
            if target_mz - last_mz > 2.2:
                break

            v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]

            if len(v) == 0:
                i += 1
                continue

            # an isotope can't have intensity 1.2 fold or higher than M0 or 0.1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < valid_intensity_ratio_range[1]*r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > valid_intensity_ratio_range[0]*total_int]

            if len(v) == 0:
                i += 1
                continue
            
            # for high-resolution data, C and N isotopes can be separated and need to be summed
            total_int = np.sum([d.rois[vi].peak_height for vi in v])
            last_mz = np.mean([d.rois[vi].mz for vi in v])

            r.isotope_mz_seq.append(last_mz)
            r.isotope_int_seq.append(total_int)
            
            for vi in v:
                d.rois[vi].is_isotope = True
                isotope_id_seq.append(d.rois[vi].id)

            target_mz = last_mz + 1.003355/r.charge_state
            i += 1

        r.charge_state = get_charge_state(r.isotope_mz_seq)
        r.isotope_id_seq = isotope_id_seq


def annotate_in_source_fragment(d, mz_tol=0.01, rt_tol=0.05):
    """
    Function to annotate in-source fragments in the MS data.
    Only [M+O] (roi.is_isotope==True) will be considered in this function.
    Two criteria are used to annotate in-source fragments:
    1. The precursor m/z of the child is in the MS2 spectrum of the parent.
    2. Peak-peak correlation > params.ppr.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    # sort ROI by m/z from high to low
    d.rois.sort(key=lambda x: x.mz)

    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes or rois with length < 4 can't be parent or child
    for idx, r in enumerate(d.rois):
        if r.is_isotope or len(r.scan_idx_seq) < 4:
            roi_to_label[idx] = False
    
    # find in-source fragments
    for idx in range(len(d.rois)-1, -1, -1):

        r = d.rois[idx]

        # isotpes and in-source fragments cannot be the parent
        if not roi_to_label[idx] or r.best_ms2 is None:
            continue

        for m in r.best_ms2.peaks[:, 0]:

            v = np.logical_and(np.abs(d.roi_mz_seq - m) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            # an in-source fragment can't have intensity 3 fold or higher than the parent
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < 3*r.peak_height]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if scan_to_scan_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].is_in_source_fragment = True
                d.rois[v].isf_parent_roi_id = r.id
                r.isf_child_roi_id.append(d.rois[v].id)


def annotate_adduct(d, mz_tol=0.01, rt_tol=0.05):
    """
    A function to annotate adducts from the same compound.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    d.rois.sort(key=lambda x: x.mz)
    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes, in-source fragments, and rois with length < 4 can't be parent or child
    for idx, r in enumerate(d.rois):
        if r.is_isotope or r.is_in_source_fragment or len(r.scan_idx_seq) < 4:
            roi_to_label[idx] = False

    if d.params.ion_mode.lower() == "positive":
        default_adduct = "[M+H]+"
        adduct_mass_diffence = _ADDUCT_MASS_DIFFERENCE_POS_AGAINST_H
        adduct_mass_diffence['[2M+H]+'] = r.mz - 1.007276
        adduct_mass_diffence['[3M+H]+'] = 2*(r.mz - 1.007276)

    elif d.params.ion_mode.lower() == "negative":
        default_adduct = "[M-H]-"
        adduct_mass_diffence = _ADDUCT_MASS_DIFFERENCE_NEG_AGAINST_H
        adduct_mass_diffence['[2M-H]-'] = r.mz + 1.007276
        adduct_mass_diffence['[3M-H]-'] = 2*(r.mz + 1.007276)


    # find adducts by assuming the current roi is the [M+H]+ ion in positive mode and [M-H]- ion in negative mode
    for idx, r in enumerate(d.rois):
        
        if not roi_to_label[idx]:
            continue

        if r.charge_state == 2:
            if d.params.ion_mode.lower() == "positive":
                r.adduct_type = "[M+2H]2+"
                roi_to_label[idx] = False
            elif d.params.ion_mode.lower() == "negative":
                r.adduct_type = "[M-2H]2-"
                roi_to_label[idx] = False
            continue
        
        for adduct in adduct_mass_diffence.keys():
            m = r.mz + adduct_mass_diffence[adduct]
            v = np.logical_and(np.abs(d.roi_mz_seq - m) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if scan_to_scan_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].adduct_type = adduct
                d.rois[v].adduct_parent_roi_id = r.id
                r.adduct_child_roi_id.append(d.rois[v].id)

        if len(r.adduct_child_roi_id) > 0:
            r.adduct_type = default_adduct
        
    for r in d.rois:
        if r.adduct_type is None:
            r.adduct_type = default_adduct

# _adduct_pos_primary = {
#     '[M+H]+': [-21.981945, -17.02655, 18.010564],
#     '[M+Na]+': [21.981945, 4.955395, 39.992509],
#     '[M+NH4]+': [17.02655, -4.955395, 35.037114],
#     '[M+H-H2O]+': [-18.010564, -39.992509, -35.037114]
# }

# _adduct_neg_primary = {
#     '[M-H]-': [18.010565, -46.005479, -60.021129],
#     '[M-H-H2O]-': [-18.010565, -64.016044, -78.031694],
#     '[M+FA]-': [46.005479, 64.016044, -14.01565],
#     '[M+Ac]-': [60.021129, 78.031694, 14.01565]
# }

# could include more adducts
_ADDUCT_MASS_DIFFERENCE_POS_AGAINST_H = {
    '[M+H-H2O]+': -18.010565,
    '[M+Na]+': 21.981945,
    '[M+K]+': 37.955882,
    '[M+NH4]+': 17.02655,
}

_ADDUCT_MASS_DIFFERENCE_NEG_AGAINST_H = {
    '[M-H-H2O]-': -18.010564,
    '[M+Cl]-': 35.976677,
    '[M+CH3COO]-': 60.021129,
    '[M+HCOO]-': 46.005479,
}

_adduct_pos_primary = {
    '[M+H]+': [-17.02655, 18.010564],
    '[M+NH4]+': [17.02655, 35.037114],
    '[M+H-H2O]+': [-18.010564, -35.037114]
}

def find_adduct_form(mz, signals, mode="positive", mz_tol=0.01):
    """
    Find the most likely adduct form of a feature based on its m/z and MS1 signals.
    By default, the postive mode adducts considered are [M+H]+, [M+Na]+, [M+NH4]+, and [M+H-H2O]+, 
    and the negative mode adducts considered are [M-H]-, [M-H-H2O]-, [M+FA]-, and [M+Ac]-.

    Parameters
    ----------
    mz: float
        The m/z value of the feature.
    signals: np.array
        The MS1 signals as [[m/z, intensity], ...]
    mode: str
        The ionization mode. "positive" or "negative".
    mz_tol: float
        The m/z tolerance to find the adduct.
    
    Returns
    -------
    str
        The most likely adduct.
    """

    if mode.lower() == "positive":
        return "[M+H]+"
    elif mode.lower() == "negative":
        return "[M-H]-"
    else:
        return None

    # diff = np.abs(mz - signals[:, 0])
    # scores = np.zeros(len(adduct_dict.keys()))
    # for i, values in enumerate(adduct_dict.values()):
    #     for v in values:
    #         scores[i] += np.sign(np.sum(np.abs(diff - v) < mz_tol)) * signals[:,1][np.argmin(np.abs(diff - v))]
    
    # return list(adduct_dict.keys())[np.argmax(scores)]
'''
