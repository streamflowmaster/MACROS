import torch
import numpy as np
from typing import List, Dict, Union
from collections.abc import Sequence

hnmr_category = {
    'dd': 0, 'm': 1, 's': 2, 't': 3, 'ddd': 4, 'd': 5, 'pd': 6, 'tt': 7, 'dtdq': 8,
    'dt': 9, 'hd': 10, 'h': 11, 'q': 12, 'dq': 13, 'dtd': 14, 'dp': 15, 'ddq': 16, 'td': 17,
    'dddd': 18, 'ddt': 19, 'p': 20, 'dqdd': 21, 'hept': 22, 'qdd': 23, 'dddt': 24,
    'dtdd': 25, 'ddddd': 26, 'dtq': 27, 'dtt': 28, 'dtddd': 29, 'qd': 30, 'dqd': 31,
    'ddtd': 32, 'dhept': 33, 'tq': 34, 'ddp': 35, 'qt': 36, 'ttd': 37, 'tdd': 38,
    'tdt': 39, 'tddd': 40, 'dh': 41, 'qddd': 42, 'pt': 43, 'dqt': 44, 'dddq': 45,
    'ddtt': 46, 'heptd': 47, 'dddp': 48, 'ddddtd': 49, 'dttd': 50, 'tp': 51, 'tdq': 52,
    'qdt': 53, 'qq': 54, 'pdd': 55, 'dddqd': 56, 'ttt': 57, 'ttq': 58, 'dtdt': 59,
    'th': 60, 'ddddq': 61, 'tddt': 62, 'ddddt': 63, 'ddtq': 64, 'tqd': 65,
    'dtdtd': 66, 'ddtdd': 67, 'tddq': 68, 'dpdd': 69, 'ttdt': 70, 'ddh': 71,
    'tdp': 72
}

class NMRSpectrumTokenizer:
    CONFIG = {
        'vacum_token_idx': 2,
        'h_nmr_max_num_peak': 20,  # 22 - 2 for BOS and EOS
        'c_nmr_max_num_peak': 64,  # 66 - 2 for BOS and EOS
        'hsqc_nmr_max_num_peak': 64,  # 66 - 2 for BOS and EOS
        'h_nmr_jvalue_min': 0,
        'h_nmr_jvalue_max': 50,
        'j_value_disc': 100,
        'h_nmr_centroid_min': -2,
        'h_nmr_centroid_max': 10,
        'centroid_disc': 120,
        'max_nH': 100,
        'c_nmr_delta_min': -20,
        'c_nmr_delta_max': 250,
        'c_nmr_delta_disc': 1024,
        'c_nmr_intensity_min': 0,
        'c_nmr_intensity_max': 1,
        'c_nmr_intensity_disc': 100,
        'hsqc_nmr_intensity_min': -3,
        'hsqc_nmr_intensity_max': 400,
        'hsqc_nmr_intensity_disc': 500,
        'nmr_special_token_num': 3,
        'nmr_bos_token': 0,
        'nmr_eos_token': 1,
        'nmr_pad_token': 2,
        'nmr_vaccume_token_idx': 2,  # Use PAD token for vaccume
    }

    def __init__(self, NMR_category: str = 'cnmr', missing_identification: str = '<missing>'):
        self.CNMR_PADDING_TOKEN_ID = self.CONFIG['nmr_pad_token']
        self.HNMR_PADDING_TOKEN_ID = self.CONFIG['nmr_pad_token']
        self.HSQC_PADDING_TOKEN_ID = self.CONFIG['nmr_pad_token']

        self.CNMR_EOS_TOKEN_ID = self.CONFIG['nmr_eos_token']
        self.HNMR_EOS_TOKEN_ID = self.CONFIG['nmr_eos_token']
        self.HSQC_EOS_TOKEN_ID = self.CONFIG['nmr_eos_token']

        self.HNMR_BOS_TOKEN_ID = self.CONFIG['nmr_bos_token']
        self.CNMR_BOS_TOKEN_ID = self.CONFIG['nmr_bos_token']
        self.HSQC_BOS_TOKEN_ID = self.CONFIG['nmr_bos_token']

        self.special_token_num = self.CONFIG['nmr_special_token_num']
        self.hnmr_category_dict = hnmr_category
        self.hnmr_category_id_to_str = {v: k for k, v in self.hnmr_category_dict.items()}
        self.NMR_category = NMR_category.lower()
        self.missing_identification = missing_identification

        if self.NMR_category == 'hnmr':
            self.eos_token_id = self.HNMR_EOS_TOKEN_ID
            self.pad_token_id = self.HNMR_PADDING_TOKEN_ID
            self.bos_token_id = self.HNMR_BOS_TOKEN_ID
            self.max_num_peak = self.CONFIG['h_nmr_max_num_peak']
            self.fixed_length = self.max_num_peak + 2  # 22
        elif self.NMR_category == 'cnmr':
            self.eos_token_id = self.CNMR_EOS_TOKEN_ID
            self.pad_token_id = self.CNMR_PADDING_TOKEN_ID
            self.bos_token_id = self.CNMR_BOS_TOKEN_ID
            self.max_num_peak = self.CONFIG['c_nmr_max_num_peak']
            self.fixed_length = self.max_num_peak + 2  # 66
        elif self.NMR_category == 'hsqc':
            self.eos_token_id = self.HSQC_EOS_TOKEN_ID
            self.pad_token_id = self.HSQC_PADDING_TOKEN_ID
            self.bos_token_id = self.HSQC_BOS_TOKEN_ID
            self.max_num_peak = self.CONFIG['hsqc_nmr_max_num_peak']
            self.fixed_length = self.max_num_peak + 2  # 66
        else:
            raise ValueError(f'Unexpected NMR_category: {self.NMR_category}')

    def j_value_discrete(self, j_value):
        j_value = (j_value - self.CONFIG['h_nmr_jvalue_min']) / (
                self.CONFIG['h_nmr_jvalue_max'] - self.CONFIG['h_nmr_jvalue_min']) * (
                          self.CONFIG['j_value_disc'] - 1)
        return int(j_value) + self.special_token_num

    def centroid_discrete(self, centroid):
        try:
            centroid = (centroid - self.CONFIG['h_nmr_centroid_min']) / (
                    self.CONFIG['h_nmr_centroid_max'] - self.CONFIG['h_nmr_centroid_min']) * self.CONFIG[
                           'centroid_disc']
            return int(centroid) + self.special_token_num
        except:
            return 2

    def nH_discrete(self, nH):
        return nH + self.special_token_num

    def intensity_discrete(self, intensity):
        intensity = (intensity - self.CONFIG['c_nmr_intensity_min']) / (
                self.CONFIG['c_nmr_intensity_max'] - self.CONFIG['c_nmr_intensity_min']) * self.CONFIG[
                        'c_nmr_intensity_disc']
        return int(intensity) + self.special_token_num

    def delta_discrete(self, delta):
        delta = (delta - self.CONFIG['c_nmr_delta_min']) / (
                self.CONFIG['c_nmr_delta_max'] - self.CONFIG['c_nmr_delta_min']) * self.CONFIG['c_nmr_delta_disc']
        return int(delta) + self.special_token_num

    def hsqc_intensity_discrete(self, intensity):
        intensity = (intensity - self.CONFIG['hsqc_nmr_intensity_min']) / (
                self.CONFIG['hsqc_nmr_intensity_max'] - self.CONFIG['hsqc_nmr_intensity_min']) * self.CONFIG[
                        'hsqc_nmr_intensity_disc']
        return int(intensity) + self.special_token_num

    def centroid_discrete_reverse(self, centroid):
        centroid -= self.special_token_num
        return centroid / self.CONFIG['centroid_disc'] * (
                self.CONFIG['h_nmr_centroid_max'] - self.CONFIG['h_nmr_centroid_min']) + self.CONFIG[
            'h_nmr_centroid_min']

    def nH_discrete_reverse(self, nH):
        return nH - self.special_token_num

    def j_value_discrete_reverse(self, j_value):
        j_value -= self.special_token_num
        return (j_value) / (self.CONFIG['j_value_disc'] - 1) * (
                self.CONFIG['h_nmr_jvalue_max'] - self.CONFIG['h_nmr_jvalue_min']) + self.CONFIG['h_nmr_jvalue_min']

    def intensity_discrete_reverse(self, intensity):
        intensity -= self.special_token_num
        return intensity / self.CONFIG['c_nmr_intensity_disc'] * (
                self.CONFIG['c_nmr_intensity_max'] - self.CONFIG['c_nmr_intensity_min']) + self.CONFIG[
            'c_nmr_intensity_min']

    def hsqc_intensity_discrete_reverse(self, intensity):
        intensity -= self.special_token_num
        return intensity / self.CONFIG['hsqc_nmr_intensity_disc'] * (
                self.CONFIG['hsqc_nmr_intensity_max'] - self.CONFIG['hsqc_nmr_intensity_min']) + self.CONFIG[
            'hsqc_nmr_intensity_min']

    def delta_discrete_reverse(self, delta):
        delta -= self.special_token_num
        return delta / self.CONFIG['c_nmr_delta_disc'] * (
                self.CONFIG['c_nmr_delta_max'] - self.CONFIG['c_nmr_delta_min']) + self.CONFIG['c_nmr_delta_min']

    def cnmr_decode(self, spectra_dict: List[torch.Tensor]) -> List[List[Dict]]:
        """
        Decode tokenized ¹³C NMR spectra into human-readable format.

        Args:
            spectra_dict: List of 2 tensors [delta, intensity].
                          Each tensor has shape (batch_size, 66).

        Returns:
            List of lists, where each inner list contains dictionaries with keys:
            'delta (ppm)', 'intensity' (relative, optional).
        """
        batchsize = spectra_dict[0].shape[0]
        de_tokenized_spectra = []
        for batch_id in range(batchsize):
            de_tokenized_spectrum = []
            delta = spectra_dict[0][batch_id]

            # Find end of sequence using EOS
            eos_pos = (delta == self.eos_token_id).nonzero()
            eos_pos = self.fixed_length if len(eos_pos) == 0 else eos_pos[0].item()

            # Extract relevant data (skip BOS, stop before EOS)
            delta = delta[1:eos_pos]
            delta_value = self.delta_discrete_reverse(delta)
            intensity = spectra_dict[1][batch_id][1:eos_pos]
            intensity_value = self.intensity_discrete_reverse(intensity)

            # Create spectrum entries
            for i in range(len(delta)):
                entry = {'delta (ppm)': delta_value[i].cpu().item()}
                # Only include intensity if it's not a padding or special token
                if intensity[i] != self.pad_token_id and intensity[i] != self.bos_token_id and intensity[
                    i] != self.eos_token_id:
                    entry['intensity'] = intensity_value[i].cpu().item()
                de_tokenized_spectrum.append(entry)
            de_tokenized_spectra.append(de_tokenized_spectrum)
        return de_tokenized_spectra

    def hnmr_decode(self, spectra_dict: List[torch.Tensor]) -> List[List[Dict]]:
        batchsize = spectra_dict[0].shape[0]
        de_tokenized_spectra = []
        for batch_id in range(batchsize):
            de_tokenized_spectrum = []
            centroids = spectra_dict[0][batch_id]

            # Find end of sequence using EOS
            eos_pos = (centroids == self.eos_token_id).nonzero()
            eos_pos = self.fixed_length if len(eos_pos) == 0 else eos_pos[0].item()

            # Extract relevant data (skip BOS, stop before EOS)
            centroids = centroids[1:eos_pos]
            nH = spectra_dict[1][batch_id][1:eos_pos]
            category = spectra_dict[2][batch_id][1:eos_pos]
            jvalue = spectra_dict[3][batch_id][1:eos_pos]

            # Reverse discretization
            centroids_value = self.centroid_discrete_reverse(centroids)
            jvalue_value = self.j_value_discrete_reverse(jvalue)
            nH_value = self.nH_discrete_reverse(nH)
            category_ids = category - self.special_token_num
            category_strs = [self.hnmr_category_id_to_str.get(cid.item(), 'Others') for cid in category_ids]

            # Create spectrum entries
            for i in range(len(centroids)):
                entry = {
                    'centroid': centroids_value[i].cpu().item(),
                    'nH': nH_value[i].cpu().item()
                }
                # Only include category if it's not a padding or special token
                if category[i] != self.pad_token_id and category[i] != self.bos_token_id and category[
                    i] != self.eos_token_id:
                    entry['category'] = category_strs[i]
                # Only include jvalue if it's not a padding or special token
                if jvalue[i] != self.pad_token_id and jvalue[i] != self.bos_token_id and jvalue[i] != self.eos_token_id:
                    entry['jvalue'] = jvalue_value[i].cpu().item()
                de_tokenized_spectrum.append(entry)
            de_tokenized_spectra.append(de_tokenized_spectrum)
        return de_tokenized_spectra

    def hsqc_decode(self, spectra_dict: List[torch.Tensor]) -> List[List[Dict]]:
        batchsize = spectra_dict[0].shape[0]
        de_tokenized_spectra = []
        for batch_id in range(batchsize):
            de_tokenized_spectrum = []
            c13_centroid = spectra_dict[0][batch_id]

            # Find end of sequence using EOS
            eos_pos = (c13_centroid == self.eos_token_id).nonzero()
            eos_pos = self.fixed_length if len(eos_pos) == 0 else eos_pos[0].item()

            # Extract relevant data (skip BOS, stop before EOS)
            c13_centroid = c13_centroid[1:eos_pos]
            h1_centroid = spectra_dict[1][batch_id][1:eos_pos]
            nH = spectra_dict[2][batch_id][1:eos_pos]

            # Reverse discretization
            c13_value = self.delta_discrete_reverse(c13_centroid)
            h1_value = self.centroid_discrete_reverse(h1_centroid)
            nH_value = self.nH_discrete_reverse(nH)

            # Create spectrum entries
            for i in range(len(c13_centroid)):
                de_tokenized_spectrum.append({
                    '13C_centroid': c13_value[i].cpu().item(),
                    '1H_centroid': h1_value[i].cpu().item(),
                    'nH': nH_value[i].cpu().item()
                })
            de_tokenized_spectra.append(de_tokenized_spectrum)
        return de_tokenized_spectra

    def decode(self, spectra_dict: List[torch.Tensor]) -> List[List[Dict]]:
        if self.NMR_category == 'hnmr':
            return self.hnmr_decode(spectra_dict)
        elif self.NMR_category == 'cnmr':
            return self.cnmr_decode(spectra_dict)
        elif self.NMR_category == 'hsqc':
            return self.hsqc_decode(spectra_dict)
        else:
            raise ValueError(f'Decoding not supported for NMR_category: {self.NMR_category}')

    def batch_decode(self, spectra_dict: List[torch.Tensor]) -> List[List[Dict]]:
        return self.decode(spectra_dict)

    def validate_input(self, spectra: Dict[str, List[Union[Dict, str]]]) -> None:
        """Validate input format and content for the original encode method."""
        if not isinstance(spectra, dict):
            raise TypeError("Input must be a dictionary with modality keys")
        valid_modalities = {'[cnmr]', '[hnmr]', '[hsqc]', '[ir]'}
        for modality in spectra:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality: {modality}. Expected one of {valid_modalities}")
            if not isinstance(spectra[modality], list):
                raise TypeError(f"Value for modality {modality} must be a list")
            for item in spectra[modality]:
                if item != self.missing_identification:
                    if not isinstance(item, dict):
                        raise TypeError(f"Non-missing item in {modality} must be a dictionary")
                    if modality == '[ir]':
                        if 'ir' not in item:
                            raise ValueError(f"Dictionary in {modality} must contain 'ir' key")
                    else:
                        peak_key = {'[cnmr]': 'c_nmr_peaks', '[hnmr]': 'h_nmr_peaks', '[hsqc]': 'hsqc_nmr_peaks'}[
                            modality]
                        if peak_key not in item:
                            raise ValueError(f"Dictionary in {modality} must contain '{peak_key}' key")
                        if not isinstance(item[peak_key], list):
                            raise TypeError(f"'{peak_key}' in {modality} must be a list of peak dictionaries")
                        for peak in item[peak_key]:
                            if not isinstance(peak, dict):
                                raise TypeError(f"Peak in {peak_key} must be a dictionary")
                            if modality == '[cnmr]':
                                if 'delta (ppm)' not in peak:
                                    raise ValueError(f"Peak in {peak_key} must contain 'delta (ppm)' key")
                            elif modality == '[hnmr]':
                                if 'centroid' not in peak or 'nH' not in peak:
                                    raise ValueError(f"Peak in {peak_key} must contain 'centroid' and 'nH' keys")
                            elif modality == '[hsqc]':
                                if '13C_centroid' not in peak or '1H_centroid' not in peak or 'nH' not in peak:
                                    raise ValueError(
                                        f"Peak in {peak_key} must contain '13C_centroid', '1H_centroid', and 'nH' keys")

    def validate_dict_list_input(self, spectra_dict: List[Dict[str, List[Dict]]]) -> None:
        """Validate input format for encode_from_dict_list."""
        if not isinstance(spectra_dict, list):
            raise TypeError("Input must be a list of dictionaries")
        valid_modalities = {'c_nmr_peaks', 'h_nmr_peaks', 'hsqc_nmr_peaks', 'ir'}
        for item in spectra_dict:
            if not isinstance(item, dict):
                raise TypeError("Each item in spectra_dict must be a dictionary")
            for key in item:
                if key not in valid_modalities:
                    raise ValueError(f"Invalid modality key: {key}. Expected one of {valid_modalities}")
                if key == 'ir':
                    if not isinstance(item[key], (np.ndarray, torch.Tensor)):
                        raise TypeError(f"'ir' must be a numpy array or torch tensor")
                else:
                    if not isinstance(item[key], list):
                        raise TypeError(f"'{key}' must be a list of peak dictionaries")
                    for peak in item[key]:
                        if not isinstance(peak, dict):
                            raise TypeError(f"Peak in {key} must be a dictionary")
                        if key == 'c_nmr_peaks':
                            if 'delta (ppm)' not in peak:
                                raise ValueError(f"Peak in {key} must contain 'delta (ppm)' key")
                        elif key == 'h_nmr_peaks':
                            if 'centroid' not in peak or 'nH' not in peak:
                                raise ValueError(f"Peak in {key} must contain 'centroid' and 'nH' keys")
                        elif key == 'hsqc_nmr_peaks':
                            if '13C_centroid' not in peak or '1H_centroid' not in peak or 'nH' not in peak:
                                raise ValueError(
                                    f"Peak in {key} must contain '13C_centroid', '1H_centroid', and 'nH' keys")

    def cnmr_encode(self, spectra: List[Union[Dict, str]], device: torch.device = None) -> List[
        Union[List[torch.Tensor], str]]:
        """Encode ¹³C NMR spectra, handling missing modalities and intensity, with robust range clamping."""
        batch_size = len(spectra)
        fixed_length = 66  # Fixed length including BOS and EOS
        max_length = fixed_length - 2  # Max number of peaks
        encoded_spectra = []

        for i, spec in enumerate(spectra):
            if spec == self.missing_identification:
                encoded_spectra.append(self.missing_identification)
                continue
            # Fill data
            peaks = spec['c_nmr_peaks']
            if isinstance(peaks, str):
                if peaks == self.missing_identification:
                    encoded_spectra.append(self.missing_identification)
                    continue
                else:
                    raise TypeError
            # Calculate max intensity for normalization (if intensity exists)
            intensities = [peak.get('intensity', 0.0) for peak in peaks if 'intensity' in peak]
            max_intensity = max(intensities) if intensities else 1.0

            # Initialize tensors for each feature (with BOS, EOS, PAD)
            delta = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            intensity = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            delta[0] = self.bos_token_id
            intensity[0] = self.bos_token_id

            num_peaks = min(len(peaks), max_length)
            for j in range(num_peaks):
                peak = peaks[j]
                # Clamp delta to predefined range
                delta_value = min(max(peak['delta (ppm)'], self.CONFIG['c_nmr_delta_min']), self.CONFIG['c_nmr_delta_max']-2)
                delta[j + 1] = self.delta_discrete(delta_value)
                # Handle missing intensity
                if 'intensity' in peak:
                    # Clamp intensity to predefined range
                    intensity_value = min(max(peak['intensity'] / max_intensity, self.CONFIG['c_nmr_intensity_min']),
                                        self.CONFIG['c_nmr_intensity_max'])
                    intensity[j + 1] = self.intensity_discrete(intensity_value)
                else:
                    intensity[j + 1] = self.pad_token_id
            delta[num_peaks + 1] = self.eos_token_id
            intensity[num_peaks + 1] = self.eos_token_id

            # Move to device if specified
            if device is not None:
                delta = delta.to(device)
                intensity = intensity.to(device)

            encoded_spectra.append([delta, intensity])

        return encoded_spectra

    def hnmr_encode(self, spectra: List[Union[Dict, str]], device: torch.device = None) -> List[
        Union[List[torch.Tensor], str]]:
        """Encode ¹H NMR spectra, handling missing modalities, j_values, and category, with robust range clamping."""
        batch_size = len(spectra)
        fixed_length = 22  # Fixed length including BOS and EOS
        max_length = fixed_length - 2  # Max number of peaks
        encoded_spectra = []

        for i, spec in enumerate(spectra):
            if spec == self.missing_identification:
                encoded_spectra.append(self.missing_identification)
                continue
            # Fill data
            peaks = spec['h_nmr_peaks']
            if isinstance(peaks, str):
                if peaks == self.missing_identification:
                    encoded_spectra.append(self.missing_identification)
                    continue
                else:
                    raise TypeError

            # Initialize tensors for each feature (with BOS, EOS, PAD)
            centroids = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            nH = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            category = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            jvalue = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            centroids[0] = self.bos_token_id
            nH[0] = self.bos_token_id
            category[0] = self.bos_token_id
            jvalue[0] = self.bos_token_id

            num_peaks = min(len(peaks), max_length)
            for j in range(num_peaks):
                peak = peaks[j]
                # Clamp centroid to predefined range
                centroid_value = min(max(peak['centroid'], self.CONFIG['h_nmr_centroid_min']),
                                    self.CONFIG['h_nmr_centroid_max'])
                centroids[j + 1] = self.centroid_discrete(centroid_value)
                # Clamp nH to predefined range
                nH_value = min(max(peak['nH'], 0), self.CONFIG['max_nH'])
                nH[j + 1] = self.nH_discrete(nH_value)
                # Handle missing category
                cat_id = self.hnmr_category_dict.get(peak.get('category', 'Others'), -1)
                category[j + 1] = cat_id + self.special_token_num
                # Handle missing j_values
                try:
                    if 'j_values' in peak and peak['j_values']:
                        j_val = float(peak['j_values'].split('_')[0])
                        # Clamp j_value to predefined range
                        j_val = min(max(j_val, self.CONFIG['h_nmr_jvalue_min']), self.CONFIG['h_nmr_jvalue_max'])
                        jvalue[j + 1] = self.j_value_discrete(j_val)
                    else:
                        jvalue[j + 1] = self.pad_token_id
                except:
                    jvalue[j + 1] = self.pad_token_id
            centroids[num_peaks + 1] = self.eos_token_id
            nH[num_peaks + 1] = self.eos_token_id
            category[num_peaks + 1] = self.eos_token_id
            jvalue[num_peaks + 1] = self.eos_token_id

            # Move to device if specified
            if device is not None:
                centroids = centroids.to(device)
                nH = nH.to(device)
                category = category.to(device)
                jvalue = jvalue.to(device)

            encoded_spectra.append([centroids, nH, category, jvalue])

        return encoded_spectra

    def hsqc_encode(self, spectra: List[Union[Dict, str]], device: torch.device = None) -> List[
        Union[List[torch.Tensor], str]]:
        """Encode HSQC NMR spectra, handling missing modalities, with robust range clamping."""
        batch_size = len(spectra)
        fixed_length = 66  # Fixed length including BOS and EOS
        max_length = fixed_length - 2  # Max number of peaks
        encoded_spectra = []

        for i, spec in enumerate(spectra):
            if spec == self.missing_identification:
                encoded_spectra.append(self.missing_identification)
                continue
            # Fill data
            peaks = spec['hsqc_nmr_peaks']
            if isinstance(peaks, str):
                if peaks == self.missing_identification:
                    encoded_spectra.append(self.missing_identification)
                    continue
                else:
                    raise TypeError

            # Initialize tensors for each feature (with BOS, EOS, PAD)
            c13_centroid = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            h1_centroid = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            nH = torch.full((fixed_length,), self.pad_token_id, dtype=torch.long)
            c13_centroid[0] = self.bos_token_id
            h1_centroid[0] = self.bos_token_id
            nH[0] = self.bos_token_id

            num_peaks = min(len(peaks), max_length)
            for j in range(num_peaks):
                peak = peaks[j]
                # Clamp 13C centroid to predefined range
                c13_value = min(max(peak['13C_centroid'], self.CONFIG['c_nmr_delta_min']),
                                self.CONFIG['c_nmr_delta_max'])
                c13_centroid[j + 1] = self.delta_discrete(c13_value)
                # Clamp 1H centroid to predefined range
                h1_value = min(max(peak['1H_centroid'], self.CONFIG['h_nmr_centroid_min']),
                               self.CONFIG['h_nmr_centroid_max'])
                h1_centroid[j + 1] = self.centroid_discrete(h1_value)
                # Clamp nH to predefined range
                nH_value = min(max(peak['nH'], 0), self.CONFIG['max_nH'])
                nH[j + 1] = self.nH_discrete(nH_value)
            c13_centroid[num_peaks + 1] = self.eos_token_id
            h1_centroid[num_peaks + 1] = self.eos_token_id
            nH[num_peaks + 1] = self.eos_token_id

            # Move to device if specified
            if device is not None:
                c13_centroid = c13_centroid.to(device)
                h1_centroid = h1_centroid.to(device)
                nH = nH.to(device)

            encoded_spectra.append([c13_centroid, h1_centroid, nH])

        return encoded_spectra

    def encode(self, spectra: Dict[str, List[Union[Dict, str]]], device: torch.device = None) -> Dict[
        str, List[Union[List[torch.Tensor], str]]]:
        """Encode NMR spectra for multiple modalities, handling missing data."""
        self.validate_input(spectra)
        encoded_spectra = {}

        for modality in spectra:
            if modality == '[cnmr]':
                encoded_spectra[modality] = self.cnmr_encode(spectra[modality], device)
            elif modality == '[hnmr]':
                encoded_spectra[modality] = self.hnmr_encode(spectra[modality], device)
            elif modality == '[hsqc]':
                encoded_spectra[modality] = self.hsqc_encode(spectra[modality], device)
            elif modality == '[ir]':
                encoded_spectra[modality] = ir_encode(spectra[modality], device)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        return encoded_spectra

    def encode_from_dict_list(self, spectra_dict: List[Dict[str, List[Dict]]], device: torch.device = None) -> Dict[
        str, List[Union[List[torch.Tensor], str]]]:
        """
        Encode spectra from a list of dictionaries, where each dictionary contains modality keys
        mapping to lists of peak dictionaries.

        Args:
            spectra_dict: List of dictionaries, each containing modality keys ('c_nmr_peaks', 'h_nmr_peaks',
                          'hsqc_nmr_peaks', 'ir') mapping to lists of peak dictionaries or IR data.
            device: Optional torch device to move tensors to.

        Returns:
            Dictionary with modality keys ('[cnmr]', '[hnmr]', '[hsqc]', '[ir]') mapping to lists of
            encoded spectra (tensors or '<missing>').
        """
        self.validate_dict_list_input(spectra_dict)
        # Convert to the format expected by encode: Dict[str, List[Union[Dict, str]]]
        converted_spectra = {
            '[cnmr]': [],
            '[hnmr]': [],
            '[hsqc]': [],
            '[ir]': []
        }
        for item in spectra_dict:
            # For each modality, add the corresponding data or '<missing>'
            for modality_key, converted_key in [
                ('c_nmr_peaks', '[cnmr]'),
                ('h_nmr_peaks', '[hnmr]'),
                ('hsqc_nmr_peaks', '[hsqc]'),
                ('ir', '[ir]')
            ]:
                if modality_key in item:
                    if modality_key == 'ir':
                        converted_spectra[converted_key].append({'ir': item[modality_key]})
                    else:
                        converted_spectra[converted_key].append({modality_key: item[modality_key]})
                else:
                    converted_spectra[converted_key].append(self.missing_identification)

        return self.encode(converted_spectra, device)

    def batch_encode(self, spectra: Dict[str, List[Union[Dict, str]]], device: torch.device = None) -> Dict[
        str, List[Union[List[torch.Tensor], str]]]:
        """Batch encode NMR spectra (same as encode, for API consistency)."""
        return self.encode(spectra, device)


def ir_encode(spectra: List[Union[Dict, str]], device: torch.device = None) -> List[Union[torch.Tensor, str]]:
    """Encode IR spectra, handling missing modalities."""
    encoded_spectra = []
    for spec in spectra:
        if isinstance(spec['ir'], np.ndarray):
            ir_data = torch.tensor(spec['ir'], dtype=torch.float)
            if device is not None:
                ir_data = ir_data.to(device)
            encoded_spectra.append(ir_data)

        elif isinstance(spec['ir'], torch.Tensor):
            ir_data = torch.tensor(spec['ir'], dtype=torch.float)
            if device is not None:
                ir_data = ir_data.to(device)
            encoded_spectra.append(ir_data)

        elif isinstance(spec['ir'], str) and spec['ir'] == '<missing>':
            encoded_spectra.append('<missing>')
            continue
        else:
            raise TypeError

    return encoded_spectra