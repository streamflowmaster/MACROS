import torch
import numpy as np
from dataset_spectra import  dataset as dataset_vaccum
from typing import List,Dict

hnmr_category = {'dd': 0, 'm': 1, 's': 2, 't': 3, 'ddd': 4, 'd': 5, 'pd': 6, 'tt': 7, 'dtdq': 8,
                 'dt': 9, 'hd': 10, 'h': 11, 'q': 12, 'dq': 13, 'dtd': 14, 'dp': 15, 'ddq': 16, 'td': 17,
                 'dddd': 18, 'ddt': 19, 'p': 20, 'dqdd': 21, 'hept': 22, 'qdd': 23, 'dddt': 24,
                 'dtdd': 25, 'ddddd': 26, 'dtq': 27, 'dtt': 28, 'dtddd': 29, 'qd': 30, 'dqd': 31,
                 'ddtd': 32, 'dhept': 33, 'tq': 34, 'ddp': 35, 'qt': 36, 'ttd': 37, 'tdd': 38,
                 'tdt': 39, 'tddd': 40, 'dh': 41, 'qddd': 42, 'pt': 43, 'dqt': 44, 'dddq': 45,
                 'ddtt': 46, 'heptd': 47, 'dddp': 48, 'ddddtd': 49, 'dttd': 50, 'tp': 51, 'tdq': 52,
                 'qdt': 53, 'qq': 54, 'pdd': 55, 'dddqd': 56, 'ttt': 57, 'ttq': 58, 'dtdt': 59,
                 'th': 60, 'ddddq': 61, 'tddt': 62, 'ddddt': 63, 'ddtq': 64, 'tqd': 65,
                 'dtdtd': 66, 'ddtdd': 67, 'tddq': 68, 'dpdd': 69, 'ttdt': 70, 'ddh': 71,
                 'tdp': 72, 'Others': 73}

class NMRSpectrumTokenizer:
    def __init__(self, NMR_category:str='cnmr'):
        self.CNMR_PADDING_TOKEN_ID = 0
        self.HNMR_PADDING_TOKEN_ID = 101
        self.HSQC_PADDING_TOKEN_ID = 0
        self.dataset_vaccum = dataset_vaccum(data_path=None)
        self.hnmr_category = np.array(list(hnmr_category.keys()))
        self.NMR_category = NMR_category.lower()

        if self.NMR_category == 'hnmr':
            self.eos_token_id = self.HNMR_PADDING_TOKEN_ID
            self.pad_token_id = self.HNMR_PADDING_TOKEN_ID
            self.bos_token_id = self.HNMR_PADDING_TOKEN_ID
        elif self.NMR_category == 'cnmr':
            self.eos_token_id = self.CNMR_PADDING_TOKEN_ID
            self.pad_token_id = self.CNMR_PADDING_TOKEN_ID
            self.bos_token_id = self.CNMR_PADDING_TOKEN_ID
        elif self.NMR_category == 'hsqc':
            self.eos_token_id = self.HSQC_PADDING_TOKEN_ID
            self.pad_token_id = self.HSQC_PADDING_TOKEN_ID
            self.bos_token_id = self.HSQC_PADDING_TOKEN_ID
        else:
            print('Unexpected NMR_category', self.NMR_category)

    def cnmr_decode(self, spectra_dict: list[torch.Tensor]) -> list[dict]:

        """
            Decode tokenized ¹³C NMR spectra into human-readable format.

            Args:
                spectra_dict: List of 4 tensors [delta, width, integral, intensity].
                              Each tensor has shape (batch_size, max_length).

            Returns:
                List of lists, where each inner list contains dictionaries with keys:
                'delta (ppm)', 'width', 'integral', 'intensity'.
            """

        batchsize, max_length = spectra_dict[0].shape
        de_tokenized_spectra = []
        for batch_id in range(batchsize):
            de_tokenized_spectrum = []
            delta = spectra_dict[0][batch_id]

            # Find end of sequence
            eos_id = torch.nonzero(delta == self.CNMR_PADDING_TOKEN_ID)
            eos_id = max_length if len(eos_id) == 0 else eos_id[0]

            # Extract relevant data
            delta = spectra_dict[0][batch_id, :eos_id]
            width = spectra_dict[1][batch_id, :eos_id]
            integral = spectra_dict[2][batch_id, :eos_id]
            intensity = spectra_dict[3][batch_id, :eos_id]

            # Reverse discretization
            delta_value = self.dataset_vaccum.delta_discrete_reverse(delta)
            width_value = self.dataset_vaccum.width_discrete_reverse(width)
            integral_value = self.dataset_vaccum.integral_discrete_reverse(integral)
            intensity_value = self.dataset_vaccum.intensity_discrete_reverse(intensity)

            # Create spectrum entries
            for i in range(eos_id):
                de_tokenized_spectrum.append({
                    'delta (ppm)': delta_value[i].cpu(),
                    'width (ppm)': width_value[i].cpu(),
                    'integral': integral_value[i].cpu(),
                    'intensity': intensity_value[i].cpu()
                })
            de_tokenized_spectra.append(de_tokenized_spectrum)
        return de_tokenized_spectra

    def hnmr_decode(self, spectra_dict: list[torch.Tensor]) -> list[dict]:
        batchsize, max_length = spectra_dict[0].shape
        de_tokenized_spectra = []
        for batch_id in range(batchsize):
            de_tokenized_spectrum = []
            centroids = spectra_dict[1][batch_id]

            # Find end of sequence
            eos_id = torch.nonzero(centroids == self.HNMR_PADDING_TOKEN_ID)
            eos_id = max_length if len(eos_id) == 0 else eos_id[0]

            # Extract relevant data
            centroids = spectra_dict[0][batch_id, :eos_id]
            category = spectra_dict[1][batch_id, :eos_id]
            jvalue = spectra_dict[3][batch_id, :eos_id]
            nH = spectra_dict[2][batch_id, :eos_id]

            # Reverse discretization
            centroids_value = self.dataset_vaccum.centroid_discrete_reverse(centroids)
            jvalue_value = self.dataset_vaccum.j_value_discrete_reverse(jvalue)
            category = torch.where(category < len(self.hnmr_category),
                                   category,
                                   len(self.hnmr_category) - 1)
            category = self.hnmr_category[category.cpu().numpy()]

            # Create spectrum entries
            for i in range(eos_id):
                de_tokenized_spectrum.append({
                    'category': category[i],
                    'centroid': centroids_value[i].cpu(),
                    'jvalue': jvalue_value[i].cpu(),
                    'nH': nH[i].cpu()
                })
            de_tokenized_spectra.append(de_tokenized_spectrum)
        return de_tokenized_spectra
    
    def decode(self, spectra_dict: list[torch.Tensor]) -> list:
        if self.NMR_category == 'hnmr':
            de_tokenized_spectra = self.hnmr_decode(spectra_dict)

        elif self.NMR_category == 'cnmr':
            de_tokenized_spectra = self.cnmr_decode(spectra_dict)

        return de_tokenized_spectra

    def batch_decode(self, spectra_dict: list[torch.Tensor]) -> list:
        if self.NMR_category == 'hnmr':
            de_tokenized_spectra = self.hnmr_decode(spectra_dict)

        elif self.NMR_category == 'cnmr':
            de_tokenized_spectra = self.cnmr_decode(spectra_dict)

        return de_tokenized_spectra

    def validate_input(self, spectra: List[Dict]) -> None:
        """验证输入格式和内容"""
        if not isinstance(spectra, list):
            raise TypeError("Input must be a list of dictionaries")
        for spec in spectra:
            if not isinstance(spec, dict) or len(spec) != 1:
                raise ValueError("Each spectrum must be a dictionary with one modality key")
            modality = list(spec.keys())[0]
            if modality != self.NMR_category:
                raise ValueError(f"Modality {modality} does not match tokenizer's NMR_category {self.NMR_category}")
            if not isinstance(spec[modality], list):
                raise TypeError(f"{modality} value must be a list of peak dictionaries")

    def cnmr_encode(self, spectra: List[Dict]) -> List[torch.Tensor]:
        """编码¹³C NMR谱"""
        # self.validate_input(spectra)
        batch_size = len(spectra)
        # max_length = max(len(spec['c_nmr_peaks']) for spec in spectra)  # 动态确定最大长度
        max_length = self.dataset_vaccum.c_nmr_max_num_peak
        # 初始化张量，填充值为CNMR_PADDING_TOKEN_ID
        delta = torch.full((batch_size, max_length), self.CNMR_PADDING_TOKEN_ID, dtype=torch.float)
        width = torch.full((batch_size, max_length), self.CNMR_PADDING_TOKEN_ID, dtype=torch.float)
        integral = torch.full((batch_size, max_length), self.CNMR_PADDING_TOKEN_ID, dtype=torch.float)
        intensity = torch.full((batch_size, max_length), self.CNMR_PADDING_TOKEN_ID, dtype=torch.float)

        # 填充数据
        for i, spec in enumerate(spectra):
            peaks = spec['c_nmr_peaks']
            for j, peak in enumerate(peaks[:max_length]):
                delta[i, j] = self.dataset_vaccum.delta_discrete(peak['delta (ppm)'])
                width[i, j] = self.dataset_vaccum.width_discrete(peak['width (ppm)'])
                integral[i, j] = self.dataset_vaccum.integral_discrete(peak['integral'])
                intensity[i, j] = self.dataset_vaccum.intensity_discrete(peak['intensity'])

        return [delta.long(), width.long(), integral.long(), intensity.long()]

    def hnmr_encode(self, spectra: List[Dict]) -> List[torch.Tensor]:
        """编码¹H NMR谱"""
        # self.validate_input(spectra)
        batch_size = len(spectra)
        # max_length = max(len(spec['h_nmr_peaks']) for spec in spectra)
        max_length = self.dataset_vaccum.h_nmr_max_num_peak
        # 初始化张量，填充值为HNMR_PADDING_TOKEN_ID
        centroids = torch.full((batch_size, max_length), self.HNMR_PADDING_TOKEN_ID, dtype=torch.float)
        category = torch.full((batch_size, max_length), self.HNMR_PADDING_TOKEN_ID, dtype=torch.long)
        nH = torch.full((batch_size, max_length), self.HNMR_PADDING_TOKEN_ID, dtype=torch.long)
        jvalue = torch.full((batch_size, max_length), self.HNMR_PADDING_TOKEN_ID, dtype=torch.float)

        # 填充数据
        for i, spec in enumerate(spectra):
            peaks = spec['h_nmr_peaks']
            for j, peak in enumerate(peaks[:max_length]):
                centroids[i, j] = self.dataset_vaccum.centroid_discrete(peak['centroid'])
                # 将category字符串映射为索引
                category[i, j] = hnmr_category.get(peak['category'], hnmr_category['Others'])
                nH[i, j] = peak['nH']
                if peak['j_values'] is not None:
                    jvalue[i, j] = self.dataset_vaccum.j_value_discrete(float(peak['j_values'].split('_')[0]))
                else:
                    jvalue[i,j] = 0

        return [centroids.long(), category.long(), nH.long(), jvalue.long()]

    def hsqc_encode(self, spectra: List[Dict]) -> List[torch.Tensor]:
        # self.validate_input(spectra)
        batch_size = len(spectra)
        # max_length = max(len(spec['hsqc_nmr_peaks']) for spec in spectra)
        max_length = self.dataset_vaccum.hsqc_nmr_max_num_peak
        # 初始化4个张量，填充值为HSQC_PADDING_TOKEN_ID
        c13_centroid = torch.full((batch_size, max_length), self.HSQC_PADDING_TOKEN_ID, dtype=torch.float)
        h1_centroid = torch.full((batch_size, max_length), self.HSQC_PADDING_TOKEN_ID, dtype=torch.float)
        c13_max = torch.full((batch_size, max_length), self.HSQC_PADDING_TOKEN_ID, dtype=torch.float)
        c13_min = torch.full((batch_size, max_length), self.HSQC_PADDING_TOKEN_ID, dtype=torch.float)
        nH = torch.full((batch_size, max_length), self.HSQC_PADDING_TOKEN_ID, dtype=torch.float)

        for i, spec in enumerate(spectra):
            peaks = spec['hsqc_nmr_peaks']
            for j, peak in enumerate(peaks):
                c13_centroid[i, j] = self.dataset_vaccum.delta_discrete(peak['13C_centroid'])
                h1_centroid[i, j] = self.dataset_vaccum.centroid_discrete(peak['1H_centroid'])
                c13_max[i, j] = self.dataset_vaccum.hsqc_intensity_discrete(peak['13C_max'])
                c13_min[i, j] = self.dataset_vaccum.hsqc_intensity_discrete(peak['13C_min'])
                nH[i,j] = peak['nH']

        return [c13_centroid.long(), h1_centroid.long(), c13_max.long(), c13_min.long(),nH.long()]


    def encode(self, spectra: List[Dict]) -> List[torch.Tensor]:
        """编码单个或批量NMR谱"""

        if self.NMR_category == 'hnmr':
            return self.hnmr_encode(spectra)
        elif self.NMR_category == 'cnmr':
            return self.cnmr_encode(spectra)
        elif self.NMR_category =='hsqc':
            return self.hsqc_encode(spectra)

    def batch_encode(self, spectra: List[Dict]) -> List[torch.Tensor]:
        """批量编码NMR谱（与encode相同，但保留以支持API一致性）"""
        return self.encode(spectra)

def ir_encode(spectra: List[Dict]) -> List[torch.Tensor]:
    item = spectra[0]
    ir_length = item['ir'].shape[0]
    batch_size = len(spectra)
    ir_saver = torch.zeros(batch_size,ir_length)
    for i, spec in enumerate(spectra):
        ir_saver[i,:] = torch.tensor(spec['ir'])

    return ir_saver.unsqueeze(1)

