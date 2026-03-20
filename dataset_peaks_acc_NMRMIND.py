import os
import torch
import torch.utils.data as tud
import yaml
import gc
import logging
import tqdm
from joblib import Parallel, delayed
from tokenizer import MolTranBertTokenizer
from rdkit import Chem
# from skfp.fingerprints import MACCSFingerprint

class dataset(tud.Dataset):
    CONFIG = {
        'vacum_token_idx': 2,
        'h_nmr_max_num_peak': 20,
        'c_nmr_max_num_peak': 64,
        'hsqc_nmr_max_num_peak': 64,
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

    def __init__(self, data_path: str='',
                 which_spectra: str = 'src',
                 if_assemble: bool = False,
                 if_smiles: bool = False,
                 if_random_smiles: bool = False,
                 if_MACCS: bool = False,
                 if_mol_prompt: bool = False,
                 type: str = 'train',
                 relative_path: str = '../',
                 tasks: str = 'mol_predict_ir',
                 device: str = 'cuda:1',
                 cache_dir: str = 'cache',
                 mol_prompt_path: str = 'collect_training/smiles_data.csv',
                 batch_size: int = 16,
                 empty:bool = False,):
        """
        Initializes a dataset for cheminformatics tasks, preprocessing and caching data to disk.
        Deletes large src and tgt tensors after processing to free memory.

        Args:
            data_path: Path to YAML configuration file.
            which_spectra: 'src' or 'tgt' to select data type.
            if_functional_group: Whether to compute functional groups.
            if_assemble: Whether to process spectral data.
            if_smiles: Whether to tokenize SMILES.
            if_random_smiles: Whether to generate random SMILES.
            type: Dataset split ('train', 'valid', 'test').
            relative_path: Base path for data files.
            tasks: Task type (e.g., 'mol_predict_ir', 'cnmr_self_supervised_peaks').
            device: Device for tensors ('cuda:0', 'cpu').
            cache_dir: Directory to store preprocessed data.
            batch_size: Batch size for preprocessing.
        """
        logging.basicConfig(level=logging.INFO)
        if empty:
            pass
        else:
            # self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.device = 'cpu'
            self.spectra = which_spectra
            self.tasks = tasks
            self.cache_dir = cache_dir
            self.if_random_smiles = if_random_smiles
            self.if_MACCS = if_MACCS
            self.batch_size = batch_size
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{type}_preprocessed.pt")
            maccs_cache_file = os.path.join(cache_dir, f"{type}_maccs.pt")
            self.mol_tokenizer = MolTranBertTokenizer(vocab_file=os.path.join(relative_path, 'bert_vocab.txt'))

            self.vacum_token_idx = self.CONFIG['vacum_token_idx']
            self.nmr_bos_token = self.CONFIG['nmr_bos_token']
            self.nmr_eos_token = self.CONFIG['nmr_eos_token']
            self.nmr_pad_token = self.CONFIG['nmr_pad_token']
            self.nmr_special_token_num = self.CONFIG['nmr_special_token_num']
            self.nmr_vaccume_token_idx = self.CONFIG['nmr_vaccume_token_idx']
            self.mol_block_size = 84
            self.atom_idx = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9}
            self.hnmr_category = {
                'dd': 0, 'm': 1, 's': 2, 't': 3, 'ddd': 4, 'd': 5, 'pd': 6, 'tt': 7, 'dtdq': 8, 'dt': 9,
                'hd': 10, 'h': 11, 'q': 12, 'dq': 13, 'dtd': 14, 'dp': 15, 'ddq': 16, 'td': 17, 'dddd': 18,
                'ddt': 19, 'p': 20, 'dqdd': 21, 'hept': 22, 'qdd': 23, 'dddt': 24, 'dtdd': 25, 'ddddd': 26,
                'dtq': 27, 'dtt': 28, 'dtddd': 29, 'qd': 30, 'dqd': 31, 'ddtd': 32, 'dhept': 33, 'tq': 34,
                'ddp': 35, 'qt': 36, 'ttd': 37, 'tdd': 38, 'tdt': 39, 'tddd': 40, 'dh': 41, 'qddd': 42,
                'pt': 43, 'dqt': 44, 'dddq': 45, 'ddtt': 46, 'heptd': 47, 'dddp': 48, 'ddddtd': 49,
                'dttd': 50, 'tp': 51, 'tdq': 52, 'qdt': 53, 'qq': 54, 'pdd': 55, 'dddqd': 56, 'ttt': 57,
                'ttq': 58, 'dtdt': 59, 'th': 60, 'ddddq': 61, 'tddt': 62, 'ddddt': 63, 'ddtq': 64,
                'tqd': 65, 'dtdtd': 66, 'ddtdd': 67, 'tddq': 68, 'dpdd': 69, 'ttdt': 70, 'ddh': 71,
                'tdp': 72
            }

            if os.path.exists(cache_file):
                logging.info(f"Loading cached data from {cache_file}")
                cached = torch.load(cache_file, map_location=self.device,weights_only=False)
                self.mol_token = cached["mol_token"]
                self.h_peak = cached.get("h_peak", None)
                self.c_peak = cached.get("c_peak", None)
                self.ir = cached.get("ir", None)
                self.hsqc_peak = cached.get("hsqc_peak", None)

                self.formula = cached.get("formula", None)
                self.h_nmr = cached.get("h_nmr", None)
                self.c_nmr = cached.get("c_nmr", None)
                self.len = len(self.mol_token)
            else:
                logging.info("Preprocessing data")
                data = self.read_yaml(data_path)
                self.data = data['data'].get(type, data['data'].get('corpus_1') if type == 'train' else {})
                src = torch.load(os.path.join(relative_path, self.data['path_src']), map_location=self.device)
                tgt = torch.load(os.path.join(relative_path, self.data['path_tgt']), map_location=self.device)
                self.len = len(src)

                if if_smiles:
                    self.mol_tokenizer = MolTranBertTokenizer(vocab_file=os.path.join(relative_path, 'bert_vocab.txt'))
                self.preprocess_data(src, tgt, if_smiles, if_random_smiles, if_assemble)
                logging.info(f"Saving preprocessed data to {cache_file}")
                torch.save({
                    "mol_token": self.mol_token,
                    "h_peak": self.h_peak,
                    "c_peak": self.c_peak,
                    "ir": self.ir,
                    "hsqc_peak": self.hsqc_peak,
                    "formula": self.formula,
                    "h_nmr": self.h_nmr,
                    "c_nmr": self.c_nmr,
                }, cache_file)

                del src
                del tgt
                gc.collect()
                logging.info("Deleted src and tgt tensors")

            # Load or compute MACCS fingerprints cache
            if if_MACCS and os.path.exists(maccs_cache_file):
                logging.info(f"Loading MACCS fingerprints from {maccs_cache_file}")
                self.maccs = torch.load(maccs_cache_file, map_location=self.device)["maccs"]
            elif if_MACCS:
                logging.info("Computing MACCS fingerprints")
                data = self.read_yaml(data_path)
                self.data = data['data'].get(type, data['data'].get('corpus_1') if type == 'train' else {})
                tgt = torch.load(os.path.join(relative_path, self.data['path_tgt']), map_location=self.device)
                self.preprocess_maccs(tgt)
                logging.info(f"Saving MACCS fingerprints to {maccs_cache_file}")
                torch.save({"maccs": self.maccs}, maccs_cache_file)
                del tgt
                gc.collect()
                logging.info("Deleted tgt tensor after MACCS preprocessing")

            self.task_handlers = {
                'mol_predict_ir': lambda idx: (self.ir[idx].unsqueeze(0), self.get_mol_token(idx)),
                'mol_predict_hnmr_token': lambda idx: (self.h_peak[idx].long(), self.get_mol_token(idx)),
                'mol_predict_cnmr_token': lambda idx: (self.c_peak[idx].long(), self.get_mol_token(idx)),
                'mol_predict_hsqc_token': lambda idx: (self.hsqc_peak[idx].long(), self.get_mol_token(idx)),
                'ir_predict_mol': lambda idx: (self.get_mol_token(idx), self.ir[idx].unsqueeze(0)),
                'ir_self_supervised': lambda idx: (self.ir[idx].unsqueeze(0), self.ir[idx].unsqueeze(0)),
                'mol_self_supervised': lambda idx: (self.get_mol_token(idx)),
                'hnmr_self_supervised': lambda idx: (self.h_nmr[idx].unsqueeze(0), self.h_nmr[idx].unsqueeze(0)),

                'hnmr_ir_predict_mol': lambda idx: (
                    self.get_mol_token(idx), {'[hnmr]': self.h_peak[idx], '[ir]': self.ir[idx].unsqueeze(0)}
                ),
                'hnmr_self_supervised_peaks': lambda idx: (self.h_peak[idx].long(), self.get_mol_token(idx)),
                'cnmr_self_supervised_peaks': lambda idx: (self.c_peak[idx].long(), self.get_mol_token(idx)),
                'hsqc_self_supervised_peaks': lambda idx: (self.hsqc_peak[idx], self.get_mol_token(idx)),

                'cnmr_hnmr_ir_predict_mol': lambda idx: (
                    self.get_mol_token(idx), {
                        '[cnmr]': self.c_peak[idx].long(),
                        '[hnmr]': self.h_peak[idx].long(),
                        '[ir]': self.ir[idx].unsqueeze(0)
                    }
                ),

                'cnmr_hsqc_hnmr_ir_predict_mol': lambda idx: (
                    self.get_mol_token(idx), {
                        '[cnmr]': self.c_peak[idx].long(),
                        '[hsqc]': self.hsqc_peak[idx].long() if self.hsqc_peak is not None else torch.ones((66*3)).long()*2,
                        '[hnmr]': self.h_peak[idx].long(),
                        '[ir]': self.ir[idx].unsqueeze(0) if self.ir is not None else torch.zeros(1,512).float(),
                    }
                ),
                'mol_predict_cnmr_hnmr_ir': lambda idx: (
                    self.get_mol_token(idx), {
                        '[cnmr]': self.c_peak[idx].long(),
                        '[hsqc]': self.hsqc_peak[idx].long(),
                        '[hnmr]': self.h_peak[idx].long(),
                        '[ir]': self.ir[idx].unsqueeze(0)
                    }
                ),
                'cnmr_hnmr_predict_mol': lambda idx: (
                    self.get_mol_token(idx), {
                        '[cnmr]': self.c_peak[idx].long(),
                        '[hnmr]': self.h_peak[idx].long(),
                    }
                ),

                'hsqc_cnmr_hnmr_predict_mol': lambda idx: (
                    self.get_mol_token(idx), {
                        '[cnmr]': self.c_peak[idx].long(),
                        '[hsqc]': self.hsqc_peak[idx].long() if self.hsqc_peak is not None else torch.ones(
                            (66 * 3)).long() * 2,
                        '[hnmr]': self.h_peak[idx].long(),
                    }
                )

            }

    def read_yaml(self, yaml_path: str):
        """Reads and validates YAML configuration file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file {yaml_path} not found")
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        if 'data' not in data:
            raise ValueError("YAML file must contain 'data' key")
        return data

    def preprocess_data(self, src, tgt, if_smiles, if_random_smiles, if_assemble):
        """Preprocesses src and tgt data, handling SMILES, functional groups, and spectra."""
        if if_smiles:
            self.preprocess_smiles(tgt, self.if_random_smiles)

        if if_assemble:
            self.preprocess_assemble(src if self.spectra == 'src' else tgt)

    def preprocess_smiles(self, tgt, if_random_smiles):
        """Tokenizes SMILES strings in batches, optionally with random SMILES."""

        def tokenize_smile(smile):
            mol = Chem.MolFromSmiles(smile.replace(' ', ''))
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smile}")
            return self.mol_tokenizer(smile)['input_ids']

        max_len = 84
        if if_random_smiles:
            random_size = 8
            self.mol_token = torch.full((self.len, random_size, max_len), self.vacum_token_idx, dtype=torch.long,
                                        device=self.device)
            for start_idx in tqdm.tqdm(range(0, self.len, self.batch_size), desc='Tokenizing Random SMILES'):
                end_idx = min(start_idx + self.batch_size, self.len)
                batch_smiles = tgt[start_idx:end_idx]
                for i, smile in enumerate(batch_smiles):
                    mol = Chem.MolFromSmiles(smile.replace(' ', ''))
                    if mol is None:
                        logging.warning(f"Invalid SMILES at index {start_idx + i}: {smile}")
                        continue
                    tokens = [self.mol_tokenizer(smile)['input_ids']]
                    for r in range(1, random_size):
                        rand_smile = Chem.MolToSmiles(mol, doRandom=True)
                        tokens.append(self.mol_tokenizer(rand_smile)['input_ids'])


                    for r, token in enumerate(tokens):
                        self.mol_token[start_idx + i, r, :len(token)] = torch.tensor(token, device=self.device)
        else:
            self.mol_token = torch.full((self.len, max_len), self.vacum_token_idx, dtype=torch.long, device=self.device)
            for start_idx in tqdm.tqdm(range(0, self.len, self.batch_size), desc='Tokenizing SMILES'):
                end_idx = min(start_idx + self.batch_size, self.len)
                batch_smiles = tgt[start_idx:end_idx]
                tokens = Parallel(n_jobs=-1)(delayed(tokenize_smile)(smile) for smile in batch_smiles)
                for i, token in enumerate(tokens):
                    self.mol_token[start_idx + i, :len(token)] = torch.tensor(token, device=self.device)


    def preprocess_maccs(self, tgt):
        """Computes MACCS fingerprints from SMILES strings using scikit-chem."""

        def compute_maccs(smiles):
            mols = [Chem.MolFromSmiles(smile.replace(' ', '')) for smile in smiles]
            maccs_fp = MACCSFingerprint().transform(mols)
            return torch.tensor(maccs_fp, dtype=torch.bool, device=self.device)

        self.maccs = torch.zeros((self.len, 166), dtype=torch.bool, device=self.device)
        for start_idx in tqdm.tqdm(range(0, self.len, self.batch_size), desc='Computing MACCS Fingerprints'):
            end_idx = min(start_idx + self.batch_size, self.len)
            batch_smiles = tgt[start_idx:end_idx]
            self.maccs[start_idx:end_idx] = compute_maccs(batch_smiles)

    def preprocess_assemble(self, spectra):
        """Assembles spectral data into fixed-size tensors with BOS, EOS, and PAD tokens."""
        if_exists = {
            'formula': None, 'h_nmr_spectrum': None, 'c_nmr_spectrum': None, 'ir': None,
            'h_nmr_peaks': None, 'c_nmr_peaks': None,
            'hsqc_nmr_peaks': None
        }
        for key in if_exists:
            if key in spectra[0]:
                if_exists[key] = spectra[0][key]

        exists = [key for key, val in if_exists.items() if val is not None]
        spectra_saver = []
        for key in exists:
            if key == 'formula':
                spectra_saver.append(torch.zeros(self.len, len(self.atom_idx), device=self.device))
            elif key == 'h_nmr_peaks':
                # 4 features per peak + BOS/EOS
                spectra_saver.append(torch.full(
                    (self.len, 4 * (self.CONFIG['h_nmr_max_num_peak'] + 2)),
                    self.nmr_pad_token, device=self.device))
            elif key == 'c_nmr_peaks':
                # 2 features per peak + BOS/EOS
                spectra_saver.append(torch.full(
                    (self.len, 2 * (self.CONFIG['c_nmr_max_num_peak'] + 2)),
                    self.nmr_pad_token, device=self.device))
            elif key == 'hsqc_nmr_peaks':
                # 3 features per peak + BOS/EOS
                spectra_saver.append(torch.full(
                    (self.len, 3 * (self.CONFIG['hsqc_nmr_max_num_peak'] + 2)),
                    self.nmr_pad_token, device=self.device))
            else:
                spectra_saver.append(torch.zeros(self.len, spectra[0][key].shape[0], device=self.device))

        for num in tqdm.tqdm(range(self.len), desc='Assembling Spectra'):
            item = spectra[num]
            for key in exists:
                idx = exists.index(key)
                if key == 'formula':
                    formula = item[key].split(' ')
                    embed = torch.zeros(len(self.atom_idx), device=self.device)
                    mark = 0
                    for i in range(1, len(formula), 2):
                        atom = formula[i - 1 + mark]
                        count = formula[i + mark]
                        if count.isdigit():
                            embed[self.atom_idx[atom]] = int(count)
                        else:
                            embed[self.atom_idx[atom]] = 1
                            mark -= 1
                    spectra_saver[idx][num] = embed

                elif key == 'h_nmr_peaks':
                    h_nmr = torch.full((4, self.CONFIG['h_nmr_max_num_peak'] + 2),
                                       self.nmr_pad_token, device=self.device)
                    h_nmr[:, 0] = self.nmr_bos_token  # BOS token
                    for i, peak in enumerate(item[key][:self.CONFIG['h_nmr_max_num_peak']]):
                        h_nmr[0, i + 1] = self.centroid_discrete(peak['centroid'])
                        nH = min(peak['nH'], self.CONFIG['max_nH'])  # Cap nH
                        h_nmr[1, i + 1] = self.nH_discrete(nH)
                        h_nmr[2, i + 1] = self.hnmr_category.get(peak['category'], len(self.hnmr_category)) + self.nmr_special_token_num
                        h_nmr[3, i + 1] = (self.j_value_discrete(float(peak['j_values'].split('_')[0])) if peak[
                            'j_values'] else 0) + self.nmr_special_token_num
                    h_nmr[:, i + 2] = self.nmr_eos_token  # EOS token
                    spectra_saver[idx][num] = h_nmr.reshape(-1).long()

                elif key == 'c_nmr_peaks':
                    c_nmr = torch.full((2, self.CONFIG['c_nmr_max_num_peak'] + 2),
                                       self.nmr_pad_token, device=self.device)
                    c_nmr[:, 0] = self.nmr_bos_token  # BOS token

                    # Calculate max intensity for normalization
                    intensities = [peak['intensity'] for peak in item[key][:self.CONFIG['c_nmr_max_num_peak']]]
                    max_intensity = max(intensities) if intensities else 1.0  # Avoid division by zero
                    # print(max_intensity)
                    for i, peak in enumerate(item[key][:self.CONFIG['c_nmr_max_num_peak']]):
                        c_nmr[0, i + 1] = self.delta_discrete(peak['delta (ppm)'])
                        # Normalize intensity by dividing by max_intensity
                        c_nmr[1, i + 1] = self.intensity_discrete(peak['intensity'] / max_intensity)

                    c_nmr[:, i + 2] = self.nmr_eos_token  # EOS token
                    spectra_saver[idx][num] = c_nmr.reshape(-1).long()

                elif key == 'hsqc_nmr_peaks':
                    hsqc_nmr = torch.full((3, self.CONFIG['hsqc_nmr_max_num_peak'] + 2),
                                          self.nmr_pad_token, device=self.device)
                    hsqc_nmr[:, 0] = self.nmr_bos_token  # BOS token
                    for i, peak in enumerate(item[key][:self.CONFIG['hsqc_nmr_max_num_peak']]):
                        print(peak)
                        hsqc_nmr[0, i + 1] = self.delta_discrete(peak['13C_centroid'])
                        hsqc_nmr[1, i + 1] = self.centroid_discrete(peak['1H_centroid'])
                        hsqc_nmr[2, i + 1] = self.nH_discrete(peak['nH'])
                        # print(hsqc_nmr[:, i + 1])
                    hsqc_nmr[:, i + 2] = self.nmr_eos_token  # EOS token
                    spectra_saver[idx][num] = hsqc_nmr.reshape(-1).long()
                else:
                    spectra_saver[idx][num] = torch.tensor(item[key], device=self.device)

        for i, key in enumerate(exists):
            if key == 'formula':
                self.formula = spectra_saver[i].long()
            elif key == 'h_nmr_spectrum':
                self.h_nmr = spectra_saver[i].float()
            elif key == 'c_nmr_spectrum':
                self.c_nmr = spectra_saver[i].float()
            elif key == 'h_nmr_peaks':
                self.h_peak = spectra_saver[i].long()
            elif key == 'c_nmr_peaks':
                self.c_peak = spectra_saver[i].long()
            elif key == 'ir':
                self.ir = spectra_saver[i].float()
            elif key == 'hsqc_nmr_peaks':
                self.hsqc_peak = spectra_saver[i].long()

    def j_value_discrete(self, j_value):
        j_value = (j_value - self.CONFIG['h_nmr_jvalue_min']) / (
                    self.CONFIG['h_nmr_jvalue_max'] - self.CONFIG['h_nmr_jvalue_min']) * (
                              self.CONFIG['j_value_disc'] - 1)
        return int(j_value) + self.nmr_special_token_num

    def centroid_discrete(self, centroid):
        centroid = (centroid - self.CONFIG['h_nmr_centroid_min']) / (
                    self.CONFIG['h_nmr_centroid_max'] - self.CONFIG['h_nmr_centroid_min']) * self.CONFIG[
                       'centroid_disc']
        return int(centroid) + self.nmr_special_token_num

    def nH_discrete(self, nH):
        return nH + self.nmr_special_token_num

    def intensity_discrete(self, intensity):
        # print('c_nmr_intensity_disc', self.CONFIG[
        #                 'c_nmr_intensity_disc'])
        intensity = (intensity - self.CONFIG['c_nmr_intensity_min']) / (
                    self.CONFIG['c_nmr_intensity_max'] - self.CONFIG['c_nmr_intensity_min']) * self.CONFIG[
                        'c_nmr_intensity_disc']
        return int(intensity) + self.nmr_special_token_num

    def delta_discrete(self, delta):
        delta = (delta - self.CONFIG['c_nmr_delta_min']) / (
                    self.CONFIG['c_nmr_delta_max'] - self.CONFIG['c_nmr_delta_min']) * self.CONFIG['c_nmr_delta_disc']
        return int(delta) + self.nmr_special_token_num

    def hsqc_intensity_discrete(self, intensity):
        intensity = (intensity - self.CONFIG['hsqc_nmr_intensity_min']) / (
                    self.CONFIG['hsqc_nmr_intensity_max'] - self.CONFIG['hsqc_nmr_intensity_min']) * self.CONFIG[
                        'hsqc_nmr_intensity_disc']
        return int(intensity) + self.nmr_special_token_num

    def centroid_discrete_reverse(self, centroid):
        centroid -= self.nmr_special_token_num
        return centroid / self.CONFIG['centroid_disc'] * (
                    self.CONFIG['h_nmr_centroid_max'] - self.CONFIG['h_nmr_centroid_min']) + self.CONFIG[
            'h_nmr_centroid_min']

    def nH_discrete_reverse(self, nH):
        return nH - self.nmr_special_token_num

    def j_value_discrete_reverse(self, j_value):
        j_value -= self.nmr_special_token_num
        return (j_value - 1) / (self.CONFIG['j_value_disc'] - 1) * (
                    self.CONFIG['h_nmr_jvalue_max'] - self.CONFIG['h_nmr_jvalue_min']) + self.CONFIG['h_nmr_jvalue_min']

    def intensity_discrete_reverse(self, intensity):
        intensity -= self.nmr_special_token_num
        return (intensity - 1) / self.CONFIG['c_nmr_intensity_disc'] * (
                    self.CONFIG['c_nmr_intensity_max'] - self.CONFIG['c_nmr_intensity_min']) + self.CONFIG[
            'c_nmr_intensity_min']

    def hsqc_intensity_discrete_reverse(self, intensity):
        intensity -= self.nmr_special_token_num
        return (intensity - 1) / self.CONFIG['hsqc_nmr_intensity_disc'] * (
                    self.CONFIG['hsqc_nmr_intensity_max'] - self.CONFIG['hsqc_nmr_intensity_min']) + self.CONFIG[
            'hsqc_nmr_intensity_min']

    def delta_discrete_reverse(self, delta):
        delta -= self.nmr_special_token_num
        return (delta - 1) / self.CONFIG['c_nmr_delta_disc'] * (
                    self.CONFIG['c_nmr_delta_max'] - self.CONFIG['c_nmr_delta_min']) + self.CONFIG['c_nmr_delta_min']

    def width_discrete(self, width):
        # Placeholder: Implement based on actual requirements
        return int(width) + self.nmr_special_token_num

    def integral_discrete(self, integral):
        # Placeholder: Implement based on actual requirements
        return int(integral) + self.nmr_special_token_num

    def plot(self, num_samples=10, spectra_type='ir', save_path=None):
        """Plots specified spectra type for visualization."""
        import matplotlib.pyplot as plt
        data = getattr(self, spectra_type, None)
        if data is None:
            raise ValueError(f"{spectra_type} data not available")
        x = torch.arange(400, data.shape[1] * 2, 2) if spectra_type == 'ir' else torch.arange(data.shape[1])
        plt.plot(x.cpu(), data[:num_samples].cpu().T)
        plt.xlabel('Wavenumber' if spectra_type == 'ir' else 'Index')
        plt.ylabel('Intensity')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_mol_token(self, index):
        """Retrieves tokenized SMILES for a given index."""
        if self.if_random_smiles:
            random_idx = torch.randint(0, self.mol_token.shape[1], (1,)).item()
            return self.mol_token[index, random_idx]
        return self.mol_token[index]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.tasks in self.task_handlers:
            return self.task_handlers[self.tasks](index)


if __name__ == '__main__':
    data = dataset(
        data_path='dataset_[HNMR_CNMR_HSQCNMR_IR_Mol]/input.yaml',
        if_smiles=True,
        type='test',
        if_assemble=True,
        if_MACCS=True,
        if_random_smiles=True,
        tasks='cnmr_hnmr_ir_predict_mol',
        relative_path='',
        cache_dir='cache'
    )
    import time
    t1 = time.time()
    data_loader = tud.DataLoader(data, batch_size=16, num_workers=4, pin_memory=True, shuffle=True,
                                 )
    for i, (mol_token, spec) in enumerate(data_loader):
        if i >= 10000:
            break

    t2 = time.time()

    print(t2 - t1)

    # data = dataset(
    #     data_path='dataset_[HNMR_CNMR_HSQCNMR_IR_Mol]/input.yaml',
    #     if_smiles=True,
    #     type='train',
    #     if_assemble=True,
    #     if_random_smiles=True,
    #     if_MACCS=True,
    #     tasks='cnmr_self_supervised_peaks',
    #     relative_path='',
    #     cache_dir='cache'
    # )
    #
    # data = dataset(
    #     data_path='dataset_[HNMR_CNMR_HSQCNMR_IR_Mol]/input.yaml',
    #     if_smiles=True,
    #     type='valid',
    #     if_assemble=True,
    #     if_random_smiles=True,
    #     if_MACCS=True,
    #     tasks='cnmr_self_supervised_peaks',
    #     relative_path='',
    #     cache_dir='cache'
    # )




    dataloder = tud.DataLoader(data, batch_size=16, shuffle=True)
    mol_token, function_group = next(iter(dataloder))
    print(mol_token.shape, function_group.shape)