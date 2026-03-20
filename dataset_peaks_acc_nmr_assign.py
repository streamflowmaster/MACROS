
import os
import torch
import torch.utils.data as tud
import logging
import glob
import time
import psutil
from tokenizer import MolTranBertTokenizer

class dataset(tud.Dataset):
    CONFIG = {
        'vacum_token_idx': 2,
        'h_nmr_max_num_peak': 20,
        'c_nmr_max_num_peak': 64,
        'h_nmr_centroid_min': -2,
        'h_nmr_centroid_max': 10,
        'centroid_disc': 120,
        'max_nH': 100,
        'c_nmr_delta_min': -20,
        'c_nmr_delta_max': 250,
        'c_nmr_delta_disc': 1024,
        'nmr_special_token_num': 3,
        'nmr_bos_token': 0,
        'nmr_eos_token': 1,
        'nmr_pad_token': 2,
    }

    def __init__(self, type: str='train', relative_path: str='../', tasks: str='cnmr_hnmr_predict_mol',
                 device: str='cuda:1', cache_dir: str='cache', batch_size: int=64,
                 chunk_id_range: list=None, use_int16: bool=True,set_device: str='cpu'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        self.set_device = torch.device(set_device if torch.cuda.is_available() else 'cpu')
        self.tasks = tasks
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.relative_path = relative_path
        os.makedirs(cache_dir, exist_ok=True)
        self.vacum_token_idx = self.CONFIG['vacum_token_idx']
        self.nmr_bos_token = self.CONFIG['nmr_bos_token']
        self.nmr_eos_token = self.CONFIG['nmr_eos_token']
        self.nmr_pad_token = self.CONFIG['nmr_pad_token']
        self.nmr_special_token_num = self.CONFIG['nmr_special_token_num']
        self.mol_block_size = 84
        self.use_int16 = use_int16
        self.dtype = torch.int16 if use_int16 else torch.int32

        self.chunk_id_range = chunk_id_range
        self.mol_tokenizer = MolTranBertTokenizer(vocab_file=os.path.join(relative_path, 'bert_vocab.txt'))
        self.available_modalities = []
        if 'cnmr' in tasks:
            self.available_modalities.append('[cnmr]')
        if 'hnmr' in tasks:
            self.available_modalities.append('[hnmr]')
        # self.available_modalities = ['[cnmr]', '[hnmr]']
        self.padding_hnmr = (torch.ones((2,22))*self.nmr_pad_token).to(self.set_device)
        self.padding_cnmr = (torch.ones((1,66))*self.nmr_pad_token).to(self.set_device)
        # Find chunk files
        if chunk_id_range is None:
            self.smiles_files = sorted(glob.glob(os.path.join(cache_dir, f"{type}_smiles_*.pt")))

            self.spectra_files = sorted(glob.glob(os.path.join(cache_dir, f"{type}_spectra_*.pt")))
            self.prompt_files = sorted(glob.glob(os.path.join(cache_dir, f"{type}_mol_prompts_*.pt")))
            self.assign_files = sorted(glob.glob(os.path.join(cache_dir, f"{type}_assignments_*.pt")))
        else:
            self.smiles_files = [os.path.join(cache_dir, f"{type}_smiles_{i}.pt") for i in chunk_id_range]
            self.spectra_files = [os.path.join(cache_dir, f"{type}_spectra_{i}.pt") for i in chunk_id_range]
            self.prompt_files = [os.path.join(cache_dir, f"{type}_mol_prompts_{i}.pt") for i in chunk_id_range]
            self.assign_files = [os.path.join(cache_dir, f"{type}_assignments_{i}.pt") for i in chunk_id_range]

        if not self.smiles_files or not self.spectra_files:
            raise FileNotFoundError(f"No preprocessed files in {cache_dir} for type {type}")
        if len(self.smiles_files) != len(self.spectra_files):
            raise ValueError(f"Mismatch between smiles ({len(self.smiles_files)}) and spectra ({len(self.spectra_files)}) files")

        self.total_chunks = len(self.smiles_files)
        logging.info(f"Total chunks: {self.total_chunks}")

        # # Set chunk range
        # if chunk_id_range is not None:
        #     if not (isinstance(chunk_id_range, list) and len(chunk_id_range) == 2 and
        #             0 <= chunk_id_range[0] <= chunk_id_range[1] < self.total_chunks):
        #         raise ValueError(f"Invalid chunk_id_range {chunk_id_range}. Must be a list [start, end] with 0 <= start <= end < {self.total_chunks}")
        #     self.chunk_start = chunk_id_range[0]
        #     self.chunk_end = min(chunk_id_range[1] + 1, self.total_chunks)
        # else:
        #     self.chunk_start = 0
        #     self.chunk_end = self.total_chunks
        #
        # self.chunk_valid_counts = []
        # self.total_len = 0
        #
        # # Calculate record counts
        # for i in range(self.chunk_start, self.chunk_end):
        #     smiles_data = torch.load(self.smiles_files[i], map_location=self.cpu_device, weights_only=True, mmap=True)
        #     valid_count = smiles_data['mol_token'].shape[0]
        #     self.chunk_valid_counts.append(valid_count)
        #     self.total_len += valid_count

        # if self.total_len == 0:
        #     raise ValueError(f"No valid records in preprocessed files for type {type} in chunk range {self.chunk_start} to {self.chunk_end}")
        # logging.info(f"Total records in chunk range {self.chunk_start} to {self.chunk_end}: {self.total_len}")

        # Load chunks
        self.mol_tokens_cache = []
        self.h_peaks_cache = []
        self.c_peaks_cache = []
        self.h_assign_cache = []
        self.c_assign_cache = []
        self.prompt_cache = []
        self.load_chunks()

        self.task_handlers = {
            'mol_predict_hnmr_token': lambda idx: (self.get_h_peak(idx), self.get_mol_token(idx)),
            'mol_predict_cnmr_token': lambda idx: (self.get_c_peak(idx), self.get_mol_token(idx)),
            'hnmr_self_supervised_peaks': lambda idx: (self.get_h_peak(idx),0),
            'cnmr_self_supervised_peaks': lambda idx: (self.get_c_peak(idx),0),
            'cnmr_hnmr_predict_mol': lambda idx: (
                self.get_mol_token(idx),
                {'[cnmr]': self.get_c_peak(idx),
                '[hnmr]': self.get_h_peak(idx)}

            ),
            'mol_self_supervised': lambda idx: (self.get_mol_token(idx)),
            'cnmr_predict_mol': lambda idx: (
                self.get_mol_token(idx),
                {'[cnmr]': self.get_c_peak(idx),
                 }

            ),
            'cnmr_hsqc_hnmr_ir_predict_mol': lambda idx: (
                self.get_mol_token(idx), {
                    '[cnmr]': self.get_c_peak(idx),
                    '[hsqc]': torch.ones(
                        (66 * 3)).long() * 2,
                    '[hnmr]': self.get_h_peak(idx),
                    '[ir]': torch.zeros(1, 512).float(),
                }
            ),

            'prompt_cnmr_hnmr_predict_mol': lambda idx: (
                self.get_mol_token(idx), {
                    '[cnmr]': self.get_c_peak(idx),
                    '[hnmr]': self.get_h_peak(idx),
                },self.get_prompt_token(idx)
            ),

            'mol_prompt_cnmr_hnmr_predict_mol': lambda idx: (
                self.get_mol_token(idx), {
                    '[cnmr]': self.get_c_peak(idx),
                    '[hnmr]': self.get_h_peak(idx),
                }, self.get_prompt_token(idx)
            ),

            'assign_cnmr_hnmr_predict_mol': lambda idx: (
                self.get_mol_token(idx), {
                    '[cnmr]': self.get_c_peak(idx),
                    '[hnmr]': self.get_h_peak(idx),
                }, self.get_assign(idx)
            ),
        }

    def load_chunks(self):

        for i in range(len(self.smiles_files)):
            logging.info(f"Loading chunk {i}...")
            t_start = time.time()
            smiles_data = torch.load(self.smiles_files[i], map_location=self.cpu_device, weights_only=True, mmap=True)
            spectra_data = torch.load(self.spectra_files[i], map_location=self.cpu_device, weights_only=True, mmap=True)
            prompt_data = torch.load(self.prompt_files[i], map_location=self.cpu_device, weights_only=True, mmap=True)
            assign_data = torch.load(self.assign_files[i], map_location=self.cpu_device, weights_only=True, mmap=True)

            self.mol_tokens_cache.append(smiles_data['mol_token'].to(dtype=self.dtype))
            self.h_peaks_cache.append(spectra_data['h_peak'].to(dtype=self.dtype) if spectra_data['h_peak'] is not None else None)
            self.c_peaks_cache.append(spectra_data['c_peak'].to(dtype=self.dtype) if spectra_data['c_peak'] is not None else None)
            self.h_assign_cache.append(assign_data['h_assign'].to(dtype=self.dtype) if assign_data['h_assign'] is not None else None)
            self.c_assign_cache.append(assign_data['c_assign'].to(dtype=self.dtype) if assign_data['c_assign'] is not None else None)
            self.prompt_cache.append(prompt_data.to(dtype=self.dtype))
            logging.info(f"Loaded chunk {i} with {len(self.smiles_files)} records in {time.time() - t_start:.2f} seconds")
        for i in range(len(self.mol_tokens_cache)):
            print(self.mol_tokens_cache[i].shape, self.h_peaks_cache[i].shape,self.c_peaks_cache[i].shape)
        self.mol_tokens = self.mol_tokens_cache[0].to(self.set_device)
        self.h_peaks = self.h_peaks_cache[0].to(self.set_device)
        self.c_peaks = self.c_peaks_cache[0].to(self.set_device)
        self.prompts = self.prompt_cache[0].to(self.set_device)
        self.h_assign = self.h_assign_cache[0].to(self.set_device)
        self.c_assign = self.c_assign_cache[0].to(self.set_device)
        # self.mol_tokens = torch.cat(self.mol_tokens)
        # self.h_peaks = torch.cat(self.h_peaks)
        # self.c_peaks = torch.cat(self.c_peaks)

        self.len = self.mol_tokens.shape[0]
        self.current_chunk_idx = 0

    def refresh_cache_chunks(self):
        self.current_chunk_idx += 1
        if self.current_chunk_idx >= len(self.mol_tokens_cache):
            self.current_chunk_idx = 0
        # 释放旧数据
        if hasattr(self, 'mol_tokens'):
            del self.mol_tokens
            del self.h_peaks
            del self.c_peaks
            # del self.prompts
            del self.h_assign
            del self.c_assign
            torch.cuda.empty_cache()  # 释放 GPU 内存
        self.mol_tokens = self.mol_tokens_cache[self.current_chunk_idx].to(self.set_device)
        self.h_peaks = self.h_peaks_cache[self.current_chunk_idx].to(self.set_device)
        self.c_peaks = self.c_peaks_cache[self.current_chunk_idx].to(self.set_device)
        self.prompts = self.prompt_cache[self.current_chunk_idx].to(self.set_device)
        self.h_assign = self.h_assign_cache[self.current_chunk_idx].to(self.set_device)
        self.c_assign = self.c_assign_cache[self.current_chunk_idx].to(self.set_device)
        self.len = self.mol_tokens.shape[0]
        logging.info(f"Chunk {self.current_chunk_idx} loaded to {self.set_device}")


    def get_mol_token(self, index):
        # chunk_idx, local_idx = self._get_chunk_and_local_index(index)
        mol_token = self.mol_tokens[index]
        if mol_token.dim() == 2:  # Handle if_random_smiles=True (N, random_size, 84)
            random_idx = torch.randint(0, mol_token.shape[0], (1,)).item()
            return mol_token[random_idx].long()
        return mol_token.long()

    def get_prompt_token(self, index):
        # chunk_idx, local_idx = self._get_chunk_and_local_index(index)
        mol_token = self.prompts[index]
        return mol_token.long()

    def get_assign(self, index):
        h_assign = self.h_assign[index]
        c_assign = self.c_assign[index]
        return h_assign, c_assign

    def get_h_peak(self, index):
        # chunk_idx, local_idx = self._get_chunk_and_local_index(index)
        h_peaks = torch.cat([self.h_peaks[index],self.padding_hnmr], dim=0)
        return h_peaks.view(-1).long() if self.h_peaks[index] is not None else None

    def get_c_peak(self, index):
        # chunk_idx, local_idx = self._get_chunk_and_local_index(index)
        c_peaks = torch.cat([self.c_peaks[index], self.padding_cnmr], dim=0)
        return c_peaks.view(-1).long() if self.c_peaks[index] is not None else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.task_handlers[self.tasks](index)

if __name__ == '__main__':
    import time
    import psutil
    process = psutil.Process()
    t1 = time.time()
    data = dataset(
        type='train',
        tasks='assign_cnmr_hnmr_predict_mol',
        relative_path='.',
        cache_dir='SimPub/cache_with_assign/',
        chunk_id_range=[0, 1],
        use_int16=True,
        set_device='cpu'
    )
    t2 = time.time()
    logging.info(f"Initialization took {t2 - t1:.2f} seconds")
    logging.info(f"Memory usage after init: {process.memory_info().rss / 1024**2:.2f}MB")
    data_loader = tud.DataLoader(data, batch_size=64,shuffle=True)
    for i, (mol_token, peaks, assign) in enumerate(data_loader):
        # mol_token = mol_token.to('cuda:1')
        # h_peak = h_peak.to('cuda:1') if h_peak is not None else None
        # c_peak = c_peak.to('cuda:1') if c_peak is not None else None
        # print(mol_token.shape)
        # for j, peak in enumerate(peaks):
        #     print(peak,peaks[peak])
        print(assign)
        if i == 0:
            logging.info(f"First batch loaded, memory usage: {process.memory_info().rss / 1024**2:.2f}MB")
        if i >= 10000:
            break
    t3 = time.time()
    logging.info(f"Data loading for 10000 iterations took {t3 - t2:.2f} seconds")
