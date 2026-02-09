import os.path
import torch
import torch.utils.data as tud
import yaml
import random
import tqdm
from functional_group import *
from tokenizer import MolTranBertTokenizer
from rdkit import Chem

VACUM_TOKEN_IDX = 2499

def read_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


class dataset(tud.Dataset):
    def __init__(self, data_path: str,
                 which_spectra: str = 'src',
                 if_functional_group: bool = False,
                 if_assemble: bool = False,
                 if_smiles: bool = False,
                 if_random_smiles: bool = False,
                 type: str = 'train',
                 relative_path: str = '../',
                 tasks:str = 'mol_predict_ir',
                 device='cuda:1'):
        '''

        Args:
            data_path:
            which_spectra:
            if_functional_group:
            if_assemble:
            type:
            relative_path:
            tasks: mol_predict_ir, mol_predict_msms, mol_predict_nmr,
            ir_predict_mol, msms_predict_mol, nmr_predict_mol
            ir_predict_functional_group, msms_predict_functional_group, nmr_predict_functional_group
            ...
        '''
        self.device = 'cpu'
        if data_path!= None:
            if type == 'train':
                self.data = read_yaml(data_path)['data']['corpus_1']
            elif type == 'valid':
                self.data = read_yaml(data_path)['data']['valid']
            elif type == 'test':
                self.data = read_yaml(data_path)['data']['test']
            self.if_random_smiles = if_random_smiles
            self.spectra = which_spectra
            for item in self.data:
                if 'src' in item:
                    pth = os.path.join(relative_path, self.data[item])
                    self.src = torch.load(pth, map_location=self.device)
                if 'tgt' in item:
                    pth = os.path.join(relative_path, self.data[item])
                    self.tgt = torch.load(pth, map_location=self.device)
            self.len = len(self.src)
            print(self.data)

        self.atom_idx = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9}

        self.hnmr_category = {'dd': 0, 'm': 1, 's': 2, 't': 3, 'ddd': 4, 'd': 5, 'pd': 6, 'tt': 7, 'dtdq': 8,
                              'dt': 9, 'hd': 10, 'h': 11, 'q': 12, 'dq': 13, 'dtd': 14, 'dp': 15, 'ddq': 16, 'td': 17,
                              'dddd': 18, 'ddt': 19, 'p': 20, 'dqdd': 21, 'hept': 22, 'qdd': 23, 'dddt': 24,
                              'dtdd': 25, 'ddddd': 26, 'dtq': 27, 'dtt': 28, 'dtddd': 29, 'qd': 30, 'dqd': 31,
                              'ddtd': 32, 'dhept': 33, 'tq': 34, 'ddp': 35, 'qt': 36, 'ttd': 37, 'tdd': 38,
                              'tdt': 39, 'tddd': 40, 'dh': 41, 'qddd': 42, 'pt': 43, 'dqt': 44, 'dddq': 45,
                              'ddtt': 46, 'heptd': 47, 'dddp': 48, 'ddddtd': 49, 'dttd': 50, 'tp': 51, 'tdq': 52,
                              'qdt': 53, 'qq': 54, 'pdd': 55, 'dddqd': 56, 'ttt': 57, 'ttq': 58, 'dtdt': 59,
                              'th': 60, 'ddddq': 61, 'tddt': 62, 'ddddt': 63, 'ddtq': 64, 'tqd': 65,
                              'dtdtd': 66, 'ddtdd': 67, 'tddq': 68, 'dpdd': 69, 'ttdt': 70, 'ddh': 71,
                              'tdp': 72}


        self.tasks = tasks
        self.nmr_bos_token = 0
        self.nmr_eos_token = 1
        self.nmr_pad_token = 2
        self.nmr_special_token_num = 3

        # h_nmr parameters
        self.h_nmr_max_num_peak = 20

        self.h_nmr_jvalue_max = 50
        self.h_nmr_jvalue_min = 0
        self.j_value_disc = 100

        self.max_nH = 100

        self.h_nmr_centroid_min = -2
        self.h_nmr_centroid_max = 10
        self.centroid_disc = 120

        # c_nmr parameters
        # ['delta (ppm)', 'width (ppm)', 'integral', 'intensity']

        self.c_nmr_max_num_peak = 64

        self.c_nmr_delta_disc = 1024
        self.c_nmr_delta_min = -20
        self.c_nmr_delta_max = 250

        self.c_nmr_intensity_disc = 100
        self.c_nmr_intensity_min = 0
        self.c_nmr_intensity_max = 2


        self.hsqc_nmr_intensity_max = 400
        self.hsqc_nmr_intensity_min = -3
        self.hsqc_nmr_intensity_disc = 500
        self.hsqc_nmr_max_num_peak = 64


        if if_functional_group:
            self.functional_group()
            # self.statistics_function_groups()
        if if_assemble:self.assemble()

        if if_smiles:
            # 0-2362 is the mol token index
            # 2362-2499 is the backup token index
            # 2499 is the padding token index
            # 2500-2600 is the functional group token index
            # 2600-- is the operation token index

            self.vacum_token_idx = 2499 # for padding the smiles tokens
            self.mol_tokenizer = MolTranBertTokenizer(vocab_file=os.path.join(relative_path,'bert_vocab.txt'))
            if if_random_smiles:
                self.random_smiles_tokenization()
            else:
                self.smiles_tokenization()
        if data_path != None:
            del(self.src)
            del(self.tgt)

    def __len__(self):
        return self.len

    def smiles_tokenization(self):
        self.mol_token = torch.ones(self.len, 84)*self.vacum_token_idx # 76 is the max length of the tokenized smiles

        if self.spectra == 'src':
            tgt = self.tgt
        elif self.spectra == 'tgt':
            tgt = self.src
        # max_len = 0
        for idx,simle in tqdm.tqdm(enumerate(tgt), desc='Tokenization', total=self.len):
            tokens = self.mol_tokenizer(simle)['input_ids']
            self.mol_token[idx, :len(tokens)] = torch.tensor(tokens)
            # max_len = max(max_len, len(tokens))
        # print(max_len)
        self.mol_token = self.mol_token.long()

    def random_smiles_tokenization(self):
        self.random_size = 8
        self.mol_token = torch.ones(self.len,self.random_size,84)*self.vacum_token_idx # 76 is the max length of the tokenized smiles
        if self.spectra == 'src':
            tgt = self.tgt
        elif self.spectra == 'tgt':
            tgt = self.src
        # max_len = 0
        for idx,simle in tqdm.tqdm(enumerate(tgt), desc='Tokenization', total=self.len):
            tokens = self.mol_tokenizer(simle)['input_ids']
            self.mol_token[idx, 0, :len(tokens)] = torch.tensor(tokens)
            mol = Chem.MolFromSmiles(simle.replace(' ',''))
            for r in range(1,self.random_size):
                smiles = Chem.MolToSmiles(mol, doRandom=True)
                tokens = self.mol_tokenizer(smiles)['input_ids']
                self.mol_token[idx, r, :len(tokens)] = torch.tensor(tokens)
            # print(self.mol_token[idx])
        self.mol_token = self.mol_token.long()

    def functional_group(self):
        self.functional_groups = torch.zeros(self.len, len(functional_groups))
        if self.spectra == 'src':
            tgt = self.tgt
        elif self.spectra == 'tgt':
            tgt = self.src
        for idx,simle in tqdm.tqdm(enumerate(tgt), desc='Functional Group', total=self.len):
            self.functional_groups[idx]=torch.tensor(get_functional_groups(simle))
        self.functional_groups = self.functional_groups.long()

    def j_value_discrete(self, j_value):
        j_value = (j_value - self.h_nmr_jvalue_min) / (self.h_nmr_jvalue_max - self.h_nmr_jvalue_min) * (self.j_value_disc-1)
        # avoid the j_value is 0, because 0 is the none value
        return int(j_value)+ self.nmr_special_token_num

    def centroid_discrete(self, centroid):
        centroid = (centroid - self.h_nmr_centroid_min) / \
                   (self.h_nmr_centroid_max - self.h_nmr_centroid_min) * self.centroid_disc
        return int(centroid) + self.nmr_special_token_num

    def nH_discrete(self, nH):
        return nH+self.nmr_special_token_num

    def intensity_discrete(self, intensity):
        intensity = (intensity - self.c_nmr_intensity_min) / \
                   (self.c_nmr_intensity_max - self.c_nmr_intensity_min) * self.c_nmr_intensity_disc
        return int(intensity) + self.nmr_special_token_num

    def delta_discrete(self, delta):
        delta = (delta - self.c_nmr_delta_min) / \
                   (self.c_nmr_delta_max - self.c_nmr_delta_min) * self.c_nmr_delta_disc
        return int(delta) + self.nmr_special_token_num

    def hsqc_intensity_discrete(self, intensity):
        intensity = (intensity - self.hsqc_nmr_intensity_min) / \
                   (self.hsqc_nmr_intensity_max - self.hsqc_nmr_intensity_min) * self.hsqc_nmr_intensity_disc
        # print(intensity)
        return int(intensity)+ self.nmr_special_token_num

    def centroid_discrete_reverse(self, centroid):
        centroid -= self.nmr_special_token_num
        centroid = centroid / self.centroid_disc * (self.h_nmr_centroid_max - self.h_nmr_centroid_min) + self.h_nmr_centroid_min
        return centroid

    def nH_discrete_reverse(self, nH):
        return nH-self.nmr_special_token_num

    def j_value_discrete_reverse(self, j_value):
        j_value -= self.nmr_special_token_num
        j_value = (j_value - 1) / (self.j_value_disc - 1) * (self.h_nmr_jvalue_max - self.h_nmr_jvalue_min) + self.h_nmr_jvalue_min
        return j_value

    def intensity_discrete_reverse(self, intensity):
        intensity -= self.nmr_special_token_num
        intensity = (intensity - 1) / (self.c_nmr_intensity_disc) * (
                    self.c_nmr_intensity_max - self.c_nmr_intensity_min) + self.c_nmr_intensity_min
        return intensity

    def hsqc_intensity_discrete_reverse(self, intensity):
        intensity -= self.nmr_special_token_num
        intensity = (intensity - 1) / (self.hsqc_nmr_intensity_disc) * (
                    self.hsqc_nmr_intensity_max - self.hsqc_nmr_intensity_min) + self.hsqc_nmr_intensity_min
        return intensity

    def delta_discrete_reverse(self, delta):
        delta -= self.nmr_special_token_num
        delta = (delta - 1) / (self.c_nmr_delta_disc) * (
                    self.c_nmr_delta_max - self.c_nmr_delta_min) + self.c_nmr_delta_min
        return delta

    def assemble(self):
        if self.spectra == 'src':
            spectra = self.src
        elif self.spectra == 'tgt':
            spectra = self.tgt
        formula = None
        h_nmr = None
        c_nmr = None
        ir = None
        pos_msms = None
        neg_msms = None
        item = spectra[0]
        if 'formula' in item:
            formula = item['formula']
        if 'h_nmr_spectrum' in item:
            h_nmr = item['h_nmr_spectrum']
        if 'h_nmr_peaks' in item:
            h_peak = item['h_nmr_peaks']
        if 'c_nmr_spectrum' in item:
            c_nmr = item['c_nmr_spectrum']
        if 'c_nmr_peaks' in item:
            c_peak = item['c_nmr_peaks']
        if 'ir' in item:
            ir = item['ir']
        if 'pos_msms' in item:
            pos_msms = item['pos_msms']
        if 'neg_msms' in item:
            neg_msms = item['neg_msms']
        if 'hsqc_nmr_peaks' in item:
            hsqc_nmr = item['hsqc_nmr_peaks']


        if_exists = {'formula': formula, 'h_nmr_spectrum': h_nmr, 'c_nmr_spectrum': c_nmr,
                     'ir': ir, 'pos_msms': pos_msms, 'neg_msms': neg_msms,
                     'h_nmr_peaks': h_peak, 'c_nmr_peaks': c_peak,
                     'hsqc_nmr_peaks': hsqc_nmr}

        exists = []
        spectra_saver = []
        for key in if_exists:
            if if_exists[key] is not None:
                exists.append(key)
                if key == 'formula':
                    spectra_saver.append(torch.zeros(self.len, len(self.atom_idx)))
                elif key == 'h_nmr_peaks':
                    # ['centroid', "category", "nH", "j_values"]
                    spectra_saver.append(torch.zeros(self.len, 4*self.h_nmr_max_num_peak))
                elif key == 'c_nmr_peaks':
                    # ['delta (ppm)', 'width (ppm)', 'integral', 'intensity']
                    spectra_saver.append(torch.zeros(self.len, 4*self.c_nmr_max_num_peak))
                elif key == 'hsqc_nmr_peaks':
                    # ['13C_centroid', '1H_centroid', '13C_max', '13C_min' ,'nH']
                    spectra_saver.append(torch.zeros(self.len, 5*self.hsqc_nmr_max_num_peak))

                else:
                    spectra_saver.append(torch.zeros(self.len,  item[key].shape[0]))

        for num,item in tqdm.tqdm(enumerate(spectra), desc='Assembling', total=self.len):
            for key in exists:
                idx = exists.index(key)
                if key == 'formula':
                    # format of formula C 17 H 14 N 2 O 2
                    # convert to atom count
                    formula = item[key].split(' ')
                    embed = torch.zeros(len(self.atom_idx))
                    # judge if the following is a number
                    # the formula is in the form of [atom, count, atom, count, ...]
                    # but sometimes the count is missing because it is 1
                    mark = 0
                    for i in range(1, len(formula), 2):
                        atom = formula[i - 1 + mark]
                        count = formula[i+mark]
                        if count.isdigit():
                            embed[self.atom_idx[atom]] = int(count)
                        else:
                            embed[self.atom_idx[atom]] = 1
                            mark -= 1
                    spectra_saver[idx][num] = embed

                elif key == 'h_nmr_peaks':
                    # ['centroid', "category", "nH", "j_values"]
                    # ['dd', 'm', 's', 't']

                    h_nmr = torch.ones(4,self.h_nmr_max_num_peak)*self.nmr_vaccume_token_idx
                    for i, peak in enumerate(item[key]):
                        h_nmr[0,i] = self.centroid_discrete(peak['centroid'])
                        if peak['category'] in self.hnmr_category:
                            h_nmr[1,i] = self.hnmr_category[peak['category']]
                        else:
                            h_nmr[1,i] = len(self.hnmr_category)

                        h_nmr[2,i] = peak['nH']
                        if peak['nH'] > self.max_nH:
                            self.max_nH = peak['nH']
                        if peak['j_values'] is not None:
                            j_values = peak['j_values'].split('_')
                            h_nmr[3,i] = self.j_value_discrete(float(j_values[0]))

                        else:
                            h_nmr[3,i] = 0
                        if i == 9:
                            break
                    h_nmr = h_nmr.reshape(-1)

                    spectra_saver[idx][num] = h_nmr
                elif key == 'c_nmr_peaks':
                    # ['delta (ppm)', 'width (ppm)', 'integral', 'intensity']
                    c_nmr = torch.ones(4, self.c_nmr_max_num_peak)*self.c_nmr_vaccume_token_idx
                    for i, peak in enumerate(item[key]):
                        c_nmr[0, i] = self.delta_discrete(peak['delta (ppm)'])
                        c_nmr[1, i] = self.width_discrete(peak['width (ppm)'])
                        c_nmr[2, i] = self.integral_discrete(peak['integral'])
                        c_nmr[3, i] = self.intensity_discrete(peak['intensity'])
                        if i == self.c_nmr_max_num_peak-1:
                            break
                    c_nmr = c_nmr.reshape(-1).long()
                    spectra_saver[idx][num] = c_nmr

                elif key == 'hsqc_nmr_peaks':
                    # ['13C_centroid', '1H_centroid', '13C_max', '13C_min' ,'nH']
                    hsqc_nmr = torch.ones(5, self.hsqc_nmr_max_num_peak)*self.c_nmr_vaccume_token_idx
                    for i,peak in enumerate(item[key]):
                        hsqc_nmr[0,i] = self.delta_discrete(peak['13C_centroid'])
                        hsqc_nmr[1,i] = self.centroid_discrete(peak['1H_centroid'])
                        hsqc_nmr[2,i] = self.hsqc_intensity_discrete(peak['13C_max'])
                        hsqc_nmr[3,i] = self.hsqc_intensity_discrete(peak['13C_min'])
                        if peak['13C_min'] > self.hsqc_nmr_intensity_max:
                            self.hsqc_nmr_intensity_max = peak['13C_max']
                            print(self.hsqc_nmr_intensity_max)
                        if peak['13C_min'] < self.hsqc_nmr_intensity_min:
                            self.hsqc_nmr_intensity_min = peak['13C_min']
                            print(self.hsqc_nmr_intensity_min)

                        hsqc_nmr[4,i] = peak['nH']
                        if i == self.c_nmr_max_num_peak-1:
                            break
                    hsqc_nmr = hsqc_nmr.reshape(-1).long()
                    spectra_saver[idx][num] = hsqc_nmr
                else:
                    spectra_saver[idx][num] = torch.tensor(item[key])

        for i in range(len(spectra_saver)):
            dir='dataset_[IR_Mol]/regist_data/train'
            os.makedirs(dir, exist_ok=True)

            if exists[i] == 'formula':
                self.formula = spectra_saver[i].long().to(self.device)
                # torch.save(self.formula,os.path.join(dir, self.spectra + '_formula.pt'),)
            if exists[i] == 'h_nmr_spectrum':
                self.h_nmr = spectra_saver[i].float().to(self.device)
                # torch.save(self.h_nmr,os.path.join(dir, self.spectra + '_h_nmr.pt'))
            if exists[i] == 'c_nmr_spectrum':
                self.c_nmr = spectra_saver[i].to(self.device)
                # torch.save(self.c_nmr,os.path.join(dir, self.spectra + '_c_nmr.pt'))
            if exists[i] == 'h_nmr_peaks':
                self.h_peak = spectra_saver[i].to(self.device).long()
                # torch.save(self.h_peak,os.path.join(dir, self.spectra + '_h_peak.pt'))
            if exists[i] == 'c_nmr_peaks':
                self.c_peak = spectra_saver[i].to(self.device)
                # torch.save(self.c_peak,os.path.join(dir, self.spectra + '_c_peak.pt'))
            if exists[i] == 'ir':
                self.ir = spectra_saver[i].float().to(self.device)
                self.ir = spectra_saver[i].float().to(self.device)
                # torch.save(self.ir,os.path.join(dir, self.spectra + '_ir.pt'))
            if exists[i] == 'pos_msms':
                self.pos_msms = spectra_saver[i].to(self.device)
                # torch.save(self.pos_msms,os.path.join(dir, self.spectra + '_pos_msms.pt'))

            if exists[i] == 'neg_msms':
                self.neg_msms = spectra_saver[i].to(self.device)
                # torch.save(self.neg_msms,os.path.join(dir, self.spectra + '_neg_msms.pt'))

            if exists[i] == 'hsqc_nmr_peaks':
                self.hsqc_peak = spectra_saver[i].to(self.device)
                # torch.save(self.hsqc_peak,os.path.join(dir, self.spectra + '_hsqc_peak.pt'))


        print(self.formula.shape)
        # print(self.ir.shape)

    def plot(self):
        import matplotlib.pyplot as plt
        x = torch.arange(400,4000, self.ir.shape[1])
        plt.plot(self.ir[0:10].T)
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        # plt.subplots_adjust(left=0.15, right=0.95, top=1, bottom=0.35)
        plt.show()

    def statistics_function_groups(self):
        import matplotlib.pyplot as plt
        plt.bar(range(len(functional_groups)), self.functional_groups.sum(0))
        plt.xticks(range(len(functional_groups)), functional_groups.keys(), rotation=90)
        plt.subplots_adjust(left=0.15, right=0.95, top=1, bottom=0.35)
        plt.show()

    def mix(self):
        # randint
        num_of_mix = torch.randint(1, 4, (1,))
        random_idx = torch.randint(0, self.len, (num_of_mix,))
        # positive random
        concentration = torch.randint(60, 100, (num_of_mix,))/100/num_of_mix
        # print(concentration.shape, self.ir[random_idx].shape)
        ir = self.ir[random_idx]*concentration[:, None]
        functional_groups = self.functional_groups[random_idx]
        ir_mix = torch.mean(ir, dim=0)
        functional_groups_mix = torch.sum(functional_groups, dim=0)
        # item in functional_groups_mix is the existence of the functional group
        # functional_groups_mix max is 1
        functional_groups_mix = torch.where(functional_groups_mix > 0, 1, 0)
        return ir_mix, functional_groups_mix

    def get_mol_token(self, index):
        if self.if_random_smiles:
            random_idx = random.randint(0, self.random_size-1)
            mol_token = self.mol_token[index, random_idx]
            # print(mol_token.shape)
        else:
            mol_token = self.mol_token[index]
        return mol_token

    def __getitem__(self, index):
        # return self.ir[index], self.functional_groups[index]
        if self.tasks == 'mol_predict_ir':
            return self.ir[index].unsqueeze(0), self.get_mol_token(index)
        if self.tasks == 'mol_predict_hnmr_token':
            return self.h_peak[index].long(), self.get_mol_token(index)
        if self.tasks == 'mol_predict_cnmr_token':
            return self.c_peak[index].long(), self.get_mol_token(index)
        if self.tasks == 'mol_predict_hsqc_token':
            return self.hsqc_peak[index].long(), self.get_mol_token(index)
        if self.tasks == 'ir_predict_mol':
            return self.get_mol_token(index), self.ir[index].unsqueeze(0)
        if self.tasks == 'ir_self_supervised':
            return self.ir[index].unsqueeze(0), self.ir[index].unsqueeze(0)
        if self.tasks == 'mol_self_supervised':
            return self.get_mol_token(index), self.functional_groups[index]
        if self.tasks == 'hnmr_self_supervised':
            return self.h_nmr[index].unsqueeze(0), self.h_nmr[index].unsqueeze(0)
        if self.tasks == 'hnmr_self_supervised_peaks':
            return self.h_peak[index], self.h_peak[index]
        if self.tasks == 'hnmr_ir_predict_mol':
            return self.get_mol_token(index), {'[hnmr]': self.h_peak[index],
                                           '[ir]': self.ir[index].unsqueeze(0)}
        if self.tasks == 'cnmr_self_supervised_peaks':
            # print(self.c_peak[index].shape)
            return self.c_peak[index].long(), self.ir[index].unsqueeze(0)

        if self.tasks == 'cnmr_hnmr_ir_predict_mol':
            return self.get_mol_token(index), {'[cnmr]': self.c_peak[index].long(),
                                           '[hnmr]': self.h_peak[index].long(),
                                           '[ir]': self.ir[index].unsqueeze(0)}
        if self.tasks == 'hsqc_nmr_self_supervised':
            return self.hsqc_peak[index], self.hsqc_peak[index]

        if self.tasks == 'cnmr_hsqc_hnmr_ir_predict_mol':
            return self.get_mol_token(index), {'[cnmr]': self.c_peak[index].long(),
                                             '[hsqc]': self.hsqc_peak[index].long(),
                                             '[hnmr]': self.h_peak[index].long(),
                                             '[ir]': self.ir[index].unsqueeze(0)}
        if self.tasks == 'mol_predict_cnmr_hnmr_ir':
            return self.get_mol_token(index), {'[cnmr]': self.c_peak[index].long(),
                                             '[hsqc]': self.hsqc_peak[index].long(),
                                             '[hnmr]': self.h_peak[index].long(),
                                             '[ir]': self.ir[index].unsqueeze(0)}
        ir_mix, functional_groups_mix = self.mix()
        return ir_mix, functional_groups_mix

# dataset('dataset_[IR_Mol]/input.yaml')

if __name__ == '__main__':
    data = dataset('dataset_[HNMR_CNMR_IR_Mol]/input.yaml',
            if_smiles=True, type='train',
            if_functional_group=False,
            if_assemble=True,
            if_random_smiles=True,
             tasks='cnmr_self_supervised_peaks',
            relative_path='')

    # mol_token, function_group = data.__getitem__(100)
    # print(mol_token, function_group)
    # groups_idx = torch.argwhere(function_group == 1)
    # print(groups_idx)
    # # find the first 2499 in the mol_token
    # print(mol_token[:torch.where(mol_token == 2499)[0][0]])


    dataloder = tud.DataLoader(data, batch_size=16, shuffle=True)
    mol_token, function_group = next(iter(dataloder))
    # print(mol_token.shape, function_group.shape)
    # print(dataloder.dataset.h_nmr_jvalue_max, dataloder.dataset.max_nH,
    #       dataloder.dataset.h_nmr_jvalue_min, dataloder.dataset.j_value_disc)

    print(mol_token, function_group)
    print(mol_token.shape, function_group.shape)

    # find the first 2499 in the mol_token
    # print(mol_token[:torch.where(mol_token == 2499)[0][0]])
    # groups_idx = torch.argwhere(function_group == 1)
    # print(groups_idx)