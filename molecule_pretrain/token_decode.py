from tokenizer import MolTranBertTokenizer, PATTERN
import re
from functional_group import functional_groups
self_define_tokens = {'[predict]':0,'[functional_group]':1,}

import re
from rdkit import Chem


# class token_decode:
#     def __init__(self, tokenizer: MolTranBertTokenizer):
#         self.tokenizer = tokenizer
#         self.regex_tokenizer = re.compile(PATTERN)
#         self.mol_range = range(0, 2362)
#         self.backup_range = range(2362, 2498)
#         self.padding_index = 2499
#         self.functional_group_range = range(2500, 2600)
#         self.operation_range = range(2600, 2600 + 100)
#         self.eos_token_id = tokenizer.convert_tokens_to_ids("<|END|>")  # 假设 EOS token
#         self.functional_groups = list(functional_groups.keys())
#         self.self_define_tokens = list(self_define_tokens.keys())
#
#     def decode(self, token_list, skip_special_tokens=True):
#         if len(token_list.shape) != 1:
#             raise ValueError("decode expects 1D tensor")
#
#         str_list = []
#         for token in token_list:
#             token = int(token.item())  # 转换为 Python int
#             if token in self.mol_range:
#                 str_list.append(self.tokenizer.decode(token))
#             elif token in self.backup_range:
#                 raise ValueError(f"Backup token {token} is not allowed")
#             elif token == self.padding_index:
#                 str_list.append('<pad>')
#             elif token == self.eos_token_id:
#                 str_list.append('<|END|>')
#             elif token in self.functional_group_range:
#                 str_list.append(self.functional_groups[token - 2500])
#             elif token in self.operation_range:
#                 str_list.append(self.self_define_tokens[token - 2600])
#             else:
#                 str_list.append("Unknown token")
#
#         # 拼接为单个 SMILES 字符串
#         if skip_special_tokens:
#             str_list = [s for s in str_list if s not in ['<pad>', '<|END|>', 'Unknown token']]
#         smiles = ''.join(str_list)
#
#         # 验证 SMILES（可选）
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return ""  # 返回空字符串表示无效 SMILES
#         except:
#             return ""
#
#         return smiles
#
#     def batch_decode(self, token_list, skip_special_tokens=True):
#         if len(token_list.shape) != 2:
#             raise ValueError("batch_decode expects 2D tensor")
#         decoded = []
#         for token in token_list:
#             print(token)
#             print(self.decode(token))
#             decoded.append(self.decode(token).split('<eos>')[0].replace('<bos>',''))
#             print(decoded[-1])
#         return decoded

class token_decode:
    def __init__(self, tokenizer: MolTranBertTokenizer):
        self.tokenizer = tokenizer
        self.regex_tokenizer = re.compile(PATTERN)
        self.mol_range = range(0, 2362)
        self.backup_range = range(2362, 2498)
        self.padding_index = 2499
        self.functional_group_range = range(2500, 2600)
        self.operation_range = range(2600, 2600 + 100)
        # 0-2362 is the mol token index
        # 2362-2499 is the backup token index
        # 2499 is the padding token index
        # 2500-2600 is the functional group token index
        # 2600-- is the operation token index
        self.functional_groups = list(functional_groups.keys())
        self.self_define_tokens = list(self_define_tokens.keys())

    def decode(self, token_list):
        # print(token_list)
        if len(token_list.shape) == 1:
            str_list = []
            for token in token_list:
                try:
                    if token in self.mol_range:
                        str_list.append(self.tokenizer.decode(token))
                    elif token in self.backup_range:
                        raise ValueError("Backup token is not allowed")
                    elif token == self.padding_index:
                        str_list.append( '<pad>')
                    elif token in self.functional_group_range:
                        str_list.append( self.functional_groups[token - 2500])

                    elif token in self.operation_range:
                        str_list.append(self.self_define_tokens[token - 2600])
                    else:
                        str_list.append("Unknown token")
                except:
                    str_list.append("Unknown token")

            return str_list

        elif len(token_list.shape) > 1:
            str_list = []
            for token in token_list:
                str_list.append(self.decode(token))
            return str_list

    def batch_decode(self, token_list, skip_special_tokens=True):
        if len(token_list.shape) != 2:
            raise ValueError("batch_decode expects 2D tensor")
        decoded = []
        for token in token_list:
            # print(''.join(self.decode(token)))
            decoded.append(''.join(self.decode(token)).split('<eos>')[0].replace('<bos>',''))
        return decoded
# from tokenizer import MolTranBertTokenizer, PATTERN
# from functional_group import functional_groups
# import torch
# import numpy as np
# import re
# from typing import List, Union
# from molecule_pretrain.MolPretrain import self_define_tokens
#
# class token_decode:
#     def __init__(self, tokenizer: 'MolTranBertTokenizer'):
#         self.tokenizer = tokenizer
#         self.regex_tokenizer = re.compile(PATTERN)  # Ensure PATTERN is defined elsewhere
#         self.mol_range = range(0, 2362)
#         self.backup_range = range(2362, 2498)
#         self.padding_index = 2499
#         self.functional_group_range = range(2500, 2600)
#         self.operation_range = range(2600, 2700)  # Adjusted to +100 as per comment
#         self.functional_groups = list(functional_groups.keys())  # Ensure functional_groups is defined
#         self.self_define_tokens = list(self_define_tokens.keys())  # Ensure self_define_tokens is defined
#
#         # Precompute lookup table for functional groups and operations
#         self._build_lookup_table()
#
#     def _build_lookup_table(self):
#         """Precompute a mapping of token IDs to their decoded strings."""
#         self.token_map = {}
#         # Padding token
#         self.token_map[self.padding_index] = '<pad>'
#         # Functional groups
#         for i, fg in enumerate(self.functional_groups):
#             self.token_map[2500 + i] = fg
#         # Operation tokens
#         for i, op in enumerate(self.self_define_tokens):
#             self.token_map[2600 + i] = op
#
#     def decode(self, token_list: Union[torch.Tensor, List[int]]) -> List[str]:
#         """
#         Decode a list or tensor of token IDs into their string representations.
#
#         Args:
#             token_list: Tensor of shape [seq_len] or [batch_size, seq_len], or a list of ints.
#
#         Returns:
#             List of decoded strings (flattened for multi-dim input).
#         """
#         # Convert to tensor if it's a list
#         if isinstance(token_list, list):
#             token_list = torch.tensor(token_list)
#
#         # Handle multi-dimensional input by flattening
#         if len(token_list.shape) > 1:
#             original_shape = token_list.shape
#             token_list = token_list.view(-1)  # Flatten to 1D
#         else:
#             original_shape = None
#
#         # Convert to CPU NumPy array for faster indexing if needed
#         tokens = token_list.cpu().numpy() if torch.is_tensor(token_list) else token_list
#
#         # Vectorized decoding
#         str_list = np.array(['Unknown token'] * len(tokens), dtype=object)
#
#         # Masks for each token category
#         mol_mask = (tokens < 2362) & (tokens >= 0)
#         backup_mask = (tokens >= 2362) & (tokens < 2498)
#         pad_mask = (tokens == self.padding_index)
#         fg_mask = (tokens >= 2500) & (tokens < 2600)
#         op_mask = (tokens >= 2600) & (tokens < 2700)
#
#         # Handle mol tokens (batch decode with tokenizer)
#         if mol_mask.any():
#             mol_tokens = tokens[mol_mask]
#             str_list[mol_mask] = self.tokenizer.decode(mol_tokens.tolist())  # Assumes tokenizer can handle lists
#
#         # Raise error for backup tokens
#         if backup_mask.any():
#             raise ValueError("Backup token is not allowed")
#
#         # Handle padding, functional groups, and operations using lookup table
#         if pad_mask.any():
#             str_list[pad_mask] = self.token_map[self.padding_index]
#         if fg_mask.any():
#             str_list[fg_mask] = [self.token_map[t] for t in tokens[fg_mask]]
#         if op_mask.any():
#             str_list[op_mask] = [self.token_map[t] for t in tokens[op_mask]]
#
#         # Convert to list and reshape if multi-dimensional
#         str_list = str_list.tolist()
#         if original_shape is not None:
#             str_list = [str_list[i:i + original_shape[1]] for i in range(0, len(str_list), original_shape[1])]
#
#         return str_list

# Example usage
# Assuming MolTranBertTokenizer, functional_groups, self_define_tokens, and PATTERN are defined
# tokenizer = MolTranBertTokenizer()
# decoder = TokenDecode(tokenizer)
# tokens = torch.tensor([[1, 2500, 2600], [2, 2499, 3]])
# decoded = decoder.decode(tokens)
# print(decoded)  # [['mol1', 'fg0', 'op0'], ['mol2', '<pad>', 'mol3']]