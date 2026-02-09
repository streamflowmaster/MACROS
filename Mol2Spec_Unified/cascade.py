import torch
import torch.nn as nn
from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from Mol2Spec_Unified.model import BERT, BERTConfig
from spectra_prediction_token.model import GPT as SpectraGPT, GPTConfig as SpectraConfig
import inspect
from typing import  List
class Mol2Spectra(nn.Module):
    def __init__(self, head_gpt: MolGPT, tail_gpt: SpectraGPT,cascade_bert:BERT,
                 frozen_model:list[str]):
        super(Mol2Spectra, self).__init__()
        self.head_gpt = head_gpt
        self.tail_gpt = tail_gpt
        self.cascade_bert = cascade_bert
        self.frozen_model = frozen_model
        self.freeze_params()

    def freeze_params(self):
        # freeze the head gpt
        if 'head_gpt' in self.frozen_model:
            for param in self.head_gpt.parameters():
                param.requires_grad = False
        if 'tail_gpt' in self.frozen_model:
            # freeze the tail gpt
            for param in self.tail_gpt.parameters():
                param.requires_grad = False
        if 'cascade_bert' in self.frozen_model:
            # freeze the cascade bert
            for param in self.cascade_bert.parameters():
                param.requires_grad = False

    def forward(self, ids, spectra):
        # the head gpt generate all the tokens
        # print(ids.shape)
        x = self.head_gpt.forward_to_embeds(ids)
        prompt,_ = self.cascade_bert.forward(x)
        # the tail gpt generate the spectra
        spectra,loss = self.tail_gpt.forward_with_prompt(spectra, prompt,training=True)
        return spectra,loss

    def forward_wce(self, ids, spectra):
        # the head gpt generate all the tokens
        x = self.head_gpt.forward_to_embeds(ids)
        prompt,_ = self.cascade_bert.forward(x)
        # the tail gpt generate the spectra
        spectra,loss = self.tail_gpt.forward_with_prompt_duringtrain_wce(spectra, prompt)
        return spectra,loss

    def rollout_logits(self,prompt_ids,prompt_completion_ids):
        if type(prompt_completion_ids) == list:
            prompt_completion_ids = torch.stack(prompt_completion_ids,dim=0).to(device=prompt_ids.device)

        x = self.head_gpt.forward_to_embeds(prompt_ids)
        prompt, _ = self.cascade_bert.forward(x)
        logits = self.tail_gpt.forward_with_prompt_logits_all(
            nmr_idx=prompt_completion_ids,
            # attention_mask=attention_mask,
            prompt=prompt
        )
        per_token_logps = []
        for idx, logit in enumerate(logits):
            log_prob = torch.softmax(logit, dim=-1).log()
            # log_prob = torch.log_softmax(logit, dim=-1)
            per_token_logp = torch.gather(
                log_prob[:, :-1],  # 移除最后一个 token 的 logit
                dim=2,
                index=prompt_completion_ids[idx, :, :].unsqueeze(-1)  # 预测下一个 token
            ).squeeze(-1)  # 形状 [batch_size * num_generations, seq_len-1]
            per_token_logps.append(per_token_logp)
        return per_token_logps

    def generate(self,prompt_ids,max_new_tokens=64):
        # the head gpt generate all the tokens
        x = self.head_gpt.forward_to_embeds(prompt_ids)
        prompt,_ = self.cascade_bert.forward(x)
        # the tail gpt generate the spectra
        spectra = self.tail_gpt.generate_with_prompt(prompt,max_new_tokens=max_new_tokens)
        return spectra

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

if __name__ == '__main__':
    # test
    head_gpt = MolGPT(MolConfig())
    tail_gpt = SpectraGPT(SpectraConfig())
    cascade_bert = BERT(BERTConfig())
    model = Mol2Spectra(head_gpt,tail_gpt,cascade_bert)
    ids = torch.randint(16, 25, (1, 100))
    spectra = torch.rand(16, 1, 512)
    spectra,loss = model(ids,spectra)
    print(spectra.shape,loss)
    print("test passed")
