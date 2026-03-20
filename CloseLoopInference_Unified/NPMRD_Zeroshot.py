import os
import torch
import pandas as pd
import numpy as np
import json
from MaskedMultiIterativeInferrence_Formula import MoleculeInferencePipeline
from tokenizer import MolTranBertTokenizer
from Mol_Similarity_Metric import batch_total_similarity
import time
from datetime import timedelta
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))

if __name__ == '__main__':

    relative_path = '../'
    device = 'cpu'
    batch_size = 2
    sample_width = 4
    select_ratio = 0.25
    num_refine_select_cycles = 0
    modalities = ['hsqc_nmr_peaks','c_nmr_peaks','h_nmr_peaks','ir']
    results = []
    output_dir = "test_NPMRD"
    output_excel = os.path.join(output_dir, "ZeroShot_Results_Refine.xlsx")
    checkpoint_prefix = 'ZeroShot_Results_Refine_v1'
    checkpoint_count = 0
    checkpoint_interval = 2
    mol_tokenizer = MolTranBertTokenizer(vocab_file=os.path.join(relative_path, 'bert_vocab.txt'))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load NMR data from JSON
    json_file = f"{output_dir}/test_data.json"
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please ensure the file exists.")
        exit(1)

    src = data["src"]
    tgt = data["tgt"]

    # Initialize pipeline
    pipeline_init_start = time.time()
    pipeline = MoleculeInferencePipeline(
        spec2mol_dir='SPEC2Mol_Multi_From_PubChem_Formula',
        spec2mol_dir_refine= None,
        spec2mol_agent_path='all_unfroz_[hsqcnmr_cnmr_hnmr_ir]_5',
        spec2mol_config_path='config_[all_unfroz]_[hsqcnmr_cnmr_hnmr_ir]_5_stage4_large_Multi_from_pubchem_formula.yaml',
        spec2mol_agent_path_refine='all_unfroz_[hsqcnmr_cnmr_hnmr_ir]_5',
        spec2mol_config_path_refine='config_[all_unfroz]_[hsqcnmr_cnmr_hnmr_ir]_5_stage4_large_Multi_from_pubchem_refine.yaml',
        hnmr_output_dir= None,
        hnmr_agent_path='all_unfroz_NPPE_hnmr',
        hnmr_config_path='config_[all_unfroz]_NoPromptPE_hnmr_Multi.yaml',
        cnmr_output_dir= None,
        cnmr_agent_path='all_unfroz_NPPE_cnmr',
        cnmr_config_path='config_[all_unfroz]_NoPromptPE_cnmr_Multi.yaml',
        hsqc_output_dir=None,
        hsqc_agent_path='all_unfroz_NPPE_hsqc',
        hsqc_config_path='config_[all_unfroz]_NoPromptPE_hsqc.yaml',
        ir_output_dir=None,
        ir_config_path='config_[all_unfroz]_NoPromptPE.yaml',
        device=device
    )
    pipeline_init_time = time.time() - pipeline_init_start

    batch_times = []
    total_inference_time = 0
    total_batches = len(src) // batch_size + (1 if len(src) % batch_size else 0)
    for batch_idx in range(0, len(src), batch_size):
            batch_start_time = time.time()
            batch_src = src[batch_idx:batch_idx + batch_size]
            batch_tgt = tgt[batch_idx:batch_idx + batch_size]
            actual_batch_size = len(batch_src)

            # print(batch_src)
            # batch_start_time = time.time()
            # batch_data = data[batch_idx:batch_idx + batch_size]
            # batch_src = batch_data
            # batch_tgt = []
            # for s in batch_data:
            #      batch_tgt.append(s['smiles'].decode('utf-8'))

            print(batch_src)

            for s in batch_src:
                s['hsqc_nmr_peaks'] = '<missing>'
                # s['c_nmr_peaks'] = '<missing>'
                s['ir'] = '<missing>'

            actual_batch_size = len(batch_src)

            # bacth_src = input_mask(batch_src, missing_modalities=mask_modality)

            # Run pipeline
            inference_start = time.time()
            refined_results = pipeline.run_pipeline_from_formula(
                formula_prompt=['' for _ in range(actual_batch_size)],
                spectra_dict = batch_src,
                select_ratio=select_ratio,
                score_weights={'cnmr': 0.2, 'hnmr': 0.2, 'ir': 0.8, 'mol_prob': 1, 'hsqc_c': 0.2,
                               'hsqc_h': 0.2},
                draft_params={
                    'search': 'hybrid',
                    'mol_max_new_token': 84,
                    'cnmr_max_new_token': 66,
                    'hnmr_max_new_token': 22,
                    'hsqc_max_new_token': 66,
                    'sample_width': sample_width
                },
                refine_params={
                    'max_new_tokens': 84,
                    'search': 'hybrid'
                },
                num_refine_select_cycles=num_refine_select_cycles
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # Debug: Print all stages
            print(f"Batch {batch_idx}: Available stages: {[result['stage'] for result in refined_results]}")

            # Ground truth SMILES
            gt_smiles_list = [batch_tgt[b].replace(' ', '') for b in range(actual_batch_size)]

            # Process results from all stages
            for result in refined_results:
                stage = result['stage']
                print(f"\nProcessing stage: {stage}")

                # Extract SMILES based on stage
                if 'select' in stage:
                    stage_smiles = result['selected_smiles']
                    current_sample_width = int(sample_width * select_ratio)
                    scores = result.get('selected_scores', {})
                else:
                    stage_smiles = result['summary_smiles']
                    current_sample_width = sample_width
                    scores = result.get('summary_scores', {})

                # Decode SMILES
                stage_smiles_list = [
                    [mol_tokenizer.decode(stage_smiles[b, s]).split('<eos>')[0].replace('<bos>', '')
                     for s in range(current_sample_width)]
                    for b in range(actual_batch_size)
                ]

                # Compute similarity for each sample
                for sample_id in range(current_sample_width):
                    batch_stage_smiles = [stage_smiles_list[b][sample_id] for b in range(actual_batch_size)]
                    similarity_scores = batch_total_similarity(gt_smiles_list, batch_stage_smiles)

                    print(f"{stage.upper()}")
                    print(f"Shape Similarity: {np.mean(similarity_scores['shape_similarity']):.4f}")
                    print(f"MCS Similarity: {np.mean(similarity_scores['mcs_similarity']):.4f}")
                    print(f"Descriptor Similarity: {np.mean(similarity_scores['descriptor_similarity']):.4f}")
                    print(f"Fingerprint Similarity: {np.mean(similarity_scores['fingerprint_similarity']):.4f}")
                    print('**' * 25)

                    # Record results for all stages
                    for batch_id in range(actual_batch_size):
                        global_sample_idx = batch_idx + batch_id
                        gt_smiles = gt_smiles_list[batch_id]
                        pred_smiles = batch_stage_smiles[batch_id]

                        # Extract scores, handling both select and refine stages
                        cnmr_score = scores.get('cnmr', [0.0] * actual_batch_size)[batch_id]
                        hnmr_score = scores.get('hnmr', [0.0] * actual_batch_size)[batch_id]
                        ir_score = scores.get('ir', [0.0] * actual_batch_size)[batch_id]
                        mol_prob = scores.get('mol_prob', [0.0] * actual_batch_size)[batch_id]

                        try:
                            results.append({
                                'Sample Index': global_sample_idx,
                                'Batch ID': batch_id,
                                'Sample ID': sample_id,
                                'Stage': stage,
                                'GT SMILES': gt_smiles,
                                'Predicted SMILES': pred_smiles,
                                'Shape Similarity': similarity_scores['shape_similarity'][batch_id],
                                'MCS Similarity': similarity_scores['mcs_similarity'][batch_id],
                                'Descriptor Similarity': similarity_scores['descriptor_similarity'][batch_id],
                                'Fingerprint Similarity': similarity_scores['fingerprint_similarity'][batch_id],
                                'CNMR Score': cnmr_score[sample_id],
                                'HNMR Score': hnmr_score[sample_id],
                                'Mol Probability': mol_prob[sample_id],
                                'IR Score': ir_score[sample_id],
                            })
                        except:
                            results.append({
                                'Sample Index': global_sample_idx,
                                'Batch ID': batch_id,
                                'Sample ID': sample_id,
                                'Stage': stage,
                                'GT SMILES': gt_smiles,
                                'Predicted SMILES': pred_smiles,
                                'Shape Similarity': similarity_scores['shape_similarity'][batch_id],
                                'MCS Similarity': similarity_scores['mcs_similarity'][batch_id],
                                'Descriptor Similarity': similarity_scores['descriptor_similarity'][batch_id],
                                'Fingerprint Similarity': similarity_scores['fingerprint_similarity'][batch_id],
                                'CNMR Score': cnmr_score,
                                'HNMR Score': hnmr_score,
                                'Mol Probability': mol_prob,
                                'IR Score': ir_score,
                            })

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            avg_batch_time = np.mean(batch_times)
            samples_processed = batch_idx + actual_batch_size
            current_batch_num = batch_idx // batch_size + 1
            eta_seconds = avg_batch_time * (total_batches - current_batch_num)


            if batch_idx + actual_batch_size >= 550:
                break

            torch.cuda.empty_cache()

            # Print batch timing info
            print(f"Batch {current_batch_num}/{total_batches} - Inference: {format_time(inference_time)}, ")

            df = pd.DataFrame(results)
            df.to_excel(output_excel, index=False)

    # Save final results
    final_save_start = time.time()

    final_save_time = time.time() - final_save_start


    samples_processed = min(len(data), 550)

    avg_inference_per_batch = total_inference_time / len(batch_times) if batch_times else 0

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)

    # print(f"Data loading time: {format_time(data_load_time)}")
    print(f"Pipeline init time: {format_time(pipeline_init_time)}")
    print(f"Total inference time: {format_time(total_inference_time)}")
    print(f"Avg inference per batch: {format_time(avg_inference_per_batch)}")
    print(f"Final save time: {format_time(final_save_time)}")
    print(f"Samples processed: {samples_processed}")
    print(f"Final results saved to: {output_excel}")
    print("=" * 80)
    print(f"Script completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")