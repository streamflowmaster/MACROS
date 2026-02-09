from Mol2SpectraTraining_MultiSet import main
import yaml
import os

# train connection between molecule and spectra, as well as the IR model
mol_related_dir = 'CNMR_Multi/all_unfroz_NPPE_cnmr'
spectra_related_dir = 'CNMR_Multi/all_unfroz_NPPE_cnmr'
os.makedirs(mol_related_dir, exist_ok=True)

config_path = 'config_[all_unfroz]_NoPromptPE_cnmr_Multi.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)



spectra_path = 'cnmr_config_multiset_large.yaml'
spectra_path = os.path.join(spectra_related_dir, spectra_path)
spectra_config = yaml.load(open(spectra_path, 'r'), Loader=yaml.FullLoader)

mol_path = 'config_rotary_multiset_large.yaml'
mol_path = os.path.join(mol_related_dir, mol_path)
mol_config = yaml.load(open(mol_path, 'r'), Loader=yaml.FullLoader)

main(config,
     spectra_config,
     mol_config,
     mol_related_dir=mol_related_dir,
     spectra_related_dir=spectra_related_dir,
     MAX_NMR_PEAKS=64)