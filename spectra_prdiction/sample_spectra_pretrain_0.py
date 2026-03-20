from Inferrence import main
import yaml
import os

config_path = 'config.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
main(config,num_samples=10,start_length=10)