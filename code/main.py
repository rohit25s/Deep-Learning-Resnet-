### YOUR CODE HERE

import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs
from Configure import preprocess_config

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':

	model = MyModel()

	train_loader, test_loader=load_data('temp',preprocess_config)
	model.train(train_loader,test_loader,250,0.1,0.00005,0.9)




