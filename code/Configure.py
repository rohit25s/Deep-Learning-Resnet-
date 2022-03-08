# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	# ...
}

preprocess_config = {
	"crop" : True,
	"crop_padding" : 4,
	"flip" : True,
 	"cutout" : True,
	"cutout_holes": 3,
	"cutout_length": 4,
	"batch_size": 128,

	# ...
}

DEFAULT_CFG = {
    'in_ch': 3,
    'num_classes': 10,
    'stem_width': 128,
    'down_kernel_size': 1,
    'actn': partial(nn.ReLU, inplace=True),
    'norm_layer': nn.BatchNorm2d,
    'zero_init_last_bn': True,
    'seblock': True,
    'reduction_ratio': 0.0625,
    'dropout_ratio': 0.25,
    'conv1': 'conv1.conv1.0',
    'stochastic_depth_rate': 0.0,
    'classifier': 'fc',
    'layers':None
}

training_configs = {
	"learning_rate": 0.01,
	# ...
}

### END CODE HERE