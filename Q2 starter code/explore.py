"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch



# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str,
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
# 
#########################################################################


# Read in image located at args.image_path




# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]



# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network







# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3 
# 
#########################################################################

def extract_filter(conv_layer_idx, model):
	""" Extracts a single filter from the specified convolutional layer,
		zero-indexed where 0 indicates the first conv layer.

		Args:
			conv_layer_idx (int): index of convolutional layer
			model (nn.Module): PyTorch model to extract from

	"""

	# Extract filter

	return the_filter

#########################################################################
#
#        QUESTION 2.1.4
# 
#########################################################################


def extract_feature_maps(input, model):
	""" Extracts the all feature maps for all convolutional layers.

		Args:
			input (Tensor): input to model
			model (nn.Module): PyTorch model to extract from

	"""

	# Extract all feature maps
	# Hint: use conv_layer_indices to access 

	return feature_maps








