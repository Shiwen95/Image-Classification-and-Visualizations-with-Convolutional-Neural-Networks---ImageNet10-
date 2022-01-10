"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


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
im=Image.open(args.image_path)


# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]



# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
    ]
)

input = transform(im) #input.size()==3*224*224
input = torch.unsqueeze(input,dim=0) #input.size()==1*3*224*224
output = model(input)



# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3 
# 
#########################################################################

""" 
Extracts a single filter from the specified convolutional layer,
	zero-indexed where 0 indicates the first conv layer.

Args:
	conv_layer_idx (int): index of convolutional layer
	model (nn.Module): PyTorch model to extract from
"""


def extract_filter(conv_layer_idx, model):

    # Extract filter

    conv_layer = model.features[conv_layer_idx]
    the_filter = conv_layer.weight.data.numpy()[filter_idx]

    return the_filter

#########################################################################
#
#        QUESTION 2.1.4
# 
#########################################################################

""" Extracts the all feature maps for all convolutional layers.

	Args:
		input (Tensor): input to model
		model (nn.Module): PyTorch model to extract from

"""

def extract_feature_maps(input, model):

    # Extract all feature maps
    # Hint: use conv_layer_indices to access

    feature_maps=[]
    input_1 = input.detach().clone()
    for index in range(12):

        if index-1 in conv_layer_indices:

            input_1 = model.features[index].forward(input_1)
            im = np.squeeze(input_1.detach().numpy())
            #im = np.transpose(im, [1, 2, 0])
            feature_maps.append(im)

        else:
            input_1 = model.features[index].forward(input_1)

    return feature_maps


#########################################################################
#
#        Visualize one filter and corresponding feature map
#
#########################################################################

#Change index I prefer
filter_idx = 180
channel_idx = 0
conv_layer_idx = 10

#Extract filter and normalization
filter = extract_filter(conv_layer_idx, model)[channel_idx]

NORM_MEAN = np.mean(filter)
NORM_STD = np.std(filter)
filter = (filter-NORM_MEAN)/NORM_STD

#Extract feature_map and transpose
feature_map = extract_feature_maps(input, model)
feature_map = feature_map[conv_layer_indices.index(conv_layer_idx)]
feature_map = np.transpose(feature_map, [1, 2, 0])

#Plot input image, filter and corresponding feature map
plt.figure()

ax = plt.subplot(1, 3 ,1)
ax.set_title('Input Image')
plt.axis('off')
plt.imshow(im)

ax = plt.subplot(1, 3 ,2)
ax.set_title('Filter[conv_layer: {} filter: {} channel: {}]'.format(conv_layer_idx,filter_idx,channel_idx))
plt.axis('off')
plt.imshow(filter, cmap='gray')

ax = plt.subplot(1, 3 ,3)
ax.set_title('Feature Maps[conv_layer: {} filter: {}]'.format(conv_layer_idx,filter_idx,channel_idx))
plt.axis('off')
plt.imshow(feature_map[:,:,filter_idx], cmap='gray')

plt.show()