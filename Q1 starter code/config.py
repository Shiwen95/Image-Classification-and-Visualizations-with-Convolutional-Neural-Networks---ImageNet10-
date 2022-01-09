"""

QUESTION 1

config.py

Defines config variables 


"""


# Define where your training and validation data is
# TODO: change this to your data root as needed
ROOT_DIR = "imagenet10/train_set/"


NORM_MEAN = [0.52283615, 0.47988218, 0.40605107]
NORM_STD = [0.29770654, 0.2888402, 0.31178293]

# Define the class labels
CLASS_LABELS = [
  "baboon",
  "banana",
  "canoe",
  "cat",
  "desk",
  "drill",
  "dumbbell",
  "football",
  "mug",
  "orange",
]