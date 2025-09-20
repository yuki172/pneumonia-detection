DATA_PATH = "data"

IMAGE_SIZE = (224, 224)

TRANSLATE = 0.1
SCALE = 0.2
HUE = 0.1
SATURATION = 0.6
ANGLE = 30


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 64
LEARNING_RATE = 1e-4

NUM_CLASSES = 2

CLASSES = {0: "Pneumonia", 1: "Normal"}

TRUE_POSITIVE = "true-positive-pneumonia"
TRUE_NEGATIVE = "true-negative-pneumonia"
FALSE_POSITIVE = "false-positive"
FALSE_NEGATIVE = "false-negative"
