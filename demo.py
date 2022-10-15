# Source: https://colab.research.google.com/drive/12x7v-_miN7-SRiziK3Cx4ffJzstBJNqb?usp=sharing&authuser=2#scrollTo=Q6tHwV3E_Cmb&uniqifier=1 

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

# def imshow(img, caption):
#     plt.imshow(img[:, :, [2, 1, 0]])
#     plt.axis("off")
#     plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)


def imshow(img, caption, ShowFig = True, SaveFig = True, path = 'tmp.png'):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)

    if SaveFig:
        plt.savefig(path)

    if ShowFig:
        plt.show()


# Use this command for evaluate the GLPT-T model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
# config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
# weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

# Retrieve an image on which we wish to test the model. Here, we use an image from the validation set of COCO
image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
caption = 'bobble heads on top of the shelf'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
imshow(result, caption)

image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
caption = 'sofa . remote . dog . person . car . sky . plane .' # the caption can also be the simple concatonation of some random categories.
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
imshow(result, caption)


def loadLocalImage(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

image_path = "/workspace/Research/GLIP/DATASET/LSU-LargeV2-Dataset/106/106-1/crop/00006.png"

image = loadLocalImage(image_path)
caption = 'sofa . remote . dog . person . car . sky . plane .' # the caption can also be the simple concatonation of some random categories.
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
imshow(result, caption)
