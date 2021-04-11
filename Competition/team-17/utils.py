import requests
import json
import os
from skimage.io import imread
import numpy as np
from time import sleep



def blackbox_inference(token, image, label_only=False):

    if type(image) is not list:
        try:
            image = image.tolist()
        except AttributeError:
            print("Bad type for image. Please use list or Numpy array type.")

    params = {'token': token, 'label_only': label_only}
    data = {'params': params, 'image': image}
    url = 'http://model-server/blackbox_inference'

    for i in range(2):
        try:
            response = requests.post(url, json=data)
            if response.status_code != 200:
                print(response, response.content, flush=True)
                continue
            ret = json.loads(response.content)
            if ret is not None:
                return ret
        except Exception as e:
            print("Exception at blackbox_inference: {}".format(e), flush=True)
        sleep(1)
    return None


def load_images(dir_name):
    images_dict = {}
    directory = './' + dir_name
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".ppm"):
            full_name = os.path.join(directory, filename)
            im = imread(full_name).astype(np.uint8)
            images_dict[filename] = im.tolist()
    return images_dict


def submit(token, path='.'):

    white_images_list = load_images(os.path.join(path, "whitebox"))
    black_images_list = load_images(os.path.join(path, "blackbox"))
    params = {'token': token, 'white_box_images': white_images_list, 'black_box_images': black_images_list}
    url = 'http://model-server/submit'

    try:
        response = requests.post(url, json=params)
        if response.status_code != 200:
            print(response, response.content, flush=True)
        else:
            print(response.content, flush=True)
    except Exception as e:
        print("Exception at submit: {}".format(e), flush=True)
