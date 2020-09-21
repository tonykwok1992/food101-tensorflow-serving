import json
from io import BytesIO
import numpy as np
import requests
from flask import Flask, request, jsonify
import operator
from PIL import Image as pil_image
import sys

app = Flask(__name__)

LARGE_CLASS_LIST = [
    'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheese_plate','cheesecake','chicken_curry','chicken_quesadilla','chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel','filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/food101/predict/', methods=['POST'])
def image_classifier():
    img = img_to_array(pil_image.open(BytesIO(request.files["file"].read())).convert('RGB').resize((299,299))) / 255.

    payload = {
        "instances": [{'images': img.tolist()}]
    }

    r = requests.post('http://tensorflow-serving:9000/v1/models/food101:predict', json=payload)

    pred = json.loads(r.content.decode('utf-8'))
    logit = dict(zip(LARGE_CLASS_LIST, np.array(pred['predictions'][0])))
    return jsonify(sorted(logit.items(), key=operator.itemgetter(1), reverse = True))

def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
    return x       
