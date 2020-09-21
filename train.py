import os

from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k, optimizers
from keras import regularizers
from keras.models import Sequential
import keras
from keras.regularizers import l2

DATASET_PATH = '../food-101/images'

# Image Parameters
N_CLASSES = 101  # CHANGE HERE, total number of classes
IMG_HEIGHT = 299  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 299  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale

BATCH_SIZE = 64
num_of_epochs = 100

LARGE_CLASS_LIST = [
    'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheese_plate','cheesecake','chicken_curry','chicken_quesadilla','chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel','filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']

# Data Augmentation
datagen_obj = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=[0.9, 1.0],
    #channel_shift_range=20,
    fill_mode='reflect',
    validation_split=0.2)

train_generator = datagen_obj.flow_from_directory(
                DATASET_PATH,
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size = BATCH_SIZE,
                class_mode = 'categorical',
                classes = LARGE_CLASS_LIST,
                subset='training')

validation_generator = datagen_obj.flow_from_directory(
                DATASET_PATH,
                target_size = (IMG_HEIGHT, IMG_WIDTH),
                batch_size = BATCH_SIZE,
                class_mode = 'categorical',
                classes = LARGE_CLASS_LIST,
                subset='validation')

def OurNN(input_dims, output_dim, n, k, act= "relu"):
    base_model = keras.applications.inception_v3.InceptionV3(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False, weights='imagenet')
    # Top Model Block
    x = base_model.output
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(N_CLASSES, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())
    print("Total number of layers: ")
    print(len(model.layers))

    for layer in model.layers[:-1+len(model.layers)]:
        layer.trainable = True

    return model

model = OurNN((IMG_HEIGHT, IMG_WIDTH, CHANNELS), N_CLASSES, 10, 4)

checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback to save model weights after every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, period=1, verbose=1)

print("Compiling")
model.compile(optimizer=tf.train.AdamOptimizer(0.00007), loss='categorical_crossentropy', metrics=['accuracy'])

log = model.fit_generator(train_generator,
                          steps_per_epoch = train_generator.samples // BATCH_SIZE,
                          epochs = num_of_epochs,
                          validation_data = validation_generator,
                          validation_steps = validation_generator.samples // BATCH_SIZE,
                          verbose = 1,
                          callbacks=[cp_callback])

print(log.history)
model.save("model.h5")
