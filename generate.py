import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
from keras.models import load_model
from matplotlib.backends.backend_agg import FigureCanvasAgg

generator = load_model('C:/Users/kevin/OneDrive/Bureau/DCGAN/AllPoke/checkpoint/gen_125_model_white.h5')



def generate(num_images):

    noise = np.random.normal(0, 1,size=(num_images,) + (1, 1, 100))
    generated_images = generator.predict(noise)
    dim = math.ceil(math.sqrt(num_images))
    fig = plt.figure(figsize=(10,10))
    for i in range(num_images):
        ax = plt.subplot(dim,dim,i+1)
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        im = ax.imshow(image.astype(np.uint8))
        plt.axis('off')
    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    return(canvas)
    
generate(7)



