import time
import os
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.layers.merge import _Merge


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # not being spammed by tf warnings

def wasserstein_loss(y_true,y_pred):
    return K.mean(y_true*y_pred)


class RandomWeightedAverage(_Merge): # For the penalty loss we mix the fake and real image 

    def _merge_function(self, inputs):
        #input[0] = real image
        #input[1] = fake image
        #input[2] = batch size
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1]) 
    

def gradient_penalty_loss(y_true, y_pred, averaged_samples):

    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)



# Using keras generator => the dataset_path must point to a folder containing a folder

def load_dataset(dataset_path, batch_size, image_shape):
    dataset_generator = ImageDataGenerator()
    dataset_generator = dataset_generator.flow_from_directory(
        dataset_path, target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        shuffle = True,
        class_mode=None)

    return dataset_generator

# Displays a figure of the generated images and saves them in as .png image
def save_generated_images(generated_images, epoch):

    plt.figure(figsize=(5, 5))  # define the size of the whole plot

    for i in range(4): # iterate through the number of image we want, 4 here
        ax = plt.subplot(2,2,i+1) # 2 rows, 2 cols
        image = generated_images[i, :, :, :] # select the i image
        image += 1 
        image *= 127.5 # denormalize (to train we normalize the data so to create the input we need to do decode the data)
        im = ax.imshow(image.astype(np.uint8)) # convert the image in a readable form for matplotlib
        plt.axis('off') # dont show the axis

    plt.tight_layout() 
    save_name = 'generated_images/generatedSamples_epoch' + str(
        epoch + 1) + '.png' 

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0) # every given number of epoch, save this plot
    plt.pause(0.0000000001)
    plt.show()

def save_loss(batches, adversarial_loss, discriminator_loss, epoch): # To plot the losses
        plt.figure(1)
        plt.plot(batches, adversarial_loss, color='green',
                 label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue',
                 label='Discriminator Loss')
        plt.title("DCGAN Train")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        plt.savefig('trainingLossPlot.png')

# Creates the discriminator model. fake vs real, the image shape does not really matter in the discriminator
# A block inside the disciminator is Convolution-batch normalization-activation layer 
# try to increase the filter for each block
# if you have any doubt with the dimensions we added the summary of the model so check that when running the code

def construct_discriminator(image_shape):

    inp = Input(shape = [image_shape[0], image_shape[1],image_shape[2]])

    x = Conv2D(filters=64, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform')(inp)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform')(x)

    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=256, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    
    return Model([inp], x)



# Create the generator model. Here the input shape (units=16*16*256) is really important since you want to have the same shape as a real image
# even though you started with 100 value, randomly generated.
# In this case every transposed convolution double the size of the image. You could also use upsampling2D but transposed convolution worked better here

def construct_generator(noise_shape):

    noise = Input(shape=noise_shape)

    x = Dense(units=16 * 16 * 256, #16*16
                        kernel_initializer='glorot_uniform')(noise)
    x = Reshape(target_shape=(16, 16, 256))(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = Activation('relu')(x)


    x = Conv2DTranspose(filters=128, kernel_size=(5, 5) , #32*32
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filters=64, kernel_size=(5, 5), #64*64
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filters=3, kernel_size=(5, 5), #128*128
                                  strides=(2, 2), padding='same', 
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform')(x)
    x = Activation('tanh')(x)
    
    return Model([noise], x)




# Main train function

def train_dcgan(batch_size, epochs, image_shape,noise_shape, dataset_path, previous=False):

    #Construct graph with the functions previously seen

    generator = construct_generator(noise_shape)
    discriminator = construct_discriminator(image_shape)

    # If you already train this model once and want to keep training you can use previous weight
    # The name depends on the number of epoch so check the checkpoint folder and change 186-187 if needed

    if previous == True:
        print("Previous model used")
        generator.load_weights('checkpoint/gen_100_scaled_images.h5')
        discriminator.load_weights('checkpoint/dis_100_scaled_images.h5')
    else:
        print("New model created")
        
    optimizer = RMSprop(lr=0.00005) # You can change it to adam
    
    # For the gradient penalty we need to pass through the discriminator a real,fake,mixed image
    # To do that we create a new discriminator model which takes as input the real and fake image (*batch_size)
    # and output 3 target label (the mixed image is created inside the model)
    # In this new discriminator we do not train the generator hence the next line
    generator.trainable = False  

    real_img = Input(shape=image_shape) # The real image
    z_disc = Input(shape=noise_shape) # the fake image
    fake_img = generator([z_disc]) # feed the fake_image in the generator to go from latent dim to image_shape 
    fake = discriminator([fake_img]) # feed the output of generator in the disc to get the output
    valid = discriminator([real_img]) # feed the real image
    
    interpolated_img = RandomWeightedAverage()([real_img, fake_img]) # create the mixed image
    validity_interpolated = discriminator([interpolated_img]) # Pass the mixed image in the discriminator
    
    partial_gp_loss = partial(gradient_penalty_loss,averaged_samples=interpolated_img) #Gradient penalty
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    discriminator_model = Model(inputs=[real_img, z_disc],outputs=[valid, fake, validity_interpolated]) # New disc model
    discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                         optimizer=optimizer,
                         loss_weights=[1, 1, 10])
    
    
    # Now create the combined model where the disc does not train but the generator does.

    discriminator.trainable = False
    generator.trainable = True
    
    z_gen = Input(shape=noise_shape) # latent dim input
    img = generator([z_gen]) # output a fake image with image_shape shape
    valid = discriminator([img]) # pass it through the disc and get output
    generator_model = Model([z_gen], valid) 
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)
    
    # Number of iterations (reminder number_of_batches*batch_size = 1 epoch)
    # 718 is the total number of our dataset
    number_of_batches = int(718 / batch_size) -1 # -1 to not have the last batch with another size (Because The RandomWeightedAverage)

    # Variables used for the loss plot

    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

    # Dynamically change the plot

    plt.ion()

    current_batch = 0

    # Let's train the DCGAN for n epochs
    for epoch in range(epochs):

        dataset_generator = load_dataset(dataset_path, batch_size, image_shape)

        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " :")

        for batch_number in range(number_of_batches):

            start_time = time.time()

            # Get the current batch and normalize the images between -1 and 1
            real_images = dataset_generator.next()
            current_batch_size = real_images.shape[0]
            real_images /= 127.5
            real_images -= 1

            # Generate noise

            noise = np.random.normal(0, 1,size=(current_batch_size,) + (1, 1, 100))
            real_y = -np.ones(current_batch_size)
            fake_y = np.ones(current_batch_size)
            averaged_y = np.zeros(current_batch_size)

            # train the discriminator

            d_loss = discriminator_model.train_on_batch([real_images, noise],
                                                      [real_y, fake_y, averaged_y])

           

            # Now it's time to train the generator

            g_loss = generator_model.train_on_batch([noise], real_y)
            
            #for the loss graph
            discriminator_loss = np.append(discriminator_loss, d_loss[0])
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, current_batch)
            
            # Generate images for visualization

            generated_images = generator.predict([noise])


            time_elapsed = time.time() - start_time
            

            # Display the results
            print("     Batch " + str(batch_number + 1) + "/" +
                  str(number_of_batches) +
                  " generator loss | discriminator loss : " +
                  str(g_loss) + " | " + str(d_loss[0]) + ' - batch took ' +
                  str(time_elapsed) + ' s.')

            current_batch += 1
        


        # Each epoch update the loss graphs and save the generated img after 25 epochs
        save_loss(batches,adversarial_loss,discriminator_loss,epoch)
        if(epoch + 1) % 25 == 0:
            save_generated_images(generated_images, epoch)    
            #save model, weights for retraining purposes and model for generation purposes 
            generator.save_weights('checkpoint/gen_'+ str(epochs) +'.h5')
            discriminator.save_weights('checkpoint/dis_'+ str(epochs) +'.h5')
            generator.save('checkpoint/gen_model_'+ str(epochs) +'.h5')


def main():
    dataset_path = './resizedData/'
    batch_size = 32
    image_shape = (128, 128, 3)
    noise_shape = (1,1,100) # 100 is the latent dim
    epochs = 200
    train_dcgan(batch_size, epochs,
                image_shape, noise_shape, dataset_path, previous = False) 
                #previous impact only the lines 184-189, if you already trained the model and want to continue

if __name__ == "__main__":
  main()
