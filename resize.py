import os
import cv2

source = "./sprites/sprites/pokemon/model"
destination = "./resizedData"
destinationkeras = "./resizedData/pokemons"

def resize(keras = True):
    if os.path.exists(destination):
        print("Warning: folder resizedData already exist and might contain picture.")
    if not os.path.exists(destination):
        os.mkdir(destination)
    if keras == True:
        if not os.path.exists(destinationkeras):
            os.mkdir(destinationkeras)
    for each in os.listdir(source):
        img = cv2.imread(os.path.join(source,each),cv2.IMREAD_UNCHANGED)
        
        # If you need to crop the image:
        #crop_img = img[45:205, 19:235] #[top:bottom,left:rigth]

        # If your img is png you need to add a white background for your image or some weird transformation will happen
        trans_mask = img[:,:,3] == 0 
        img[trans_mask] = [255, 255, 255, 255] 
        # in cmd prompt where the resized image are run: ren *.png *.jpeg

        resize_img = cv2.resize(img,(128,128))
        if keras == True:
            cv2.imwrite(os.path.join(destinationkeras,each), resize_img)
        else:
            cv2.imwrite(os.path.join(destinationkeras,each), resize_img)


resize() #resize(keras=False)
