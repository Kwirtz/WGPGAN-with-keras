# WGPGAN-with-keras
Wasserstein gradient penaly loss gan on pokemons using keras functional api

<h2>Pre-requisite</h2>
<h3>Requirements</h3>
<p>You can change tensorflow to tensorflow-gpu if you wish</p>

```
pip install -r requirements.txt
```


<h3>Data</h3>

<p>Download the data from https://github.com/PokeAPI/sprites or run</p>

```
git clone https://github.com/PokeAPI/sprites.git
```

<p> Run resize.py to homogenize the shape of data (Note that you could also use the functionality of keras to resize) </p>

```
python resize.py
```

<h2>Training</h2>

<p>Once all this is done you should have a resizedData folder and you can proceed to run WGPGAN.py</p>

```
python WGPGAN.py
```
<h2>Results</h2>
<p> The results appear in generated_images and you can generate a new image (generated.png) using generate.py <br>
generate.py make use of the checkpoint folder, maker sure the name of the model is the same in checkpoint and generate.py</p>

<p> Results : </p>
<img src="https://raw.githubusercontent.com/Kwirtz/WGPGAN-with-keras/master/generated_images/generatedSamples_epoch150.png" width="400" height="200" />


<h2>Ressources</h2>

-https://github.com/eriklindernoren/Keras-GAN <br>
-https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py <br>
-https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py <br>
-https://github.com/Neerajj9/DCGAN-Keras <br>
-https://github.com/davidreiman/mnist-wgan <br>