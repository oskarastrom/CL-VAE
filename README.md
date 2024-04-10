# CL-VAE
Repository containing the code for the CL-VAE anomaly detection model as described in "Improved Anomaly Detection through Conditional Latent Space VAE Ensembles"

![LatentMNIST](figures/Latent Space MNIST.png?raw=true "Title")
![LatentFashion](figures/Latent Space Fashion-MNIST.png?raw=true "Title")
![LatentCifar10](figures/Latent Space Cifar10.png?raw=true "Title")

The code runs in three steps.

- Start by batch training models in the Model-Builder.ipynb notebook.
- Calculate AUCs based on ensemble sizes using the Model_Evaluate.ipynb notebook.
- Visualize the results in using the Model_Visualizer.ipynb notebook.

