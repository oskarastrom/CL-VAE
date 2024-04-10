# CL-VAE: Conditional Latent space Variational Autoencoder
Repository containing the code for the CL-VAE anomaly detection model as described in "Improved Anomaly Detection through Conditional Latent Space VAE Ensembles"

![Latent Space MNIST](https://github.com/oskarastrom/CL-VAE/assets/28202051/176ab17d-88f2-4e0c-9bca-dab0264f9cf0)
![Latent Space Fashion-MNIST](https://github.com/oskarastrom/CL-VAE/assets/28202051/c4da5df9-58c9-4e5d-8b76-6c02038c7f2f)
![Latent Space Cifar10](https://github.com/oskarastrom/CL-VAE/assets/28202051/4d768796-6357-4d7c-a568-e21cad55c4c8)

The code runs in three steps.

- Start by batch training models in the Model-Builder.ipynb notebook.
- Calculate AUCs based on ensemble sizes using the Model_Evaluate.ipynb notebook.
- Visualize the results in using the Model_Visualizer.ipynb notebook.

