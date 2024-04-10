import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist, fashion_mnist, cifar10
from keras import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Layer, Dense, Lambda, Input, Conv2D, Flatten, Reshape, Conv2DTranspose

from scipy.stats import multivariate_normal




#-------------------------- LOAD DATA ----------------------
def get_config(config_edit):
    num_classes = config_edit['num_classes'] if 'num_classes' in config_edit else 9
    img_size = config_edit['network']['img_dim'] if 'img_dim' in config_edit['network'] else 28
    gaussian_mode = config_edit['gaussian_mode'] if 'gaussian_mode' in config_edit else 'fixed'

    gaussian_modes = {
        "learn_mean": {
            'mode': 'learn_mean',
            'num_classes': num_classes,
            'random_order': True
        },
        "fixed": {
            'mode': 'fixed',
            'num_classes': num_classes,
            'random_order': True,
            'r': 3,
        }
    }
    gaussian_config = gaussian_modes[gaussian_mode]

    config = {
        'dataset': 'mnist',
        'anomalous_digit': 0,
        'num_classes': num_classes,
        "loss": {
            "alpha": 1/6,
            "type": 'normal' # normal | with_p 
        },
        "network": {
            "optimizer": 'adam',
            "img_dim": img_size,
            "color_dim": 1,
            "original_dim": img_size**2,
            "intermediate_dims": [[32,3,2],[64,3,2]],
            "latent_dim": 2  
        },
        "gaussian_config": gaussian_config,
        "run_info": {
            "batch_size": 100,
            "epochs": 50
        },
    }   

    def edit_recur(conf, edit):
        for setting in edit:
            if type(edit[setting]) == dict:
                if (setting not in conf): conf[setting] = {}
                edit_recur(conf[setting], edit[setting])
            else:
                conf[setting] = edit[setting]
    
    edit_recur(config, config_edit)

    return config

def load_data(config):


    if config["dataset"] == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif config["dataset"] == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif config["dataset"] == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if "network" not in config: config["network"] = {}
    config['network']['img_dim'] = x_train.shape[1]
    config['network']['color_dim'] = 1
    if len(x_train.shape) > 3: config['network']['color_dim'] = x_train.shape[3]

    config = get_config(config)

    anomalous_digit = config["anomalous_digit"]

    x_train = x_train / 255.
    x_test = x_test / 255.

    if config["dataset"] == 'cifar10':
        if len(y_train.shape) > 1: y_train = np.reshape(y_train, [-1])
        if len(y_test.shape) > 1: y_test = np.reshape(y_test, [-1])
    
    x_train = np.reshape(x_train, [-1,config["network"]["img_dim"], config["network"]["img_dim"], config["network"]["color_dim"]])
    x_test  = np.reshape(x_test,  [-1,config["network"]["img_dim"], config["network"]["img_dim"], config["network"]["color_dim"]])

    zero_filter_train = y_train == anomalous_digit
    zero_filter_test = y_test == anomalous_digit
    x_anom = np.vstack([x_train[zero_filter_train, :, :, :], x_test[zero_filter_test, :, :, :]])
    y_anom = np.hstack([y_train[zero_filter_train], y_test[zero_filter_test]])
    filter_arr = y_train != anomalous_digit
    x_train = x_train[filter_arr, :, :, :]
    y_train = y_train[filter_arr]
    y_train[y_train > anomalous_digit] = y_train[y_train > anomalous_digit] - 1
    filter_arr = y_test != anomalous_digit
    x_test = x_test[filter_arr, :, :, :]
    y_test = y_test[filter_arr]
    y_test[y_test > anomalous_digit] = y_test[y_test > anomalous_digit] - 1
    
    return x_train, y_train, x_test, y_test, x_anom, y_anom, config

#-------------------------- BUILD MODEL ----------------------
def build_clvae(config):
    network = config["network"]
    latent_dim = network["latent_dim"]

    #-------------------------- DEFINE LAYERS ----------------------

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    class Gaussian(Layer):
        """A simple layer that computes the stats for each Gaussian outputs the mean.
        """
        def __init__(self, gaussian_config, **kwargs):

            self.config = gaussian_config
            self.mode = gaussian_config["mode"]
            self.num_classes = gaussian_config["num_classes"]

            self.random_indices = np.array(range(self.num_classes))
            np.random.shuffle(self.random_indices)
            self.sphere_points = np.array([
                [1,0,0],[-0.5,np.sqrt(3)/2,0],[-0.5,-np.sqrt(3)/2,0],
                [1/3, np.sqrt(3)/3, np.sqrt(5)/3], [-2/3, 0, np.sqrt(5)/3], [1/3, -np.sqrt(3)/3, np.sqrt(5)/3],
                [1/3, np.sqrt(3)/3, -np.sqrt(5)/3], [-2/3, 0, -np.sqrt(5)/3], [1/3, -np.sqrt(3)/3, -np.sqrt(5)/3]
            ])

            super(Gaussian, self).__init__(**kwargs)

        def build(self, input_shape):
            latent_dim = input_shape[-1]
            if self.mode == 'learn_mean':
                self.mean = self.add_weight(name='mean', shape=(self.num_classes, latent_dim), initializer='zeros')
            else:
                self.mean = np.zeros((self.num_classes, latent_dim), dtype='f')

                for i in range(self.num_classes):
                    if latent_dim == 2:
                        index = i
                        if self.config["random_order"]: 
                            index = self.random_indices[i]
                        phi = 2*3.141592*index/self.num_classes

                        radius = self.config["r"]


                        self.mean[i, :] = [radius*np.cos(phi), radius*np.sin(phi)]
                    elif latent_dim == 3:
                        index = i
                        if self.config["random_order"]: 
                            index = self.random_indices[i]
                        radius = self.config["r"]
                        self.mean[i, :] = radius*self.sphere_points[index,:]
                        #Edit: That this is optimal was proved in 1951 by:
                        #K. Schütte and B. L. van der Waerden, Auf welcher Kugel haben 5, 6, 7, 8 oder 9 Punkte mit Mindestabstand
                        #1 Platz?, Math. Ann. 123 (1951), 96–124.




        def call(self, inputs):
            z = inputs
            z = K.expand_dims(z, 1)
            return z - K.expand_dims(self.mean, 0)

        def compute_output_shape(self, input_shape):
            return (None, self.num_classes, input_shape[-1])
        
    layers = network['intermediate_dims']
    # Encoder
    x = keras.Input(shape=(network["img_dim"], network["img_dim"], config["network"]["color_dim"]))
    h = x
    for layer in layers:
        h = Conv2D(layer[0], layer[1], activation="relu", strides=layer[2], padding="same")(h)
    h = Flatten()(h)
    h = Dense(16, activation="relu")(h)
    z_mean = Dense(network["latent_dim"], name='z_mean')(h)
    z_log_var = Dense(network["latent_dim"], name='z_log_var')(h)
    clvae_encoder = Model(x, z_mean)

    reverse_layers = []
    internal_size = network["img_dim"]
    for layer in layers:
        reverse_layers = [Conv2DTranspose(layer[0], layer[1], activation="relu", strides=layer[2], padding="same")] + reverse_layers
        internal_size /= layer[2]
    internal_size = int(internal_size)

    clvae_decoder = Sequential(
        [Dense(internal_size * internal_size * layers[-1][0], activation="relu")] + 
        [Reshape((internal_size, internal_size, layers[-1][0]))] + 
        reverse_layers + 
        [Conv2DTranspose(config["network"]["color_dim"], 3, activation="sigmoid", padding="same")],
        name='Decoder'
    )

    # Class Input
    y_in = Input(shape=(config["num_classes"],), name='input_y')

    # Reconstruction
    z = Lambda(sampling, output_shape=(latent_dim,), name='sampling')([z_mean, z_log_var])
    x_recon = clvae_decoder(z)

    # Separation
    gaussian = Gaussian(config["gaussian_config"], name='priors')
    z_prior_mean = gaussian(z)

    # Full Model
    clvae = Model([x, y_in], [x_recon, z_prior_mean])


    #-------------------------- LOSS ----------------------


    z_mean = K.expand_dims(z_mean, 1)
    z_log_var = K.expand_dims(z_log_var, 1)

    # reconstruction loss
    #xent_loss =  K.mean(0.5 * K.mean((x - x_recon)**2, 0))
    # Reconstruction loss
    xent_loss = network['original_dim']*keras.metrics.binary_crossentropy( K.flatten(x), K.flatten(x_recon) )
    # KL divergence
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.batch_dot(K.expand_dims(y_in, 1), K.square(z_prior_mean)) - K.exp(z_log_var), axis=-1)

    loss =  config["loss"]["alpha"] * xent_loss + (1-config["loss"]["alpha"]) * kl_loss
    # loss = config["loss"]["alpha"] * K.sum(xent_loss) + config["loss"]["beta"] * K.sum(kl_loss)# - K.sqrt(K.sum(K.square(z)))
    # total loss
    clvae_loss = loss

    clvae.add_loss(clvae_loss)
    clvae.compile(optimizer=config["network"]["optimizer"])
    
    return clvae, clvae_encoder, clvae_decoder

def build_vae(config):
    network = config["network"]
    latent_dim = network["latent_dim"]

    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    layers = network['intermediate_dims']
    encoder_inputs = keras.Input(shape=(network["img_dim"], network["img_dim"], config["network"]["color_dim"]))
    h = encoder_inputs
    for layer in layers:
        h = Conv2D(layer[0], layer[1], activation="relu", strides=layer[2], padding="same")(h)
    x = Flatten()(h)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, z_mean, name="encoder")
    #encoder.summary()

    reverse_layers = []
    internal_size = network["img_dim"]
    for layer in layers:
        reverse_layers = [Conv2DTranspose(layer[0], layer[1], activation="relu", strides=layer[2], padding="same")] + reverse_layers
        internal_size /= layer[2]
    internal_size = int(internal_size)

    decoder = Sequential(
        [Dense(internal_size * internal_size * layers[-1][0], activation="relu")] + 
        [Reshape((internal_size, internal_size, layers[-1][0]))] + 
        reverse_layers + 
        [Conv2DTranspose(config["network"]["color_dim"], 3, activation="sigmoid", padding="same")],
        name='Decoder'
    )
    x_recon = decoder(z)
    #decoder.summary()

    vae = Model(encoder_inputs, x_recon)


    # reconstruction loss
    #xent_loss = 0.5 * K.mean((encoder_inputs - x_recon)**2, 0)
    #kl_var_loss = - 0.5 * K.mean(z_log_var)
    #loss = xent_loss + kl_var_loss

    # Reconstruction loss
    Reconstruction_loss = network['original_dim']*keras.metrics.binary_crossentropy( K.flatten(encoder_inputs), K.flatten(x_recon))
 
    # KL divergence
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    loss =  Reconstruction_loss + kl_loss

    vae.add_loss(loss)
    vae.compile(optimizer=keras.optimizers.Adam())

    return vae, encoder, decoder

def build_cnn(config):
    model = Sequential()

    network = config['network']
    layers = network['intermediate_dims']
    # Encoder
    x = keras.Input(shape=(network["img_dim"], network["img_dim"], config["network"]["color_dim"]))
    h = x
    for layer in layers:
        h = Conv2D(layer[0], layer[1], activation="relu", strides=layer[2], padding="same")(h)
    h = Flatten()(h)
    h = Dense(16, activation="relu")(h)
    output = Dense(9)(h)

    model = Model(x, output)
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model




def calc_distribution(config, z_train, y_train):
    mu = []
    sigma = []
    labels = []
    for i in range(config["num_classes"]):
        samples = z_train[y_train == i, :]
        mu.append(np.mean(samples, axis=0))
        sigma.append(np.cov(samples, rowvar=0))
        index = i if i < config["anomalous_digit"] else i+1
        labels.append(str(index))
    mu = np.array(mu)

    max_pdf = [multivariate_normal.pdf(mu[k, :], mean=mu[k, :], cov=sigma[k]) for k in range(config["num_classes"])]


    return mu, sigma, labels, max_pdf

def get_predictions(thresh, config, z_test, mu, sigma, max_pdf):
    if thresh == 1: thresh = 2

    conf = np.ones(z_test.shape[0]) * thresh
    classes = np.ones(z_test.shape[0]) * config["anomalous_digit"]
    for k in range(config["num_classes"]):
        val = multivariate_normal.pdf(z_test, mean=mu[k, :], cov=sigma[k])/max_pdf[k]
        classes[conf <= val] = k + (k >= config["anomalous_digit"])
        conf[conf <= val] = val[conf <= val]

    return classes

def get_confidences(config, z_test, mu, sigma, max_pdf):

    conf = np.zeros((z_test.shape[0], 10))
    for k in range(config["num_classes"]):
        cl = k + (k >= config["anomalous_digit"])
        if False:
            for i in range(z_test.shape[0]):
                x = (z_test[i, :] - mu[k, :])

                conf[i, cl] = np.matmul(x, np.matmul(np.linalg.inv(sigma[k]), np.transpose(x)))
        else:
            val = multivariate_normal.pdf(z_test, mean=mu[k, :], cov=sigma[k])/max_pdf[k]
            conf[:, cl] = val

    conf[:, config["anomalous_digit"]] = 1-np.max(conf, axis=1)

    return conf

def get_confusion(thresh, config, z_test, y_test, mu, sigma, max_pdf):
    if thresh == 1: thresh = 2
    confusion = np.zeros((10, 10))
    counts = np.zeros((10, 10))
    for i in range(10):
        z_class = z_test[y_test == i, :]
        N = z_class.shape[0]
        conf = np.ones(N) * thresh
        classes = np.ones(N) * -1
        for k in range(config["num_classes"]):
            val = multivariate_normal.pdf(z_class, mean=mu[k, :], cov=sigma[k])/max_pdf[k]
            classes[conf <= val] = k
            conf[conf <= val] = val[conf <= val]
        for k in range(config["num_classes"]):
            cl = k + (k >= config["anomalous_digit"])
            counts[i, cl] = np.sum(classes == k)
            confusion[i, cl] = np.sum(classes == k)/N
        counts[i, config["anomalous_digit"]] = np.sum(classes == -1)
        confusion[i, config["anomalous_digit"]] = np.sum(classes == -1)/N
    return counts, confusion

def multivariate_gaussian(ax, threshold, mu, Sigma, z_train, max_pdf):
    """Return the multivariate Gaussian distribution on array pos."""
    x_min, x_max = min(z_train[:,0]), max(z_train[:,0])
    y_min, y_max = min(z_train[:,1]), max(z_train[:,1])

    N = 40
    X = np.linspace(x_min, x_max, N)
    Y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(X, Y)

    Z = [[
        multivariate_normal.pdf([X[i,j], Y[i,j]], mean=mu, cov=Sigma)/max_pdf for j in range(N)
    ] for i in range(N)]


    ax.contour(X,Y,Z,[threshold], colors=['black'])
