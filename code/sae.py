import numpy as np

from keras import backend as K
from keras.models import Model
from keras.activations import softmax, sigmoid
from keras.objectives import mean_squared_error, binary_crossentropy, categorical_crossentropy
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Deconv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Lambda, Softmax

class StateAutoEncoder:
    def __init__(self, network_shape, data_shape=(56, 56, 1), latent_shape=(33, 2), domain=False, use_latent=False):
        self.tau = K.variable(5.0)
        self.tau_min = 0.5
        self.tau_decay = 5e-4

        self.batch_size = 64
        self.loss_func = binary_crossentropy
        self.loss_weight_func = K.variable(1.0)
        self.loss_weight_gumb = K.variable(1.0)
        self.loss_weight_zero = K.variable(1.0)

        self.latent_shape = latent_shape
        self.latent_units = latent_shape[0] * latent_shape[1]

        self.name = 'sae-' + str(network_shape) + str(latent_shape)

        # Build Network
        self.__net(data_shape, network_shape, domain)

        # Build Models
        self.autoencoder = Model(self.data_in, [self.data_out, self.latent_out]) # Added the possibility to supply wanted encoding
        self.encoder = K.function([self.data_in, K.learning_phase()], [self.latent_out])
        self.decoder = K.function([self.autoencoder.get_layer(name='decoder-0').input, K.learning_phase()], [self.autoencoder.get_layer(name='final_output').output])

        loss = lambda x, x_hat: self.loss_func(x, x_hat) * self.loss_weight_func + self.__lossGumbel() * self.loss_weight_gumb + self.__lossZero() * self.loss_weight_zero
        loss_supervised = lambda x, x_hat: categorical_crossentropy(x, x_hat)

        losses = {
            "final_output": loss,
            "latent_output": loss_supervised,
        }
        lossWeights = {
            "final_output": 1.0,
            "latent_output": 1.0 if use_latent else 0.0,
        }
        self.autoencoder.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=[self.loss_func])

    def setWeight(self, weight_func=1.0, weight_gumb=1.0, weight_zero=0.0):
        K.set_value(self.loss_weight_func, weight_func)
        K.set_value(self.loss_weight_gumb, weight_gumb)
        K.set_value(self.loss_weight_zero, weight_zero)

    # https://github.com/guicho271828/latplan/blob/master/util/layers.py
    def __gumbelSample(self, latent):
        U = K.in_train_phase(K.log(-K.log(K.random_uniform(K.shape(latent)) + 1e-20) + 1e-20), 0.0)
        y = latent - U
        y = softmax(K.reshape(y, (-1,) + self.latent_shape) / self.tau)
        return K.reshape(y, (-1, self.latent_units))

    # https://github.com/guicho271828/latplan/blob/master/util/layers.py
    def __lossGumbel(self):
        log_q = K.log(self.latent_out + 1e-20)
        loss = -K.mean(self.latent_out * log_q, axis=(1, 2))
        return K.reshape(loss, (-1, 1, 1)) / self.tau

    # https://arxiv.org/abs/1903.11277
    def __lossZero(self):
        loss = K.mean(self.latent_out, axis=1)
        loss = K.mean(loss[:, 1:], axis=1)
        return K.reshape(loss, (-1, 1, 1)) / self.tau

    def convergeEpochs(self):
        epoch = 0
        current = K.get_value(self.tau)
        while True:
            decay = np.exp(- self.tau_decay * epoch)
            current = np.max([current * decay, self.tau_min])
            epoch += 1
            if current == self.tau_min:
                return epoch

    def __net(self, data_shape, network_shape, domain):
        # Placeholder
        self.data_in = Input(shape=data_shape)

        # Noise
        layer = GaussianNoise(0.1)(self.data_in)
        #layer = self.data_in

        # Encoder
        conv = True
        conv_count = 0
        for i in network_shape:
            if type(i) is list and conv:
                layer = Conv2D(i[0], i[1], activation='relu', padding='same')(layer)
                layer = MaxPooling2D((2, 2), padding='same')(layer)
                conv_count += 1
            elif type(i) is int:
                if conv:
                    layer_shape = K.int_shape(layer)
                    layer = Flatten()(layer)
                    conv = False
                layer = Dense(i, activation='relu')(layer)
            else:
                print('network_shape error')

        # Latent
        self.latent = Dense(self.latent_units)(layer)

        self.latent_out = Reshape(self.latent_shape)(self.latent)
        self.latent_out = Softmax(name='latent_output')(self.latent_out)

        self.latent_softmax = Lambda(self.__gumbelSample, output_shape=(self.latent_units,))(self.latent)

        # Decoder
        count = 0
        layer = self.latent_softmax
        for i in network_shape[::-1]:
            if type(i) is int and domain:
                layer = Dense(i, activation='relu', name='decoder-{}'.format(count))(layer)
                count += 1
            else:
                break

        if domain:
            print('Domain Knowledge Network')
            self.name += '-DK'

            layer = Dense(7 * 7, activation='sigmoid')(layer)
            layer = Reshape((7, 7, 1))(layer)

            # Output
            self.data_out = UpSampling2D((8, 8), name='final_output')(layer)
        else:
            print('Full Network')

            layer = Dense(layer_shape[1] * layer_shape[2], name='decoder-0')(layer)
            layer = Reshape((layer_shape[1], layer_shape[2], 1))(layer)
            layer = UpSampling2D((2 * conv_count, 2 * conv_count))(layer)

            # Output
            self.data_out = Conv2D(1, network_shape[0][1], activation='sigmoid', padding='same', name='final_output')(layer)

    def train(self, data, data_latent=None, epochs=None):
        loss = []
        loss_val = []

        if data_latent is None:
            data_latent = np.zeros((data.shape[0],) + self.latent_shape)
        if epochs is None:
            epochs = self.convergeEpochs()

        for e in range(epochs):
            decay = np.exp(- self.tau_decay * e)
            K.set_value(self.tau, np.max([K.get_value(self.tau) * decay, self.tau_min]))

            print('Epoch:', e, '| Tau:', K.get_value(self.tau))
            history = self.autoencoder.fit(data, [data, data_latent], validation_split=0.1, batch_size=self.batch_size)

            loss += history.history['loss']
            loss_val += history.history['val_loss']
        return loss, loss_val

    def predict(self, data):
        return self.autoencoder.predict(data, batch_size=self.batch_size)

    def encode(self, data):
        result = None
        if data.shape[0] >= 2 * self.batch_size: # Chunk up data in batches
            overflow = data.shape[0] % self.batch_size + self.batch_size
            for batch in np.split(data[:-overflow], self.batch_size):
                result_batch = self.encoder([batch, 0])[0]
                if result is None:
                    result = result_batch
                else:
                    result = np.concatenate((result, result_batch))
            result = np.concatenate((result, self.encoder([data[-overflow:], 0])[0]))
        else:
            result = self.encoder([data, 0])[0]
        return result

    def decode(self, latent):
        if latent.shape[1:] == self.latent_shape: # Reshape latent input
            latent = np.reshape(latent, (-1, self.latent_units))

        result = None
        if latent.shape[0] >= 2 * self.batch_size: # Chunk up data in batches
            overflow = latent.shape[0] % self.batch_size + self.batch_size
            for batch in np.split(latent[:-overflow], self.batch_size):
                result_batch = self.decoder([batch, 0])[0]
                if result is None:
                    result = result_batch
                else:
                    result = np.concatenate((result, result_batch))
            result = np.concatenate((result, self.decoder([latent[-overflow:], 0])[0]))
        else:
            result = self.decoder([latent, 0])[0]
        return result

    def load(self, converged=True):
        try:
            self.autoencoder.load_weights('model/' + self.name + '.h5')
            if converged:
                K.set_value(self.tau, self.tau_min)
            print("Loaded Model")
            return True
        except OSError:
            print('Failed loading model')
        except Exception as e:
            print('Failed loading model:', repr(e))
        return False

    def save(self):
        try:
            self.autoencoder.save_weights('model/' + self.name + '.h5')
            print("Saved Model")
            return True
        except OSError:
            print('Failed saving model')
        except Exception as e:
            print('Failed saving model:', repr(e))
        return False

    def finish(self):
        del self.encoder
        del self.decoder
        del self.autoencoder


# Model Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from peg_sim import PegSimulator

    # Load Data
    data = PegSimulator.sampleRandom(2000)
    data_show = PegSimulator.sampleRandom(2)

    # Load Model
    model = StateAutoEncoder([[4, 7], 275])
    model.setWeight()
    model.autoencoder.summary()

    # Run Training
    try:
        for e in [10] * 3:
            loss, loss_val = model.train(data, epochs=e)
            print(loss[-1], loss_val[-1])

            latent = model.encode(data_show)
            data_out, _ = model.predict(data_show)

            for i in range(np.shape(data_show)[0]):
                image = np.concatenate((data_show[i, :, :, 0], data_out[i, :, :, 0]))
                plt.imshow(image, cmap=cm.gray, vmin=0, vmax=1)
                plt.show()
                plt.imshow(np.reshape(latent[i], model.latent_shape), cmap=cm.gray, vmin=0, vmax=1)
                plt.show()
    except KeyboardInterrupt:
        pass
    model.finish()