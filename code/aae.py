import numpy as np

from keras import backend as K
from keras.models import Model
from keras.activations import softmax, sigmoid
from keras.objectives import mean_squared_error, binary_crossentropy
from keras.layers import Input, Dense, concatenate, Lambda, Reshape, Softmax

class ActionAutoEncoder:
    def __init__(self, data_shape=(33,), network_shape=[40, 80], latent_shape=(1, 33 * 4)):
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

        self.name = 'aae-' + str(network_shape) + str(latent_shape)

        # Build Network
        self.__net(data_shape, network_shape)

        # Build Models
        self.autoencoder = Model([self.data_in, self.data_co], self.data_out)
        self.encoder = K.function([self.data_in, self.data_co, K.learning_phase()], [self.latent_out])
        self.decoder = K.function([self.data_in, self.autoencoder.get_layer(name='decoder-0').input, K.learning_phase()], [self.autoencoder.get_layer(index=-1).output])

        loss = lambda x, x_hat: self.loss_func(x, x_hat) * self.loss_weight_func + self.__lossGumbel() * self.loss_weight_gumb + self.__lossZero() * self.loss_weight_zero
        self.autoencoder.compile(optimizer='adam', loss=loss, metrics=[self.loss_func])

    def setWeight(self, weight_func=5.0, weight_gumb=1.0, weight_zero=1.0):
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

    def __net(self, data_shape, network_shape):
        print('Building Network')

        # Placeholder
        self.data_in = Input(shape=data_shape)
        self.data_co = Input(shape=data_shape) #desired output

        # Encoder
        layer = self.data_co
        for i in network_shape:
            layer = Dense(i, activation='relu')(layer)
            layer = concatenate([layer, self.data_in])

        # Latent
        self.latent = Dense(self.latent_units)(layer)

        self.latent_out = Reshape(self.latent_shape)(self.latent)
        self.latent_out = Softmax(name='latent_output')(self.latent_out)

        self.latent_softmax = Lambda(self.__gumbelSample, output_shape=(self.latent_units,))(self.latent)

        # Decoder
        count = 0
        layer = self.latent_softmax
        for i in network_shape[::-1]:
            layer = Dense(i, activation='relu', name='decoder-{}'.format(count))(layer)
            layer = concatenate([layer, self.data_in])
            count += 1

        # Output
        self.data_out = Dense(data_shape[0], activation='sigmoid')(layer)

    def train(self, data_in, data_out, epochs=None):
        loss = []
        loss_val = []

        if epochs is None:
            epochs = self.convergeEpochs()

        for e in range(epochs):
            decay = np.exp(- self.tau_decay * e)
            K.set_value(self.tau, np.max([K.get_value(self.tau) * decay, self.tau_min]))

            print('Epoch:', e, '| Tau:', K.get_value(self.tau))
            history = self.autoencoder.fit([data_in, data_out], data_out, validation_split=0.1, batch_size=self.batch_size)

            loss += history.history['loss']
            loss_val += history.history['val_loss']
        return loss, loss_val

    def predict(self, data, data_co):
        return self.autoencoder.predict([data, data_co], batch_size=self.batch_size)

    def encode(self, data, data_co):
        return self.encoder([data, data_co, 0])[0]

    def decode(self, data, latent):
        if latent.shape[1:] == self.latent_shape:
            latent = np.reshape(latent, (-1, self.latent_units))
        return self.decoder([data, latent, 0])[0]

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

    # Generate Debug Data (latent expected to learn some logic operation)
    data_in = np.random.randint(2, size=(2000, 10))
    data_out = np.random.randint(2, size=(2000, 10))

    # Load Model
    model = ActionAutoEncoder(data_shape=(10,), latent_shape=(10, 2), network_shape=[40, 20, 10])
    model.autoencoder.summary()
    model.setWeight(5.0, 1.0)
    model.tau_min = 1e-5

    # Run Training
    try:
        for e in [model.convergeEpochs()//3] * 3:
            loss, loss_val = model.train(data_in, data_out, e)
            print(loss[-1], loss_val[-1])

            latent_softmax = model.encode(data_in, data_out)
            data_predicted = model.predict(data_in, data_out)

            for i in range(np.min([np.shape(data_in)[0], 3])):
                # Input, Latent Softmax, Output, Output Prediction
                image = np.concatenate((data_in[i, :, np.newaxis], np.reshape(latent_softmax[i], model.latent_shape)[:, 0, np.newaxis], data_out[i, :, np.newaxis], data_predicted[i, :, np.newaxis]), axis=1)
                plt.imshow(image, cmap=cm.gray, vmin=0, vmax=1)
                plt.show()
    except KeyboardInterrupt:
        pass
    model.finish()