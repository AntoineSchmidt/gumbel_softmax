import numpy as np

from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers import Input, Dense, concatenate

class ActionDiscriminator:
    def __init__(self, data_shape=(33,), network_shape=[80, 20]):
        self.batch_size = 64
        self.loss_func = binary_crossentropy

        self.name = 'ad-' + str(network_shape) + str(data_shape)

        # Build Network
        self.__net(data_shape, network_shape)

        # Build Models
        self.model = Model([self.data_s, self.data_t], self.prediction)
        self.model.compile(optimizer='adam', loss=self.loss_func)

    def __net(self, data_shape, network_shape):
        print('Building Network')

        # Placeholder
        self.data_s = Input(shape=data_shape)
        self.data_t = Input(shape=data_shape)

        layer = concatenate([self.data_s, self.data_t], axis=-1)

        # Network
        for i in network_shape:
            layer = Dense(i, activation='relu')(layer)

        # Output
        self.prediction = Dense(1, activation='sigmoid')(layer)

    def train(self, data_s, data_t, data_out, epochs=200):
        history = self.model.fit([data_s, data_t], data_out, epochs=epochs, validation_split=0.1, batch_size=self.batch_size)
        return history.history['loss'], history.history['val_loss']

    def predict(self, data_s, data_t):
        return self.model.predict([data_s, data_t], batch_size=self.batch_size)

    def load(self):
        try:
            self.model.load_weights('model/' + self.name + '.h5')
            print("Loaded Model")
            return True
        except OSError:
            print('Failed loading model')
        except Exception as e:
            print('Failed loading model:', repr(e))
        return False

    def save(self):
        try:
            self.model.save_weights('model/' + self.name + '.h5')
            print("Saved Model")
            return True
        except OSError:
            print('Failed saving model')
        except Exception as e:
            print('Failed saving model:', repr(e))
        return False

    def finish(self):
        del self.model

# Model Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Generate Debug Data
    data_s = np.random.randint(2, size=(2000, 10))
    data_t = np.random.randint(2, size=(2000, 10))
    data_out = np.random.randint(2, size=(2000, 1))

    # Load Model
    model = ActionDiscriminator(data_shape=(10,), network_shape=[40, 20, 10])
    model.model.summary()

    # Run Training
    try:
        loss, loss_val = model.train(data_s, data_t, data_out)
        plt.plot(np.arange(200), loss)
        plt.plot(np.arange(200), loss_val)
        plt.show()
    except KeyboardInterrupt:
        pass
    model.finish()