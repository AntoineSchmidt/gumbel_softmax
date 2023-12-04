import logging
logging.basicConfig(level=logging.ERROR)

import math
import numpy as np
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from keras import backend as K
from keras.objectives import binary_crossentropy

from manage import startSession, finishSession
from sae import StateAutoEncoder
from helper import roundLatentBinary
from peg_sim import PegSimulator


class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setting = 0
        try:
            self.data = np.load('data/sae_data.npy')
        except:
            self.data = PegSimulator.sampleRandom(int(1e5))
            #self.data = PegSimulator().sampleSequence(int(1e5), unique=True)
            np.save('data/sae_data.npy', self.data)

        # Final Comparison Data
        test_split = math.ceil(self.data.shape[0] * 0.1) # 0.1 set in train function
        self.data_compare = self.data[-test_split:]

    def compute(self, config, budget, **kwargs):
        self.setting += 1
        startSession(memory=0.5)

        epochs = int(budget)

        LATENT = (33, 2)
        NETWORK = [[4, 7], 275, 175, 135]

        model = StateAutoEncoder(network_shape=NETWORK, latent_shape=LATENT, domain=True)

        weight_func = config['weight_func']
        weight_gumb = 1#config['weight_gumb']
        model.setWeight(weight_func, weight_gumb, 0)
        model.train(self.data, epochs=epochs)

        # Calculate Rounded Accuracy
        latent_softmax = model.encode(self.data_compare)
        latent_softmax_round = roundLatentBinary(latent_softmax)
        data_output = model.decode(latent_softmax_round)

        #result = K.get_value(K.mean(binary_crossentropy(self.data_compare, data_output)))
        result = np.mean(np.abs(self.data_compare - data_output))

        model.finish()
        finishSession()

        print(self.setting, weight_func, weight_gumb, result)
        return ({'loss': result, 'info': {}})

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        weight_func = CSH.UniformFloatHyperparameter('weight_func', lower=1, upper=1e4)
        config_space.add_hyperparameters([weight_func])

        #weight_gumb = CSH.UniformFloatHyperparameter('weight_gumb', lower=1, upper=1e2)
        #config_space.add_hyperparameters([weight_gumb])

        return config_space


BUDGET = 20
ITERATIONS = 50

NS = hpns.NameServer(run_id='Search', host='127.0.0.1', port=None)
NS.start()
WK = MyWorker(nameserver='127.0.0.1', run_id='Search')
WK.run(background=True)

BS = BOHB(configspace=WK.get_configspace(), run_id='Search', nameserver='127.0.0.1', min_budget=BUDGET, max_budget=BUDGET)
BSP = BS.run(n_iterations=ITERATIONS)

BS.shutdown(shutdown_workers=True)
NS.shutdown()

print('Best found configuration:', BSP.get_id2config_mapping()[BSP.get_incumbent_id()]['config'])