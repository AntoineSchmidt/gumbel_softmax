import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from manage import startSession, finishSession
from sae import StateAutoEncoder
from peg_sim import PegSimulator

def setting(config):
    network = [[config['l1_fc'], config['l1_fs']]]
    #network += [[config['l2_fc'], config['l2_fs']]]
    network += [config['l3'], config['l4'], config['l5']]
    bottleneck = config['latent']
    return network, bottleneck

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

    def compute(self, config, budget, **kwargs):
        startSession(memory=0.5)

        self.setting += 1
        network, bottleneck = setting(config)

        model = StateAutoEncoder(network_shape=network, latent_shape=(bottleneck, 2), domain=True)
        model.setWeight(2.0, 1.0, 0.0)

        epochs = int(budget)

        _, loss_val = model.train(self.data, epochs=epochs)
        model.finish()
        finishSession()

        print(self.setting, network, bottleneck, loss_val[-1])
        return ({'loss': loss_val[-1], 'info': {}})

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        filter_count = [4]#[4, 8, 16]
        filter_size = [7]#[3, 5, 7, 9]

        l1_fc = CSH.CategoricalHyperparameter('l1_fc', filter_count)
        l1_fs = CSH.CategoricalHyperparameter('l1_fs', filter_size)
        config_space.add_hyperparameters([l1_fc, l1_fs])

        #l2_fc = CSH.CategoricalHyperparameter('l2_fc', filter_count)
        #l2_fs = CSH.CategoricalHyperparameter('l2_fs', filter_size)
        #config_space.add_hyperparameters([l2_fc, l2_fs])

        l3 = CSH.UniformIntegerHyperparameter('l3', lower=20, upper=500)
        l4 = CSH.UniformIntegerHyperparameter('l4', lower=20, upper=500)
        l5 = CSH.UniformIntegerHyperparameter('l5', lower=10, upper=300)
        config_space.add_hyperparameters([l3, l4, l5])

        latent = CSH.UniformIntegerHyperparameter('latent', lower=20, upper=100)
        #latent = CSH.CategoricalHyperparameter('latent', [33])
        config_space.add_hyperparameters([latent])

        return config_space


BUDGET = 10
ITERATIONS = 100

NS = hpns.NameServer(run_id='Search', host='127.0.0.1', port=None)
NS.start()
WK = MyWorker(nameserver='127.0.0.1', run_id='Search')
WK.run(background=True)

BS = BOHB(configspace=WK.get_configspace(), run_id='Search', nameserver='127.0.0.1', min_budget=BUDGET, max_budget=BUDGET)
BSP = BS.run(n_iterations=ITERATIONS)

BS.shutdown(shutdown_workers=True)
NS.shutdown()

print('Best found configuration:', setting(BSP.get_id2config_mapping()[BSP.get_incumbent_id()]['config']))