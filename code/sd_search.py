import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from manage import startSession, finishSession
from sd import StateDiscriminator

def setting(config):
    network = []
    for i in range(config['num_layers']):
        network.append(config['l{}'.format(i)])
    return network

class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setting = 0
        loaded = np.load('data/sd_data.npz')
        self.sd_in, self.sd_out = (loaded[i] for i in loaded.files)

    def compute(self, config, budget, **kwargs):
        startSession(memory=0.5)

        self.setting += 1
        network = setting(config)

        model = StateDiscriminator(data_shape=(33,), network_shape=network)

        _, loss_val = model.train(data_in=self.sd_in, data_out=self.sd_out, epochs=int(budget))
        model.finish()
        finishSession()

        print(self.setting, network, loss_val[-1])
        return ({'loss': loss_val[-1], 'info': {}})

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        l0 = CSH.UniformIntegerHyperparameter('l0', lower=10, upper=200, log=True)
        l1 = CSH.UniformIntegerHyperparameter('l1', lower=10, upper=200, log=True)
        l2 = CSH.UniformIntegerHyperparameter('l2', lower=10, upper=200, log=True)
        config_space.add_hyperparameters([l0, l1, l2])

        num_layers =  CSH.UniformIntegerHyperparameter('num_layers', lower=1, upper=3)
        config_space.add_hyperparameters([num_layers])
        
        cond = CS.GreaterThanCondition(l1, num_layers, 1)
        config_space.add_condition(cond)

        cond = CS.GreaterThanCondition(l2, num_layers, 2)
        config_space.add_condition(cond)

        return config_space


BUDGET = 20
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