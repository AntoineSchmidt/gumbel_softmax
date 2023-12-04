import math
import numpy as np

# Binary Latent Vector Only
def buildLatentBinary(values):
    assert(len(values.shape) == 2)
    shape_out = (values.shape[0], 2 * values.shape[1])
    latent = values[:, :, np.newaxis]
    latent = np.concatenate((latent, 1 - latent), axis=2)
    return np.reshape(latent, shape_out)

# Binary Latent Vector Only
def roundLatentBinary(latent):
    latent_round = np.ones(latent.shape) * 0.5
    latent_round[latent > 0.5] = 1
    latent_round[latent < 0.5] = 0
    return latent_round

# Round Latent Vector
def roundLatent(latent):
    assert(len(latent.shape) == 3)
    n = np.zeros(latent.shape)
    for i in range(latent.shape[0]):
        b = latent[i].argmax(axis=1)
        n[i, np.arange(latent.shape[1]), b] = 1
    return n

# Generate all possible binary actions
def allActionsBinary(encoding):
    assert(len(encoding) == 2 and encoding[1] == 2)
    total = int(math.pow(encoding[1], encoding[0]))
    actions = np.zeros((total,) + encoding)

    action = np.zeros(encoding)
    for i in range(total):
        # binary counting
        a = 0
        action[a, 0] = 1 - action[a, 0]
        while action[a, 0] == 0:
            a += 1
            if a < encoding[0]:
                action[a, 0] = 1 - action[a, 0]
            else:
                break

        action[:, 1] = 1 - action[:, 0]
        actions[i] = action

    return actions

# Generate all possible actions
def allActions(encoding):
    assert(len(encoding) == 2)
    total = int(math.pow(encoding[1], encoding[0]))
    actions = np.zeros((total,) + encoding)

    for i in range(total):
        action = np.zeros(encoding)
        action[:, 0] = 1
        indexes = []
        _allActionsIndexer(indexes, i, encoding[1])
        for a in range(len(indexes)):
            action[a, 0] = 0
            action[a, indexes[a]] = 1

        actions[i] = action

    return actions

# Basically a 10 to any base converter
def _allActionsIndexer(indexes, decimal, base):
    indexes.append(decimal % base)
    div = decimal // base
    if(div is not 0):
        _allActionsIndexer(indexes, div, base)


# Test
if __name__ == "__main__":
    a = allActionsBinary((4, 2))
    for i in range(a.shape[0]):
        print(a[i, :, 0])

    a = allActions((4, 3))
    for i in range(a.shape[0]):
        print(a[i, 2])