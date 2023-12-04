import random
import numpy as np

class HashMemory:
    def __init__(self):
        self.hashed_boards = []
        
    def insert(self, board):
        board_hash = hash(board.tostring())
        if board_hash not in self.hashed_boards:
            self.hashed_boards.append(board_hash)
            return True
        return False
    

class PegSimulator:
    SIZE = 7
    SIZE_CELL = 8
    SIZE_CORNER = 2

    VALUE_PEG = 1
    VALUE_EMPTY = 0#.5
    VALUE_CORNER = 0

    def __init__(self):
        self.reset()

    # Resets internal state to the starting setup
    def reset(self):
        initial, _ = PegSimulator.getBoardFull()
        initial[int(PegSimulator.SIZE/2), int(PegSimulator.SIZE/2)] = PegSimulator.VALUE_EMPTY
        self.state = initial

    # Returns completely full board
    def getBoardFull():
        board = np.ones((PegSimulator.SIZE, PegSimulator.SIZE)) * PegSimulator.VALUE_CORNER
        board[PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER, :] = PegSimulator.VALUE_PEG
        board[:, PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER] = PegSimulator.VALUE_PEG
        return board, PegSimulator.__create(board)

    # Returns completely empty board
    def getBoardEmpty():
        board = np.ones((PegSimulator.SIZE, PegSimulator.SIZE)) * PegSimulator.VALUE_CORNER
        board[PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER, :] = PegSimulator.VALUE_EMPTY
        board[:, PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER] = PegSimulator.VALUE_EMPTY
        return board, PegSimulator.__create(board)

    # Sample Game Sequence
    def sampleSequence(self, count, one_game=False, unique=False):
        size = PegSimulator.SIZE * PegSimulator.SIZE_CELL
        stack = np.zeros((count, size, size, 1))

        index = 0
        boards = HashMemory()
        while index < count:
            if not unique or boards.insert(self.state):
                stack[index, :, :, :] = PegSimulator.__create(self.state)
                index += 1
            if(not self.__step()):
                #print('No further moves possible', (self.state == PegSimulator.VALUE_PEG).sum())
                self.reset()
                if one_game:
                    return stack[:index]
        return stack

    def __step(self):
        possibleChoices = []
        for x in range(PegSimulator.SIZE):
            for y in range(PegSimulator.SIZE):
                # Skip Corners
                if x < 2 or x >= (PegSimulator.SIZE - PegSimulator.SIZE_CORNER):
                    if y < 2 or y >= (PegSimulator.SIZE - PegSimulator.SIZE_CORNER):
                        continue
                # Possible Target Positions
                if self.state[x, y] == PegSimulator.VALUE_EMPTY: 
                    possibleChoices.append([x, y])
        random.shuffle(possibleChoices)

        orientation = np.arange(4)
        random.shuffle(orientation)
        for x, y in possibleChoices:
            for o in orientation:
                if o == 0:
                    if x > 1 and self.state[x - 1, y] == PegSimulator.VALUE_PEG and self.state[x - 2, y] == PegSimulator.VALUE_PEG:
                        self.state[x, y] = PegSimulator.VALUE_PEG
                        self.state[x - 1, y] = PegSimulator.VALUE_EMPTY
                        self.state[x - 2, y] = PegSimulator.VALUE_EMPTY
                        return True
                if o == 1:
                    if x < PegSimulator.SIZE - 2 and self.state[x + 1, y] == PegSimulator.VALUE_PEG and self.state[x + 2, y] == PegSimulator.VALUE_PEG:
                        self.state[x, y] = PegSimulator.VALUE_PEG
                        self.state[x + 1, y] = PegSimulator.VALUE_EMPTY
                        self.state[x + 2, y] = PegSimulator.VALUE_EMPTY
                        return True
                if o == 2:
                    if y > 1 and self.state[x, y - 1] == PegSimulator.VALUE_PEG and self.state[x, y - 2] == PegSimulator.VALUE_PEG:
                        self.state[x, y] = PegSimulator.VALUE_PEG
                        self.state[x, y - 1] = PegSimulator.VALUE_EMPTY
                        self.state[x, y - 2] = PegSimulator.VALUE_EMPTY
                        return True
                if o == 3:
                    if y < PegSimulator.SIZE - 2 and self.state[x, y + 1] == PegSimulator.VALUE_PEG and self.state[x, y + 2] == PegSimulator.VALUE_PEG:
                        self.state[x, y] = PegSimulator.VALUE_PEG
                        self.state[x, y + 1] = PegSimulator.VALUE_EMPTY
                        self.state[x, y + 2] = PegSimulator.VALUE_EMPTY
                        return True
        return False

    # Sample Random Boards (includes 'illegal' setups)
    def sampleRandom(count, unique=True):
        size = PegSimulator.SIZE * PegSimulator.SIZE_CELL
        stack = np.zeros((count, size, size, 1))

        index = 0
        boards = HashMemory()
        while index < count:
            board = np.ones((PegSimulator.SIZE, PegSimulator.SIZE)) * PegSimulator.VALUE_CORNER
            board_values = PegSimulator.__randomConstellation((PegSimulator.SIZE - 2 * PegSimulator.SIZE_CORNER, PegSimulator.SIZE), np.random.randint(1, 4))
            board[PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER, :] = board_values
            board_values = PegSimulator.__randomConstellation((PegSimulator.SIZE, PegSimulator.SIZE - 2 * PegSimulator.SIZE_CORNER), np.random.randint(1, 4))
            board[:, PegSimulator.SIZE_CORNER:PegSimulator.SIZE - PegSimulator.SIZE_CORNER] = board_values

            if not unique or boards.insert(board):
                stack[index, :, :, :] = PegSimulator.__create(board)
                index += 1
        return stack

    def __randomConstellation(size, sparsity=2):
        constellation = np.ones(size)
        for i in range(sparsity):
            constellation *= np.random.randint(2, size=size)
        return constellation * (PegSimulator.VALUE_PEG - PegSimulator.VALUE_EMPTY) + PegSimulator.VALUE_EMPTY

    # Samples Single Pegs
    def sampleControlled():
        size = PegSimulator.SIZE * PegSimulator.SIZE_CELL
        stack = np.zeros((33, size, size, 1))

        stack_index = 0
        for x in range(PegSimulator.SIZE):
            for y in range(PegSimulator.SIZE):
                # Skip Corners
                if x < 2 or x >= (PegSimulator.SIZE - PegSimulator.SIZE_CORNER):
                    if y < 2 or y >= (PegSimulator.SIZE - PegSimulator.SIZE_CORNER):
                        continue

                board, _ = PegSimulator.getBoardEmpty()
                board[x, y] = PegSimulator.VALUE_PEG
                stack[stack_index, :, :, :] = PegSimulator.__create(board)
                stack_index += 1
        return stack

    # Blows up board to final size
    def __create(board):
        image = np.kron(board, np.ones((PegSimulator.SIZE_CELL, PegSimulator.SIZE_CELL)))
        return np.expand_dims(image, axis=2)

    # Converts Board into the possible perfect encoding with 33bits
    def perfectEncoding(data):
        size = PegSimulator.SIZE * PegSimulator.SIZE_CELL
        assert(data.shape[1:] == (size, size, 1))
        encoded = data[:, ::PegSimulator.SIZE_CELL, ::PegSimulator.SIZE_CELL, 0]
        encoded = np.reshape(encoded, (encoded.shape[0], PegSimulator.SIZE**2))
        indizes = np.array([0, 1, 5, 6, 7, 8, 12, 13])
        indizes = np.concatenate((indizes, PegSimulator.SIZE**2 - indizes - 1))
        return np.delete(encoded, indizes, axis=1)

    # Reconstructs Board given a perfect encoding with 33bits
    # Keeps original values
    def perfectDecoding(data):
        assert(data.shape[1:] == (33,))
        indizes = np.array([0] * 2 + [3] * 4 + [6] * 2 + [27] * 2 + [30] * 4)
        decoded = np.insert(data, indizes, np.zeros(indizes.shape), axis=1)
        decoded = np.append(decoded, np.zeros((data.shape[0], 2)), axis=1)
        decoded = np.reshape(decoded, (-1, PegSimulator.SIZE, PegSimulator.SIZE))
        return np.kron(decoded, np.ones((1, PegSimulator.SIZE_CELL, PegSimulator.SIZE_CELL)))[ :, :, :, np.newaxis]


# Peg Simulator Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    start = PegSimulator().sampleSequence(1)
    plt.imshow(start[0, :, :, 0], cmap=cm.gray, vmin=0, vmax=1)
    plt.savefig('image/peg.png', dpi=100)

    print('Sample Random')
    stack = PegSimulator.sampleRandom(5)
    stack_encoded = PegSimulator.perfectEncoding(stack)
    assert (stack == PegSimulator.perfectDecoding(stack_encoded)).all()
    for i in range(np.shape(stack)[0]):
        plt.imshow(stack[i][:, :, 0], cmap=cm.gray, vmin=0, vmax=1)
        plt.show()
        plt.imshow(stack_encoded[i, :, np.newaxis], cmap=cm.gray, vmin=0, vmax=1)
        plt.show()

    print('Sample Controlled')
    stack = PegSimulator.sampleControlled()
    for i in range(np.shape(stack)[0]):
        plt.imshow(stack[i][:, :, 0], cmap=cm.gray, vmin=0, vmax=1)
        plt.show()

    print('Simulated')
    stack = PegSimulator().sampleSequence(10)
    for i in range(np.shape(stack)[0] - 1):
        image = np.concatenate((stack[i], stack[i + 1]))[:, :, 0]
        plt.imshow(image, cmap=cm.gray, vmin=0, vmax=1)
        plt.show()