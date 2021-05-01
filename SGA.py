# SGA
import pylab as pl
import numpy as np
import random


class sga:

    def __init__(self, stringLength, popSize, nGens, pm, pc):
        # stringLength: int, popSize: int, nGens: int,
        # prob. mutation pm: float; prob. crossover pc: float
        fid = open("results.txt", "w")        # open, initialize output file
        self.fid = fid
        self.stringLength = stringLength   # number of bits in a chromosome
        if np.mod(popSize, 2) == 0:           # popSize must be even
            self.popSize = popSize
        else:
            self.popSize = popSize + 1
        self.pm = pm                       # probability of mutation
        self.pc = pc                       # probability of crossover
        self.env = np.array([
            # 1: padding for sees environment
            # 2: got to third sequentially large pipe
            # 3: got to ? tile above a floating single brick
            # 4: Passed the first mini-pyramid ditch
            # 5: right before final pyramid
            # 6: end
            # 0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 10
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 11
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 13
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 14
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 15
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
             0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 16
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
             1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # 17
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # 18
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        ])
        self.initLocx = 17                  # ant's initial location x,y
        self.initLocy = 12
        self.initDir = 'R'                 # ant's initial orientation (URDL)
        self.cw = {'R': 'L', 'L': 'R'}   # rotate orientation clockwise
        self.ccw = {'R': 'L', 'L': 'R'}   # rotate orientation counter-cw
        self.legalActs = np.array(
            ['GJ', 'GF', 'GF4', 'GF5', 'GD', 'GFD'])    # legal mario actions
        self.nGens = nGens                 # max number of generations
        self.pop = np.random.rand(self.popSize, self.stringLength)
        # str len should be 9, 6 for visual field, 3 for moves. play with this.
        self.pop = np.where(self.pop < 0.5, 1, 0)  # create initial pop
        # fitness values for initial population
        fitness = self.fitFcn(self.pop)
        self.bestfit = fitness.max()       # fitness of (first) most fit chromosome
        self.bestloc = np.where(fitness == self.bestfit)[
            0][0]  # most fit chromosome locn
        self.bestchrome = self.pop[self.bestloc,
                                   :]              # most fit chromosome
        # array of max fitness vals each generation
        self.bestfitarray = np.zeros(self.nGens + 1)
        self.bestfitarray[0] = self.bestfit  # (+ 1 for init pop plus nGens)
        # array of mean fitness vals each generation
        self.meanfitarray = np.zeros(self.nGens + 1)
        self.meanfitarray[0] = fitness.mean()
        fid.write("popSize: {}  nGens: {}  pm: {}  pc: {}\n".format(
            popSize, nGens, pm, pc))
        fid.write("initial population, fitnesses: (up to 1st 100 chromosomes)\n")
        for c in range(
                min(100, popSize)):   # for each of first 100 chromosomes
            fid.write("  {}  {}\n".format(self.pop[c, :], fitness[c]))
        fid.write("Best initially:\n  {} at locn {}, fitness = {}\n".format(
            self.bestchrome, self.bestloc, self.bestfit))

    # conduct tournaments to select two offspring
    # play with this (fitness proportionate selection)
    def tournament(self, pop, fitness, popsize):  # fitness array, pop size
        # select first parent par1
        cand1 = np.random.randint(popsize)      # candidate 1, 1st tourn., int
        cand2 = cand1                           # candidate 2, 1st tourn., int
        while cand2 == cand1:                   # until cand2 differs
            cand2 = np.random.randint(popsize)  # identify a second candidate
        if fitness[cand1] > fitness[cand2]:     # if cand1 more fit than cand2
            par1 = cand1  # then first parent is cand1
        else:  # else first parent is cand2
            par1 = cand2
        # select second parent par2
        cand1 = np.random.randint(popsize)      # candidate 1, 2nd tourn., int
        cand2 = cand1                           # candidate 2, 2nd tourn., int
        while cand2 == cand1:                   # until cand2 differs
            cand2 = np.random.randint(popsize)  # identify a second candidate
        if fitness[cand1] > fitness[cand2]:     # if cand1 more fit than cand2
            par2 = cand1  # then 2nd parent par2 is cand1
        else:  # else 2nd parent par2 is cand2
            par2 = cand2
        return par1, par2
    # play with this
    # try 2 and random 1-2 xover?

    def xover(self, child1, child2):    # single point crossover
        # cut locn to right of position (hence subtract 1)
        cuts = random.randint(1, 2)
        if cuts == 1:
            locn = np.random.randint(0, self.stringLength - 1)
            tmp = np.copy(child1)       # save child1 copy, then do crossover
            child1[locn + 1:self.stringLength] = child2[locn + 1:self.stringLength]
            child2[locn + 1:self.stringLength] = tmp[locn + 1:self.stringLength]
            return child1, child2
        # cannot xover at the first bit
        locn1 = np.random.randint(1, self.stringLength - 1)
        locn2 = locn1
        while locn2 == locn1:
            locn2 = np.random.randint(1, self.stringLength - 1)
        if locn1 > locn2:
            tmp = np.copy(child1)
            child1[0:locn2] = child2[0:locn2]
            child1[locn1:self.stringLength] = child2[locn1:self.stringLength]
            child2[0:locn2] = tmp[0:locn2]
            child2[locn1:self.stringLength] = tmp[locn1:self.stringLength]
            return child1, child2
        tmp = np.copy(child1)
        child1[0:locn2] = child2[0:locn2]
        child1[locn1:self.stringLength] = child2[locn1:self.stringLength]
        child2[0:locn2] = tmp[0:locn2]
        child2[locn1:self.stringLength] = tmp[locn1:self.stringLength]
        return child1, child2

    def mutate(self, pop):            # bitwise point mutations
        whereMutate = np.random.rand(np.shape(pop)[0], np.shape(pop)[1])
        whereMutate = np.where(whereMutate < self.pm)
        pop[whereMutate] = 1 - pop[whereMutate]
        return pop

    def runGA(self):     # run simple genetic algorithme
        fid = self.fid   # output file
        for gen in range(self.nGens):  # for each generation gen
            # Compute fitness of the pop
            fitness = self.fitFcn(self.pop)  # measure fitness
            # initialize new population
            newPop = np.zeros((self.popSize, self.stringLength), dtype='int64')
            # create new population newPop via selection and crossovers with prob pc
            # create popSize/2 pairs of offspring
            for pair in range(0, self.popSize, 2):
                # tournament selection of two parent indices
                p1, p2 = self.tournament(
                    self.pop, fitness, self.popSize)  # p1, p2 integers
                child1 = np.copy(self.pop[p1, :])       # child1 for newPop
                child2 = np.copy(self.pop[p2, :])       # child2 for newPop
                if np.random.rand() < self.pc:                 # with prob self.pc
                    child1, child2 = self.xover(child1, child2)  # do crossover
                # add offspring to newPop
                newPop[pair, :] = child1
                newPop[pair + 1, :] = child2
            # mutations to population with probability pm
            newPop = self.mutate(newPop)
            self.pop = newPop
            fitness = self.fitFcn(self.pop)    # fitness values for population
            self.bestfit = fitness.max()       # fitness of (first) most fit chromosome
            # save best fitness for plotting
            self.bestfitarray[gen + 1] = self.bestfit
            # save mean fitness for plotting
            self.meanfitarray[gen + 1] = fitness.mean()
            self.bestloc = np.where(fitness == self.bestfit)[
                0][0]  # most fit chromosome locn
            self.bestchrome = self.pop[self.bestloc,
                                       :]              # most fit chromosome
            if (np.mod(gen, 10) == 0):            # print epoch, max fitness
                print("generation: ", gen + 1, "max fitness: ", self.bestfit)
        fid.write("\nfinal population, fitnesses: (up to 1st 100 chromosomes)\n")
        fitness = self.fitFcn(self.pop)         # compute population fitnesses
        self.bestfit = fitness.max()            # fitness of (first) most fit chromosome
        self.bestloc = np.where(fitness == self.bestfit)[
            0][0]  # most fit chromosome locn
        self.bestchrome = self.pop[self.bestloc,
                                   :]              # most fit chromosome
        for c in range(min(100, self.popSize)
                       ):  # for each of first 100 chromosomes
            fid.write("  {}  {}\n".format(self.pop[c, :], fitness[c]))
        fid.write("Best:\n  {} at locn {}, fitness: {}\n\n".format(
            self.bestchrome, self.bestloc, self.bestfit))
        pl.ion()      # activate interactive plotting
        pl.xlabel("Generation")
        pl.ylabel("Fitness of Best, Mean Chromosome")
        pl.plot(self.bestfitarray, 'kx-', self.meanfitarray, 'kx--')
        pl.show()
        pl.pause(0)
        fid.close()
    # You need to write a function to compute the fitness of the population members:
# it was originally 25 for 21 moves
# so 25/21 * 194 = 231

    def fitFcn(self, pop):                   # compute population fitness
        det_fitness = np.array([])   # initialize fitness values (1D array)
        for chromosome in pop:
            det_fitness = np.append(
                det_fitness, self.simulate(chromosome, 231))
        return det_fitness

    def simulate(self, rules, tMax):  # simulate rules for tMax time steps
        # initialize environment (copy to protect baseline)
        env = np.copy(self.env)
        x = self.initLocx            # ant's initial x,y coordinates
        y = self.initLocy
        dir = self.initDir           # ant's initial orientation (direction)
        max_distance = 203  # farthest distance from the left gone
        # true distance is 214 10 padding for after
        curr_max = 12  # farthest distance from the left gone
        penalty = 0
        # additional move cost for like bunny hopping?
        #
        for t in range(tMax):
            # sees() returns visual field as 1D array
            view = self.sees(x, y, dir, env)
            able = self.validrule(x, y, dir, env)
            if able.size == 0:
                print('stuck in: x:' + str(x) + ', y:' + str(y))
            # finds, returns first matching rule     NEEDED
            r = self.rulematch(rules, view)
            if np.size(r) != 0:               # if find a rule r that applies
                # decode r's action as 'GF'/'GR'/'GL'    NEEDED
                action = self.decode(r)
                if action not in able:
                    action = able[random.randint(0, (len(able)) - 1)]
                    penalty += 1
            else:                             # none of ant's rules apply
                # random default action
                action = able[random.randint(0, (len(able)) - 1)]
                penalty += 1
            # apply rule to get agent's new state
            x, y, dir = self.userule(action, x, y, dir)
            if y > curr_max:
                curr_max = y
            if self.dies(
               x, y, env):      # if ant enters boundary cell then it dies
                return (max(curr_max - (0.25 * penalty), 0))
        return (max(curr_max - (0.25 * penalty), 0))

    def sees(self, x, y, dir, env):      # returns what ant at x,y sees in direction dir
        # sees 4 x 4 top right, right in front, 3 in front, 4 in front,
        # underneath, and underneath 1 right, assuming facing right
        rightfov = np.array([env[x - 4, y + 4], env[x, y + 1], env[x, y + 3], env[x, y + 4],
                             env[x + 1, y], env[x + 1, y + 1]])
        leftfov = np.array([env[x - 4, y - 4], env[x, y - 1], env[x, y - 3], env[x, y - 4],
                            env[x + 1, y], env[x + 1, y - 1]])
        if dir == 'R':                # if looking UP
            return rightfov
        elif dir == 'L':              # else if looking LEFT
            return leftfov
        else:
            print("ERROR IN sees(): unrecognized dir = ", dir, "\n")
            return np.zeros(6)

    def rulematch(self, rules, view):
        for chunk in np.split(rules, 12):
            if np.array_equal(chunk[:6], view):
                return chunk
        return np.array([])           # no rules match (return empty array)

    def validrule(self, x, y, dir, env):
        # GJ = Jump 4 by 4, GF = go forward one tile, GF4 = go foward 4 tiles with a little bunny hop
       # GF5 = go forward 5 tiles with a bigger jump,  GD = drop in direction facing by 1
        # NEED BOTH DIAGONAL DROP AND STRAIGHT DROP
        # legal ant actions
        normal_mvmt = np.array(['GJ', 'GF', 'GF4', 'GF5', 'GD', 'GFD'])
        new_mvmt = normal_mvmt
        if dir == 'R':
            if (env[x - 4, y + 4] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GJ'))

            if (env[x, y + 1] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF'))

            if (env[x, y + 3] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF4'))

            if (env[x, y + 4] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF5'))

            if (env[x + 1, y + 1] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GFD'))

            if (env[x + 1, y] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GD'))
            elif (env[x + 1, y] == 0):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GJ'))
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF'))
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF4'))
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF5'))

            return new_mvmt

        elif dir == 'L':
            if (env[x - 4, y - 4] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GJ'))

            if (env[x, y - 1] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF'))

            if (env[x, y - 3] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF4'))

            if (env[x, y - 4] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GF5'))

            if (env[x + 1, y - 1] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GFD'))

            if (env[x + 1, y] == 1):
                new_mvmt = np.delete(new_mvmt, np.where(new_mvmt == 'GD'))

            return new_mvmt

        else:
            print('Error in validrule')

    def userule(self, act, x, y, dir):   # take action act at locn x,y having orientation dir
        if act == 'GJ':               # if action is Go Forward, return agent's new state
            if dir == 'R':
                return x - 4, y + 4, dir
            elif dir == 'L':
                return x - 4, y - 4, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        elif act == 'GF':
            if dir == 'R':
                return x, y + 1, dir
            elif dir == 'L':
                return x, y - 1, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        elif act == 'GF4':
            if dir == 'R':
                return x, y + 3, dir
            elif dir == 'L':
                return x, y - 3, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        elif act == 'GF5':
            if dir == 'R':
                return x, y + 4, dir
            elif dir == 'L':
                return x, y - 4, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        elif act == 'GD':
            if dir == 'R':
                return x + 1, y, dir
            elif dir == 'L':
                return x + 1, y, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        elif act == 'GFD':
            if dir == 'R':
                return x + 1, y + 1, dir
            elif dir == 'L':
                return x + 1, y - 1, dir
            else:
                print("ERROR in userule(): unrecognize dir = ", dir)
                return x, y, dir
        else:
            print("ERROR unrecognized action in userule(): ", act)
            return x, y, dir

# You need to write the following function that, given a set of bits extracted from
# a chromosome, returns the action that they represent ('GF', 'GR', or 'GL')
# need decode

    def decode(self, bits):
        # *** TO BE WRITTEN ***
        # also maybe elitism?
        # asdfasdfasdf useful chunk 1 asdfasdf     adsfasdf          asdfasdfasdf
        # markers for useful/useless dna ^
        # randomly useful/useless play with that
        # maybe deactivate multiples
        #

        # np.array(['GJ','GF','GF4', 'GF5', 'GD', 'GFD'])
        movement = 'GJ'
        if np.array_equal(bits[6:], [0, 0, 0]):
            movement = 'GD'
        if np.array_equal(bits[6:], [0, 0, 1]):
            movement = 'GFD'
        if np.array_equal(bits[6:], [0, 1, 0]):
            movement = 'GF5'
        if np.array_equal(bits[6:], [1, 0, 0]):
            movement = 'GF4'
        if np.array_equal(bits[6:], [0, 1, 1]):
            movement = 'GF'
        if np.array_equal(bits[6:], [1, 0, 1]):
            movement = 'GF'
        return movement


# need dies


    def dies(self, x, y, env):          # agent dies if in boundary cell
        if (x >= 17) or (y <= 9) or (y >= 205):
            return True                          # so it dies
        return False                            # ant lives
