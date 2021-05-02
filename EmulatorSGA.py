# Simple Genetic Algorithm for Mario Emulator AI

import pylab as pl
import numpy as np
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Chromosome: 10 rules
# Rules:
#   Visual: 4 * 5 bits (5 visual types [air, block, pipe, goomba] and 5 spots)
#   Action: 3 bits

class sga:
    def __init__(self, stringLength, popSize, nGens, pm, pc):
        fid=open("results.txt","w")        # open, initialize output file
        self.fid = fid
        
        # Input Values
        self.stringLength = stringLength   # number of bits in a chromosome
        self.popSize = popSize + (popSize % 2) # Even population size
        self.pm = pm                       # probability of mutation
        self.pc = pc                       # probability of crossover
        self.nGens = nGens                 # max number of generations

        self.bestfitarray = np.zeros(self.nGens + 1)  # array of max fitness vals each generation
        self.meanfitarray = np.zeros(self.nGens + 1)  # array of mean fitness vals each generation

        self.pop = np.random.rand(self.popSize,self.stringLength)
        self.pop = np.where(self.pop<0.5,1,0)  # create initial pop

        #self.initPopulation()

    def initPopulation(self):
        # Initial Population
        self.pop = np.random.rand(self.popSize,self.stringLength)
        self.pop = np.where(self.pop<0.5,1,0)  # create initial pop

        fitness = self.fitFcn(self.pop)    # fitness values for initial population
        self.bestfit = fitness.max()       # fitness of (first) most fit chromosome
        self.bestloc = np.where(fitness == self.bestfit)[0][0]  # most fit chromosome locn
        self.bestchrome = self.pop[self.bestloc,:]              # most fit chromosome
        self.bestfitarray[0] = self.bestfit           #  (+ 1 for init pop plus nGens)
        self.meanfitarray[0] = fitness.mean()

        fid = self.fid
        fid.write("popSize: {}  nGens: {}  pm: {}  pc: {}\n".format(self.popSize,self.nGens,self.pm,self.pc))
        fid.write("initial population, fitnesses: (up to 1st 100 chromosomes)\n")
        for c in range(min(100,self.popSize)):   # for each of first 100 chromosomes 
            fid.write("  {}  {}\n".format(self.pop[c,:],fitness[c]))
        fid.write("Best initially:\n  {} at locn {}, fitness = {}\n".format(self.bestchrome,self.bestloc,self.bestfit))

    def fitFcn(self,pop, gen=0):          # compute population fitness values   
        fitness = np.zeros(self.popSize)     # initialize fitness values (1D array)
        TMax = int(gen / 10) * 50 + 100 
        for p in range(self.popSize):        # Loop through population
            fitness[p] = self.simulate(pop[p],TMax)   # Simulate run for fitness
        return fitness

    def simulateBest(self, tMax):
        self.simulate(self.bestchrome, tMax, visualize=True)

    def simulate(self,rules,tMax,visualize=False):  # simulate rules for tMax time steps 
        state = env.reset()         # Restart the game environment
        rTotal = 0
        state, reward, done, info = env.step(0) # Conduct null move

        for step in range(0,tMax,5):    # Loop through each game step
            visualEcoding, marioLoc = self.stateEncode(state, info)   # Get Visual Encoding
            rule = self.rulematchE(rules, visualEcoding, marioLoc)     # finds, returns first matching rule
            action = self.decode(rule)                      # Gets decoded action

            for i in range(5):
                state, reward, done, info = env.step(action) # Conduct move based on AI
                rTotal += reward

                if(visualize):
                    self.fid.write(str(action))
                    env.render()
                if done or info["life"] < 2:                # End if end reached
                    break

        return rTotal               # Return reward given by emulator

    def stateEncode(self, state, info):
        encoded = np.zeros((25,32), dtype=int)

        #Mario
        yLevel = int((len(state) - info["y_pos"] + 49) / 8) - 5
        marioSearch = state[yLevel * 8: yLevel * 8 + 16,:]
        marioTL = []
        for y in range(yLevel - 1, yLevel + 4):
            for x in range(16):
                pixels = state[y * 8 + 44: y * 8 + 44 + 8, x * 8: x * 8 + 8].tolist()
                for line in pixels:
                    if([248, 56, 0] in line):
                        encoded[y][x] = 4
                        if(marioTL == []):
                            marioTL = [x,y]
                        break

        for y in range(25):
            for x in range(32):
                pixel = list(state[y * 8 + 44][x * 8 + 4])
                if encoded[y][x] != 0: # If already found (If mario or goomba is there)
                    continue
                elif(pixel == [228, 92, 16] or pixel == [252, 160, 68] or pixel == [136, 20, 0]):
                    encoded[y][x] = 1 # Ground / Box

                    if(y <= 22): # Check for goomba
                        legs = state[(y * 8 + 44) + 10: (y * 8 + 44) + 12, (x * 8): (x * 8) + 8]
                        if([0,0,0] in legs or [240, 208, 176] in legs):
                            encoded[y][x] = 3
                            encoded[y+1][x] = 3 # Goomba
                            if(x < 31):
                                encoded[y][x+1] = 3
                                encoded[y+1][x+1] = 3
                elif(y != 0 and encoded[y - 1][x] == 2):
                    encoded[y][x] = 2 # Lower Pipe
                elif(pixel == [104, 136, 252] or pixel == [252, 252, 252] or pixel == [172, 124, 0] or pixel == [248, 56, 0]):
                    encoded[y][x] = 0 # Sky / Cloud
                elif(pixel == [184, 248, 24]):
                    encoded[y][x] = 0 # Light Grass counted as sky
                elif(pixel == [0, 168, 0]): # Pipe or Dark Grass
                    xTemp = x * 8 + 4
                    while list(state[y * 8 + 44][xTemp - 2]) == [0, 168, 0]: # Go to top left of green
                        xTemp -= 2
                    if(list(state[y * 8 + 44 + 16][xTemp]) == [104, 136, 252]): # If there is a lip
                        encoded[y][x] = 2 # Pipe
                    else:
                        encoded[y][x] = 0 # Otherwise Dark Grass
                else:
                    encoded[y][x] = 5 # Unknown
                    print(str(pixel) + ", x: " + str(x) + ", y: " + str(y))
        return (encoded, marioTL)

    def rulematch(self, rules, visual, marioLoc):
        if marioLoc == []:
            marioLoc = [19,14]
        importantSpots = [(marioLoc[0] + 2, marioLoc[1]), (marioLoc[0], marioLoc[1] - 2), (marioLoc[0] + 5, marioLoc[1]), (marioLoc[0] + 5, marioLoc[1] - 2), (marioLoc[0] + 12, marioLoc[1] + 4)]
        visualToGene = []
        for x,y in importantSpots:
            blockType = np.zeros(4)
            if(visual[y][x] != 5 and visual[y][x] != 4):
                blockType[visual[y][x]] = 1
            visualToGene = np.concatenate((visualToGene, blockType))

        best = 0
        i = -1
        for r in range(0, len(rules), 23):
            match = np.dot(rules[r:r+20], visualToGene)
            if match > best:
                best = match
                i = r

        if(i == -1):
            return [0,0,0]
        return rules[i+20:i+23]

    def rulematchE(self, rules, visual, marioLoc):
        if marioLoc == []:
            marioLoc = [19,14]
        importantSpots = [(marioLoc[0] + 2, marioLoc[1]), (marioLoc[0], marioLoc[1] - 2), (marioLoc[0] + 5, marioLoc[1]), (marioLoc[0] + 5, marioLoc[1] - 2), (marioLoc[0] + 12, marioLoc[1] + 4)]
        visualToGene = []
        
        for x,y in importantSpots:
            if(visual[y][x] == 1 or visual[y][x] == 2 or visual[y][x] == 3):
                visualToGene.append(1)
            else:
                visualToGene.append(0)

        best = 0
        i = -1
        for r in range(0, len(rules), 8):
            match = 5 - np.count_nonzero(rules[r:r+5] - visualToGene)
            if match > best:
                best = match
                i = r

        if(i == -1):
            return [0,0,0]
        return rules[i+5:i+8]

    def decode(self, rule):
        rule = list(rule)
        if(rule == [0,0,0]):
            return 1 # Right
        elif(rule == [0,0,1]):
            return 1 # Right
        elif(rule == [0,1,0]):
            return 2 # Right Jump
        elif(rule == [0,1,1]):
            return 3 # Right Run
        elif(rule == [1,0,0]):
            return 4 # Right Run Jump
        elif(rule == [1,0,1]):
            return 5 # Jump
        elif(rule == [1,1,0]):
            return 6 # Left
        return 0 # No Move

    # conduct tournaments to select two offspring
    def tournament(self,pop,fitness,popsize):  # fitness array, pop size
        # select first parent par1
        cand1 = np.random.randint(popsize)      # candidate 1, 1st tourn., int
        cand2 = cand1                           # candidate 2, 1st tourn., int
        while cand2 == cand1:                   # until cand2 differs
            cand2 = np.random.randint(popsize)   #   identify a second candidate
        if fitness[cand1] > fitness[cand2]:     # if cand1 more fit than cand2 
            par1 = cand1                         #   then first parent is cand1
        else:                                   #   else first parent is cand2
            par1 = cand2
        # select second parent par2
        cand1 = np.random.randint(popsize)      # candidate 1, 2nd tourn., int
        cand2 = cand1                           # candidate 2, 2nd tourn., int
        while cand2 == cand1:                   # until cand2 differs
            cand2 = np.random.randint(popsize)   #   identify a second candidate
        if fitness[cand1] > fitness[cand2]:     # if cand1 more fit than cand2 
            par2 = cand1                         #   then 2nd parent par2 is cand1
        else:                                   #   else 2nd parent par2 is cand2
            par2 = cand2
        return par1,par2

    def xover(self,child1,child2):    # single point crossover
        # cut locn to right of position (hence subtract 1)
        locn = np.random.randint(0,self.stringLength - 1)
        tmp = np.copy(child1)       # save child1 copy, then do crossover
        child1[locn+1:self.stringLength] = child2[locn+1:self.stringLength]
        child2[locn+1:self.stringLength] = tmp[locn+1:self.stringLength]
        return child1,child2

    def mutate(self,pop):            # bitwise point mutations
        whereMutate = np.random.rand(np.shape(pop)[0],np.shape(pop)[1])
        whereMutate = np.where(whereMutate < self.pm)
        pop[whereMutate] = 1 - pop[whereMutate]
        return pop

    def runGA(self):     # run simple genetic algorithme
        fid=self.fid   # output file
        bestChrom = [0,0,0,0]
        for gen in range(self.nGens): # for each generation gen
            # Compute fitness of the pop
            if(gen == 0):
                fitness = self.fitFcn(self.pop, gen=gen)  # measure fitness 
            # initialize new population
            newPop = np.zeros((self.popSize,self.stringLength),dtype = 'int64')
            # create new population newPop via selection and crossovers with prob pc
            for pair in range(0,self.popSize,2):  # create popSize/2 pairs of offspring
                # tournament selection of two parent indices
                p1, p2 = self.tournament(self.pop,fitness,self.popSize)  # p1, p2 integers
                child1 = np.copy(self.pop[p1,:])       # child1 for newPop
                child2 = np.copy(self.pop[p2,:])       # child2 for newPop
                if np.random.rand() < self.pc:                 # with prob self.pc 
                    child1, child2 = self.xover(child1,child2)  #   do crossover
                newPop[pair,:] = child1                # add offspring to newPop
                newPop[pair + 1,:] = child2
            # mutations to population with probability pm
            newPop = self.mutate(newPop)
            self.pop = newPop 
            fitness = self.fitFcn(self.pop, gen=gen)    # fitness values for population
            self.bestfit = fitness.max()       # fitness of (first) most fit chromosome
            self.bestfitarray[gen + 1] = self.bestfit        # save best fitness for plotting
            self.meanfitarray[gen + 1] = fitness.mean()      # save mean fitness for plotting
            self.bestloc = np.where(fitness == self.bestfit)[0][0]  # most fit chromosome locn
            self.bestchrome = self.pop[self.bestloc,:]              # most fit chromosome
            
            if (self.bestfit > bestChrom[3]):
                bestChrom = [self.bestchrome, gen, self.bestloc, self.bestfit]
                
            if (np.mod(gen,2)==0):            # print epoch, max fitness
                print("generation: ",gen+1,"max fitness: ",self.bestfit) 
        fid.write("\nfinal population, fitnesses: (up to 1st 100 chromosomes)\n")
        fitness = self.fitFcn(self.pop)         # compute population fitnesses
        self.bestfit = fitness.max()            # fitness of (first) most fit chromosome
        self.bestloc = np.where(fitness == self.bestfit)[0][0]  # most fit chromosome locn
        self.bestchrome = self.pop[self.bestloc,:]              # most fit chromosome
        for c in range(min(100,self.popSize)):  # for each of first 100 chromosomes
            fid.write("  {}  {}\n".format(self.pop[c,:],fitness[c])) 
        fid.write("Best:\n  {} at locn {}, fitness: {}\n\n".format(self.bestchrome,self.bestloc,self.bestfit))
        fid.write("Best of all Gens:\n  {} at gen: {}, locn {}, fitness: {}\n".format(bestChrom[0], bestChrom[1], bestChrom[2], bestChrom[3]))
        
        # English Best Chromosome Printout 
        #for r in range(0,len(bestChrom[0]),7):
        #    visual = np.where(bestChrom[0][r:r+5]<0.5,"empty"," food")  # create initial pop
        #    move = self.decode(bestChrom[0][r+5:r+7])
        #    fid.write("Rule {}: If ant sees {}, {}, {}, {}, {}, then {}\n".format(int(r/7), visual[0], visual[1], visual[2], visual[3], visual[4], move))
        

        # Simulate Best
        #env = np.copy(self.env)      # initialize environment (copy to protect baseline)
        #x = self.initLocx            # ant's initial x,y coordinates
        #y = self.initLocy
        #dir = self.initDir           # ant's initial orientation (direction)
        #actionSeq= []
        #for t in range(25):
        #    view = self.sees(x,y,dir,env)     # sees() returns visual field as 1D array
        #    r = self.rulematch(bestChrom[0],view)    # finds, returns first matching rule     NEEDED
        #    if np.size(r) != 0:               # if find a rule r that applies
        #        action = self.decode(r)         # decode r's action as 'GF'/'GR'/'GL'    NEEDED
        #    else:                             # none of ant's rules apply
        #        action = self.legalActs[random.randint(0,2)]  # random default action 
        #    x,y,dir = self.userule(action,x,y,dir)   # apply rule to get agent's new state
        #    actionSeq.append(action)
        #    if self.dies(x,y,env):      # if ant enters boundary cell then it dies
        #        break          # return food found so far 
        #    if env[x,y] == 1:           # if food found at new location
        #        env[x,y] = 0              # consume food 
        #fid.write("Sequence of Actions: {}".format(actionSeq))
        #fid.close()
        #
        #pl.ion()      # activate interactive plotting
        #pl.xlabel("Generation")
        #pl.ylabel("Fitness of Best, Mean Chromosome")
        #pl.plot(self.bestfitarray,'kx-',self.meanfitarray,'kx--')
        #pl.show()
        #pl.pause(0)


if __name__ == "__main__":
    # genExample = sga(115, 50, 100, 0.5, 0.001) # Change to 230 for 10 rule run
    genExample = sga(40, 50, 50, 0.5, 0.001) # Smaller Chromosome Version with Increasing TMax
    #genExample.runGA()
    #input("Ready for best run?")
    #genExample.simulateBest(500)
    #humanMadeRules = [0,0,0,0, 0,0,0,0, 0,1,1,1, 0,1,0,0, 0,0,0,0, 1,0,0,
    #                  1,0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,
    #                  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,
    #                  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,
    #                  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,]
    AIMadeRules = np.array([1,0,0,0,0, 0,1,1,
                            0,1,0,1,1, 0,0,1,
                            1,0,1,1,1, 1,0,0,
                            1,1,1,0,1, 1,0,0,
                            1,1,1,1,1, 1,0,0])
    print(genExample.simulate(AIMadeRules, 500, True))
    input("")
    fid.close()

# Chromosome: 10 rules
# Rules:
#   Visual: 4 * 5 bits (4 visual types [air, block, pipe, goomba] and
#                       5 spots [infront, beneath, medium infront, medium infront beneath, far above])
#   Action: 3 bits ([right, right, right jump, right run, right run jump, jump, left, none]