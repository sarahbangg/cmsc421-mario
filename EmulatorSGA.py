# Simple Genetic Algorithm for Mario Emulator AI

import matplotlib.pyplot as plt
import numpy as np
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Chromosome: 10 rules
# Rules:
#   Visual: 5 bits (binary visual type [1: block/entity, 0: air] and 5 spots)
#   Action: 3 bits

class sga:
    def __init__(self, ruleSize, numRules, popSize, nGens, pm, pc, visualSpots):
        fid=open("results.txt","w")        # open, initialize output file
        self.fid = fid
        
        # Input Values
        self.ruleSize = ruleSize
        self.numRules = numRules
        self.stringLength = ruleSize * numRules   # number of bits in a chromosome
        self.popSize = popSize + (popSize % 2) # Even population size
        self.pm = pm                       # probability of mutation
        self.pc = pc                       # probability of crossover
        self.nGens = nGens                 # max number of generations
        self.visualSpots = visualSpots     # Locations of visual inputs

        self.bestfitarray = np.zeros(self.nGens)  # array of max fitness vals each generation
        self.meanfitarray = np.zeros(self.nGens)  # array of mean fitness vals each generation

        self.pop = np.random.rand(self.popSize,self.stringLength)
        self.pop = np.where(self.pop<0.5,1,0)  # create initial pop

    def fitFcn(self,pop, gen=0):          # compute population fitness values   
        fitness = np.zeros(self.popSize)     # initialize fitness values (1D array)
        TMax = int(gen / 5) * 15 + 200 
        for p in range(self.popSize):        # Loop through population
            fitness[p] = self.simulate(pop[p],TMax)   # Simulate run for fitness
        return fitness

    def simulateCust(self, chrome, tMax):
        self.simulate(chrome, tMax, visualize=True)

    def simulate(self,rules,tMax,visualize=False):  # simulate rules for tMax time steps 
        state = env.reset()         # Restart the game environment
        rTotal = 0
        state, reward, done, info = env.step(0) # Conduct null move
        jumpCount = 0
        maxXcount = 0
        tempX = 0
        if(visualize):
            self.fid.write("Best Chromosome Actions for {} Steps\n * Every move is done 5 times".format(tMax))

        for step in range(0,tMax,5):    # Loop through each game step
            visualEcoding = self.stateEncode(state, info)   # Get Visual Encoding
            rule = self.rulematch(rules, visualEcoding)     # finds, returns first matching rule
            action = self.decode(rule)                      # Gets decoded action
            if(action == 2 or action == 4):
                jumpCount += 5
            if(visualize):
                self.fid.write(str(action))
            for i in range(5):
                if(jumpCount > 20):
                    state, reward, done, info = env.step(action - 1) # Refresh Jump
                    jumpCount = 0
                else:
                    state, reward, done, info = env.step(action) # Conduct move based on AI
                rTotal += reward

                if(visualize):
                    env.render()
                if done or info["life"] < 2 or maxXcount > 10:                # End if end reached
                    return rTotal

            if(tempX >= info["x_pos"]):
                maxXcount += 1
            else:
                tempX = info["x_pos"]
                maxXcount = 0

        return rTotal               # Return reward given by emulator

    def stateEncode(self, state, info):
        encoded = []

        #Mario
        yLevel = int((len(state) - info["y_pos"] + 49) / 8) - 5
        marioSearch = state[yLevel * 8: yLevel * 8 + 16,:]
        marioTL = []
        for y in range(yLevel - 1, yLevel + 4):
            for x in range(16):
                pixels = state[y * 8 + 44: y * 8 + 44 + 8, x * 8: x * 8 + 8].tolist()
                for line in pixels:
                    if([248, 56, 0] in line):
                        if(marioTL == []):
                            marioTL = [x,y]
                        break

        if marioTL == []:
            marioTL = [19,14]

        for c in self.visualSpots:
            x = min(max((marioTL[0] + c[0]) * 8 + 4, 0), 256)
            y = min(max((marioTL[1] + c[1]) * 8 + 44, 0), 240)
            pixel = list(state[y][x])

            if(pixel == [104, 136, 252] or # Sky
               pixel == [252, 252, 252] or # Cloud
               pixel == [184, 248, 24] or # Light Grass
               pixel == [172, 124, 0] or # Mario Part
               pixel == [248, 56, 0] or # Mario Part
               pixel == [240, 208, 176]): # Dead Goomba
                spotVal = 0
            elif(pixel == [228, 92, 16] or # Ground
                 pixel == [252, 160, 68] or # Box
                 pixel == [136, 20, 0]): # Also Box
                spotVal = 1
            elif(pixel == [0, 168, 0]): # Pipe or Dark Grass
                # Loop to Top Left corner of green and check for lip
                xTemp = (marioTL[0] + c[0]) * 8 + 4
                yTemp = (marioTL[1] + c[1]) * 8 + 44

                while list(state[yTemp - 2][xTemp]) == [0, 168, 0]: # Go to top of green
                    yTemp -= 2
                while list(state[yTemp][xTemp - 2]) == [0, 168, 0]: # Go to top left of green
                    xTemp -= 2

                if(list(state[yTemp + 16][xTemp]) == [104, 136, 252]): # If there is a lip
                    spotVal = 1 # Pipe
                else:
                    spotVal = 0 # Otherwise Dark Grass
            else: # Unknown
                spotVal = 0
                print(str(pixel) + ", x: " + str(x) + ", y: " + str(y))

            encoded.append(spotVal)

        return encoded

    def visualToGene(self, visual):
        r = []
        for x,y in importantSpots:
            blockType = np.zeros(4)
            if(visual[y][x] != 5 and visual[y][x] != 4):
                blockType[visual[y][x]] = 1
            r = np.concatenate((r, blockType))
        return r

    def rulematch(self, rules, visual):
        #vis = self.visualToGene(visual)

        best = 0
        i = -1
        for r in range(0, len(rules), self.ruleSize):
            match = np.dot(rules[r:r+self.ruleSize-3], visual)
            if match > best:
                best = match
                i = r

        if(i == -1):
            return [0,0,0]
        return rules[i+self.ruleSize-3:i+self.ruleSize]

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
            self.bestfitarray[gen] = self.bestfit        # save best fitness for plotting
            self.meanfitarray[gen] = fitness.mean()      # save mean fitness for plotting
            self.bestloc = np.where(fitness == self.bestfit)[0][0]  # most fit chromosome locn
            self.bestchrome = self.pop[self.bestloc,:]              # most fit chromosome
            

            if (self.bestfit > bestChrom[3]):
                bestChrom = [self.bestchrome, gen, self.bestloc, self.bestfit]
                fid.write("New Best! Generation {} - Fitness: {} with Chromosome: {}\n".format(gen, self.bestfit, self.bestchrome))
                
            if (np.mod(gen,5)==0):            # print epoch, max fitness
                print("Generation:",gen,"- Max Fitness:",self.bestfit)

        fid.write("\nfinal population, fitnesses: (up to 1st 100 chromosomes)\n")
        for c in range(min(20,self.popSize)):  # for each of first 100 chromosomes
            fid.write("  {}  {}\n".format(self.pop[c,:],fitness[c])) 
        fid.write("Best:\n  {} at locn {}, fitness: {}\n\n".format(self.bestchrome,self.bestloc,self.bestfit))
        fid.write("Best of all Gens:\n  {} at gen: {}, locn {}, fitness: {}\n".format(bestChrom[0], bestChrom[1], bestChrom[2], bestChrom[3]))

        # English Best Chromosome Printout 
        for r in range(0,len(bestChrom[0]),self.ruleSize):
            visual = np.where(bestChrom[0][r:r+self.ruleSize - 3]<0.5,"Air","Block/Entity")  # create initial pop
            move = self.decode(bestChrom[0][r+self.ruleSize - 3:r+self.ruleSize])
            fid.write("Rule {}: If agent sees in given visual spots, ".format(int(r/(self.ruleSize))))
            for v in visual:
                fid.write(str(v) + ", ")
            fid.write("then {}\n".format(move))
        fid.write("Where move 0 is nothing, move 1 is right, 2 is right jump, 3 is right run, 4 is right run jump, 5 is jump, and 6 is left\n\n")

        # Simulate Best
        self.simulateCust(bestChrom[0], 1000)
        fid.close()

        # Plot best and mean fitnesses
        x = np.array(range(self.nGens))
        y1 = self.bestfitarray
        y2 = self.meanfitarray
        y3 = np.array(x / 5, dtype=int) * 15 + 200

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness', color=color)
        line1 = ax1.plot(x, y1, color=color, label="Best Fitness")
        line2 = ax1.plot(x, y2, '--', color=color, label="Mean Fitness")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Time Steps', color=color)
        line3 = ax2.plot(x, y3, color=color, label="Time Steps")
        ax2.tick_params(axis='y', labelcolor=color)

        lns = line1+line2+line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        fig.tight_layout()
        plt.title("Mean and Best Fitnesses vs Generation with Time Steps")
        plt.savefig('figs/fig' + str(self.ruleSize) + "-" + str(self.numRules) + "-" + str(self.popSize) + "-" + str(self.nGens) + "-" + str(self.pm) + "-" + str(self.pc) + ".pdf")
        plt.show()


if __name__ == "__main__":
    visualSpots = [(3,0), (3, 1), (3, 3), (5, 0), (5, 2), (2, -2), (4, -2)]
    genExample = sga(10, 10, 50, 100, 0.07, 0.25, visualSpots) # Smaller Chromosome Version with Increasing TMax
    genExample.runGA()
    #chromo = [0,1,0,1,0,1,1,0,0,
    #          1,0,0,0,0,0,1,1,1,
    #          1,1,1,0,0,1,1,0,0,
    #          0,1,1,1,0,0,0,1,1,
    #          1,0,1,1,1,0,1,0,1,
    #          0,0,0,0,1,0,0,0,0,
    #          0,1,0,0,1,1,0,0,0,
    #          1,0,1,1,0,0,1,1,1,
    #          1,1,0,1,0,1,1,0,0,
    #          1,1,0,1,1,0,0,0,1]
    #genExample.simulateCust(chromo, 1000)