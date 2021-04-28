
import numpy as np
import random

# The first three items below might go in __init__()

# Environment's food trail (must be used):
#replace with temp_env, 0 = space, 1 = solid object

self.env =  np.array([             # environment: 0 = empty, 1 = food
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],           
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0], 
          [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
          [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
          [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
          [0,0,0,0,0,1,1,1,1,0,1,1,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

# Ant's initial state:
#adjust to 6? keep it consistent with pull
# doesn't need to look behind 
self.initLocx = 3                  # ant's initial location x,y
self.initLocy = 3
self.initDir = 'R'                 # ant's initial orientation (URDL) 

# Some potentially useful definitions:

#can only turn left or right 
self.cw  = {'R':'L','L':'R'}   # rotate orientation clockwise
self.ccw = {'R':'L','L':'R'}   # rotate orientation counter-cw

#max jump 4 up, 7 to the right/left, slight curving to tile system
#GJ = Jump 4 by 4, GF = go forward one tile, GF4 = go foward 4 tiles with a little bunny hop
#GF5 = go forward 5 tiles with a bigger jump,  GD = drop in direction facing by 1
self.legalActs = np.array(['GJ','GF','GF4', 'GF5', 'GD'])    # legal ant actions

# The remaining items refer to methods of the class sga:

# You need to write a function to compute the fitness of the population members: 
#it was originally 25 for 21 moves 
# so 25/21 * 194 = 231
def fitFcn(self,pop):                   # compute population fitness 
   det_fitness = np.array([])   # initialize fitness values (1D array)
   for chromosome in pop:
      det_fitness = np.append(det_fitness, self.simulate(chromosome, 231))
   return fitness

# In computing the fitness of an individual chromosome in the population,
# one approach would be to extract the rule set encoded by that chromosome
# and call simulate() to measure how well an ant using those rules does:
#
#  foodfound = self.simulate(rules,25)  # simulate ant for 25 time steps 
#
# A partial implementation of simulate() is given below.
  
# Run the ant simulator, returning the ant's "food found" 
# You must write rulematch() and decode() based on your encoding of
# rules in a chromosome.

#change sim
def simulate(self,rules,tMax):  # simulate rules for tMax time steps 
   env = np.copy(self.env)      # initialize environment (copy to protect baseline)
   x = self.initLocx            # ant's initial x,y coordinates
   y = self.initLocy
   dir = self.initDir           # ant's initial orientation (direction)  
   max_distance = 194                #farthest distance from the left gone
   curr_max = 6                         #farthest distance from the left gone
   curr_x = 6                           #track distance from 
   for t in range(tMax):
     view = self.sees(x,y,dir,env)     # sees() returns visual field as 1D array 
     r = self.rulematch(rules,view)    # finds, returns first matching rule     NEEDED
     if np.size(r) != 0:               # if find a rule r that applies
       action = self.decode(r)         # decode r's action as 'GF'/'GR'/'GL'    NEEDED
     else:                             # none of ant's rules apply
       action = self.legalActs[random.randint(0,2)]  # random default action 
     x,y,dir = self.userule(action,x,y,dir)   # apply rule to get agent's new state
     if self.dies(x,y,env):      # if ant enters boundary cell then it dies
       return foodfound          # return food found so far 
     if env[x,y] == 1:           # if food found at new location
       foodfound += 1            # increment foodfound
       env[x,y] = 0              # consume food 
   return foodfound


#change sees
def sees(self,x,y,dir,env):      # returns what ant at x,y sees in direction dir 
   adj = np.array([env[x-1,y-1],env[x-1,y],env[x-1,y+1],env[x,y+1], 
                   env[x+1,y+1],env[x+1,y],env[x+1,y-1],env[x,y-1]]) # adjacent cells, cw
   if dir == 'U':                # if looking UP
     return np.array([adj[7],adj[0],adj[1],adj[2],adj[3]])
   elif dir == 'R':              # else if looking RIGHT   
     return adj[1:6]
   elif dir == 'D':              # else if looking DOWN
     return adj[3:8]
   elif dir == 'L':              # else if looking LEFT
     return np.array([adj[5],adj[6],adj[7],adj[0],adj[1]])
   else:
     print("ERROR IN sees(): unrecognized dir = ",dir,"\n")
     return np.zeros(5)

# You need to write the following function that, given a set of rules extracted from
# their encoding in a chromosome, finds and returns the first rule that matches the
# ant's view (visual field). If no rules match, array([]) is returned.

#need reulmatch
def rulematch(self,rules,view):  
   # *** TO BE WRITTEN ***
   return np.array([])           # no rules match (return empty array)

#need userule
def userule(self,act,x,y,dir):   # take action act at locn x,y having orientation dir    
   if act == 'GF':               # if action is Go Forward, return agent's new state
      if   dir == 'U': return x-1,y,dir
      elif dir == 'R': return x,y+1,dir
      elif dir == 'D': return x+1,y,dir
      elif dir == 'L': return x,y-1,dir
      else:
        print("ERROR in userule(): unrecognize dir = ",dir)
        return x,y,dir
   elif act == 'GR':             # if action is Go Right, return agent's new state
      if   dir == 'U': return x,y+1,self.cw[dir]
      elif dir == 'R': return x+1,y,self.cw[dir]
      elif dir == 'D': return x,y-1,self.cw[dir]
      elif dir == 'L': return x-1,y,self.cw[dir]
      else:
        print("ERROR in userule(): unrecognize dir = ",dir)
        return x,y,dir
   elif act == 'GL':             # if action is Go Left, return agent's new state
      if   dir == 'U': return x,y-1,self.ccw[dir]
      elif dir == 'R': return x-1,y,self.ccw[dir]
      elif dir == 'D': return x,y+1,self.ccw[dir]
      elif dir == 'L': return x+1,y,self.ccw[dir]
      else:
        print("ERROR in userule(): unrecognize dir = ",dir)
        return x,y,dir
   else:
      print("ERROR unrecognized action in userule(): ",act)
      return x,y,dir

# You need to write the following function that, given a set of bits extracted from
# a chromosome, returns the action that they represent ('GF', 'GR', or 'GL')
#need decode
def decode(self,bits):           
   # *** TO BE WRITTEN ***
   return 'GF'    # DELETE THIS LINE
   

#need dies
def dies(self,x,y,env):          # agent dies if in boundary cell
   maxx,maxy = np.shape(env)     # max locns in state space
   if (x == 0) or (x == (maxx - 1)) or\
      (y == 0) or (y == (maxy - 1)):       # ant is in boundary cell
      return True                          # so it dies
   return False                            # ant lives
