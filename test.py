from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def stateEncode(state, info):
    encoded = np.zeros((25,32), dtype=int)

    #Mario
    yLevel = int((len(state) - info["y_pos"] + 49) / 8) - 5
    print("YLevel: " + str(yLevel))
    print("Y: " + str(info["y_pos"]))
    marioSearch = state[yLevel * 8: yLevel * 8 + 16,:]
    for y in range(yLevel, yLevel + 2):
        for x in range(11,16):
            pixels = state[y * 8 + 44: y * 8 + 44 + 8, x * 8: x * 8 + 8].tolist()
            for line in pixels:
                if([248, 56, 0] in line):
                    encoded[y][x] = 9
                    break

    for y in range(25):
        for x in range(32):
            pixel = list(state[y * 8 + 44][x * 8 + 4])
            if encoded[y][x] != 0:
                continue
            elif(pixel == [228, 92, 16] or pixel == [252, 160, 68]):
                encoded[y][x] = 1
                #print("Land")

                if(y <= 22):
                    legs = state[(y * 8 + 44) + 10: (y * 8 + 44) + 12, (x * 8): (x * 8) + 8]
                    if([0,0,0] in legs or [240, 208, 176] in legs):
                        encoded[y][x] = 3
                        encoded[y+1][x] = 3
                        encoded[y][x+1] = 3
                        encoded[y+1][x+1] = 3
                        #print("Goomba")
            elif(y != 0 and encoded[y - 1][x] == 2):
                encoded[y][x] = 2
                #print("Pipe")
            elif(pixel == [104, 136, 252] or pixel == [252, 252, 252]):
                encoded[y][x] = 0
                #print("Sky")
            elif(pixel == [184, 248, 24]):
                encoded[y][x] = 0 #7
                #print("Grass")
            elif(pixel == [0, 168, 0]):
                xTemp = x * 8 + 4
                while list(state[y * 8 + 44][xTemp - 2]) == [0, 168, 0]:
                    xTemp -= 2
                if(list(state[y * 8 + 44 + 16][xTemp]) == [104, 136, 252]):
                    encoded[y][x] = 2
                    #print("Pipe")
                else:
                    encoded[y][x] = 0 #8
                    #print("Dark Grass")
            else:
                encoded[y][x] = 5
                print(pixel)
    return encoded

def printState(state):
    for y in range(len(state)):
        for x in range(len(state[0])):
            print(state[y][x], end='')
        print(" Hight: " + str(y * 8 + 44))

done = True
for step in range(75):
    if done:
        state = env.reset()
    if step % 25 == 0:
        state, reward, done, info = env.step(1)
    else:
        state, reward, done, info = env.step(4)
    #state, reward, done, info = env.step(env.action_space.sample())
    #print("State:")
    print(str(info["x_pos"]) + ", " + str(info["y_pos"]))
    env.render()

printState(stateEncode(state, info))
input("")

# 6 simple movement types:
# - 0: Nothing
# - 1: Right
# - 2: Right Jump
# - 3: Right Run
# - 4: Right Run Jump
# - 5: Jump
# - 6: Left

# Rule:
# Output: 3 bits to encode 7 movements. Extra for right


