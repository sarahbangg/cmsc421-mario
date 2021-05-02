import SGA

# stringLength, popSize, nGens, pm, pc
# playing with multiple genes encoding the same thing, "epigenetics/avg"
# going to double, increasing from 12 to 24 rules
# going from 9 bit length 10 bit length
ga = SGA.sga(240, 100, 41, 0.001, 0.4)
ga.runGA()
