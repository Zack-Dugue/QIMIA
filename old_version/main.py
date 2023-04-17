import random
import math
for i in range(100):
    x = random.random()
    new_x = math.log((1-x),math.e)/(-2)
    print(new_x)