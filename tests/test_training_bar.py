import random
from time import sleep

from src.utils import training_bar

for e in range(100):
    for idx in range(100):
        sleep(0.3)
        print(training_bar(e, idx, 10000, loss_a=random.random(), loss_b=random.random()))
