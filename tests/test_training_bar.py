import random
from time import sleep

from src.api.training_processes import training_bar

for e in range(100):
    sleep(0.3)
    print(training_bar(e, 100, loss_a=random.random(), loss_b=random.random()))
