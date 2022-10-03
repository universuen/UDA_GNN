from torch import Tensor


class LossHistory:
    def __init__(self):
        self.losses = []

    @property
    def avg_loss(self):
        return sum(self.losses) / len(self.losses)

    def append(self, loss: float | Tensor):
        if type(loss) is Tensor:
            loss = loss.item()
        self.losses.append(loss)

