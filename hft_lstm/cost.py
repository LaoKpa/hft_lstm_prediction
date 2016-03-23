from blocks.bricks.cost import CostMatrix
from blocks.bricks.base import application


class AbsolutePercentageError(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = abs((y - y_hat) / y)
        return cost
