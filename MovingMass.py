import numpy as np

from swarmsim.metrics.Aggregation import Aggregation

class MovingMass(Aggregation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initial_centroid = self.center_of_mass()

    def _calculate(self):
        centroid = self.center_of_mass()
        return super()._calculate() + np.linalg.norm(centroid - self.initial_centroid)