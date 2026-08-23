# typing
from typing import override

import numpy as np

from .CaspianBinaryController import CaspianBinaryController


class CaspianBinaryRemappedController(CaspianBinaryController):
    def __init__(self, cont_mode: int):
        super().__init__()
        assert 0 <= cont_mode < 6, "There are only six different testing modes here"
        self.cont_mode = cont_mode

    @override
    def run_processor(self, observation):
        b2oh = self.bool_to_one_hot

        # translate observation to vector
        input_vector = b2oh(observation)
        # input_vector += (1,)  # add 1 as constant on input to 4th input neuron

        spikes = self.encoder.get_spikes(input_vector)
        self.processor.apply_spikes(spikes)
        self.processor.run(self.extra_ticks)
        if self.neuro_track_all:
            neuron_counts = np.asarray(self.processor.neuron_counts())
        self.processor.run(self.neuro_tpc)
        if self.neuro_track_all:
            neuron_counts += self.processor.neuron_counts()
            self.neuron_counts = neuron_counts.tolist()
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        data = self.decoder.get_data_from_processor(self.processor)
        data = [int(round(x)) for x in data]
        # three bins. One for +v, -v, omega.
        v = self.scale_v * (data[1] - data[0])
        w = self.scale_w * (data[3] - data[2])
        # these values were taken from an average of speeds/turning rates
        # from measurements of Turbopis 1, 2, 3, 4 @ (100, 90, +-0.5)
        v_mapping = [0.0, 0.276,]
        w_mapping = [0.0, 0.602,]
        v = v_mapping[data[1]] - v_mapping[data[0]]
        w = w_mapping[data[3]] - w_mapping[data[2]]

        match self.cont_mode:
            case 0:
                # Do nothing
                pass

            case 1:
                # Use RNG to split ties
                if v == 0.0:
                    v = v_mapping[1] * self.parent.rng.choice([-1, 1])
                if w == 0.0:
                    w = w_mapping[1] * self.parent.rng.choice([-1, 1])

            case 2:
                # Force v to -1, w to -1
                if v == 0.0:
                    v = -v_mapping[1]
                if w == 0.0:
                    w = -w_mapping[1]

            case 3:
                # Force v to -1, w to 1
                if v == 0.0:
                    v = -v_mapping[1]
                if w == 0.0:
                    w = w_mapping[1]

            case 4:
                # Force v to 1, w to -1
                if v == 0.0:
                    v = v_mapping[1]
                if w == 0.0:
                    w = -w_mapping[1]

            case 5:
                # Force v to 1, w to 1
                if v == 0.0:
                    v = v_mapping[1]
                if w == 0.0:
                    w = w_mapping[1]

            case _: raise ValueError("cont mode should be in [0, 6)")

        return v, w
        # return (0.08, 0.4) if not observation else (0.18, 0.0)  # CMA best controller
