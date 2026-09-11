import numpy as np

class DriverNetAgent:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def get_action(self, stacked_obs):
        model_input = np.expand_dims(stacked_obs, axis=0)
        outputs = self.session.run(["continuous_actions"], {self.input_name: model_input})
        
        return outputs[0][0]