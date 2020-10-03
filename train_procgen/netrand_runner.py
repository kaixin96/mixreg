import numpy as np
from baselines.ppo2.runner import Runner

class NetRandRunner(Runner):
    def run(self):
        self.model.clean_flag = False if np.random.random() > 0.1 else True
        self.model.sess.run(self.model._init_randcnn)
        return super().run()
