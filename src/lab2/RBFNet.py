from RBF import *
import numpy as np

def absolute_residual_error(out, T):
    return np.mean(np.absolute(out - T))

class RBFNet:
    def __init__(self, rbf, num_of_outputs=1):
        self.rbf = rbf
        self.num_of_outputs = num_of_outputs
        
        #self.W = np.random.random((rbf.get_output_dim(), self.num_of_outputs)
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(1/(rbf.get_output_dim())), size=(rbf.get_output_dim(), self.num_of_outputs))
    
    def forward(self, X, get_phi=False):
        phi = self.rbf.phi(X)
        if get_phi: return phi, phi @ self.W
        else: return phi @ self.W
    
    def set_weights(self, new_W):
        self.W = new_W
    
    def train(self, patterns, targets):
        phi = self.rbf.phi(patterns)
        least_sq_result = np.linalg.pinv(phi.T @ phi) @ phi.T @ targets
        self.set_weights(least_sq_result)
    
    def delta_step(self, pattern, target, eta):
        phi, out = self.forward(pattern, get_phi=True)
        phi = phi[0]
        diff = target - out
        res = []
        for d in diff[0]: res.append(eta * d * phi)
        self.W += np.stack(res).T
    
    def train_online(self, patterns, targets, eta):
        num_of_epochs = 0
        last_err = 9e999
        while True:
            pattern_dim = patterns.shape[1]
            target_dim = targets.shape[1]
            patterns_and_targets = np.append(patterns, targets, axis=1)
            np.random.shuffle(patterns_and_targets)
            patterns = patterns_and_targets[:,0:pattern_dim]
            targets = patterns_and_targets[:,pattern_dim:]
            # if num_of_epochs > 27: eta = 0.01
            # if num_of_epochs > 51: eta = 0.001
            for p, t in zip(patterns, targets):
                self.delta_step(np.reshape(p, (1,-1)), np.reshape(t, (1,-1)), eta)
            num_of_epochs += 1
            pred = self.forward(patterns)
            err = absolute_residual_error(pred, targets)
            if (last_err - err) < -0.5: raise Exception("eta too high, error diverging!!!")
            if (last_err - err) < 7e-7 or num_of_epochs > 100: return num_of_epochs, err
            last_err = err


        
        # total = patterns_and_targets.shape[1]
        # num_for_training = int(total * 0.8)
        # training_data = patterns_and_targets.T[0:num_for_training].T
        # validation_data = patterns_and_targets.T[num_for_training:].T
        # input_train = training_data[0:2]
        # T_train = training_data[2:]
        # input_validation = validation_data[0:2]
        # T_validation = validation_data[2:]
            




