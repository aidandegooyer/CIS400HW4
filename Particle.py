from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix


class Particle:
    def __init__(self, input_dim):
        self.q = None
        self.input_dim = input_dim
        self.model = Sequential()
        self.model.add(Dense(units=self.input_dim, activation='sigmoid', input_dim=input_dim))
        self.model.add(Dense(units=self.input_dim * 2, activation='sigmoid'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.weights = self.model.get_weights()
        self.position = self.weights  # Initial position is the current weights
        self.velocity = self._initialize_velocity()

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights
        return

    def _initialize_velocity(self):
        # Initialize velocity randomly based on the dimensionality of the weight space
        velocity = [np.random.randn(*w.shape) for w in self.weights]
        return velocity

    def update_position(self):
        # Update particle's position (weights) based on velocity
        self.position = [p + v for p, v in zip(self.position, self.velocity)]
        self.weights = self.position
        return

    def calculate_q(self, x_train, y_train, b1, b2):
        # Calculate fitness (q value) based on the particle's current position (weights)
        self.model.set_weights(self.weights)
        predictions = self.model.predict(x_train, verbose=0)
        rounded_predictions = np.rint(predictions)
        conf_matrix = confusion_matrix(y_train, rounded_predictions)
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        # Calculate q value
        self.q = (FP + FN) / (b1 + b2) + 10 * (max(0, FP / b1 - 0.1) + max(0, FN / b2 - 0.1))

    def update_velocity(self, inertia_weight, cognitive_coeff, social_coeff, best_position, global_best_position):
        # Update particle's velocity based on current velocity, best-known position, and global best position
        cognitive_velocity = [cognitive_coeff * np.random.rand(*v.shape) * (bp - p)
                              for p, v, bp in zip(self.position, self.velocity, best_position)]
        social_velocity = [social_coeff * np.random.rand(*v.shape) * (gbp - p)
                           for p, v, gbp in zip(self.position, self.velocity, global_best_position)]
        inertia_velocity = [inertia_weight * v for v in self.velocity]

        self.velocity = [iv + cv + sv for iv, cv, sv in zip(inertia_velocity, cognitive_velocity, social_velocity)]
