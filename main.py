import json
from individual import Individual
from Particle import Particle
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense

# HYPERPARAMETERS
importdata = True
randomrelabel = False
generations = 260
num_experiments = 10
population_size = 22
num_parents = 4
mutation_rate = 0.5

start_time = time.time()
# shouldn't be used other than for testing
# input_dim = 20


# AIDAN DEGOOYER - JAN 28, 2024
#
# Works Cited:
# DATASET: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
#
# https://keras.io/guides/training_with_built_in_methods/
# https://pandas.pydata.org/docs/reference/frame.html#dataframe
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
# https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
# https://www.javatpoint.com/how-to-add-two-lists-in-python
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#
# Debugging: https://datascience.stackexchange.com/questions/67047/loss-being-outputed-as-nan-in-keras-rnn
#            https://stackoverflow.com/questions/49135929/keras-binary-classification-sigmoid-activation-function
#


# data parsing begins here======================================================================================
# Load data from csv
if importdata:
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Remove id and number
    train_data = train_data.iloc[:, 2:]
    test_data = test_data.iloc[:, 2:]

    # Input missing values with the mean
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    # set number of data points from each class
    b1 = 100
    b2 = 300

    class_0_train = train_data[train_data['label'] == 0]
    class_1_train = train_data[train_data['label'] == 1]


    def resample_data():
        global train_class_0
        global train_class_1
        global train_data
        train_class_0 = resample(class_0_train, n_samples=b1, random_state=42)
        train_class_1 = resample(class_1_train, n_samples=b2, random_state=42)
        train_data = pd.concat([train_class_0, train_class_1])
        global X_train
        global y_train
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values


    resample_data()

    # Sample b1 data points from one class and b2 from the other
    if randomrelabel:
        # Randomly relabel 5% of the training data from each class
        num_samples_class_0_relabel = int(0.05 * len(train_class_0))
        num_samples_class_1_relabel = int(0.05 * len(train_class_1))

        selected_indices_class_0_relabel = np.random.choice(train_class_0.index, size=num_samples_class_0_relabel,
                                                            replace=False)
        selected_indices_class_1_relabel = np.random.choice(train_class_1.index, size=num_samples_class_1_relabel,
                                                            replace=False)

        train_data.loc[selected_indices_class_0_relabel, 'label'] = 1
        train_data.loc[selected_indices_class_1_relabel, 'label'] = 0

        # Concatenate the relabeled data with the original data
        train_data = pd.concat([train_class_0, train_class_1])

    # Separate testing data into two classes based on labels
    class_0_test = test_data[test_data['label'] == 0]
    class_1_test = test_data[test_data['label'] == 1]

    # Sample 100 data points from each class for testing
    test_class_0 = class_0_test.sample(n=100, random_state=42)
    test_class_1 = class_1_test.sample(n=100, random_state=42)
    test_data = pd.concat([test_class_0, test_class_1])

    # Split features and labels for training and testing sets
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]


# END data parsing ================================================================================================


# initialize population. Instantiates pop_size number of individuals, and calculates q from the individual.

def initialize_population(pop_size: int, input_dim):
    pop_list = []
    for i in range(pop_size):
        pop_list.append(Particle(input_dim))
        pop_list[i].calculate_q(X_train, y_train, b1, b2)
    pop_list.sort(key=lambda a: a.q)  # sorts population list by q value
    return pop_list


def update_global_best_position(population):
    # Update global best position based on the best particle's position
    global_best_position = None
    best_fitness = float('inf')
    for particle in population:
        if particle.q < best_fitness:
            best_fitness = particle.q
            global_best_position = particle.position
    return global_best_position

def evaluate_model_conf_matrix(x_test, y_test, model):
    # not really useful except when needing confusion matricies for a report
    current_model = Sequential()
    current_model.add(Dense(units=input_dim, activation='sigmoid', input_dim=input_dim))
    current_model.add(Dense(units=input_dim * 2, activation='sigmoid'))
    current_model.add(Dense(units=1, activation='sigmoid'))
    current_model.set_weights(model.get_weights())
    predictions = current_model.predict(x_test)
    rounded_predictions = np.rint(predictions)
    conf_matrix = confusion_matrix(y_test, rounded_predictions)  # needs help
    return conf_matrix


def pso_algorithm(x_train, y_train, b1, b2, input_dim, population_size, num_generations):
    best_q_values_trial = []
    best_q_values_generations = {1: [], 2: [], 4: [], 8: [], 16: [], 32: [], 64: [], 128: [], 256: []}
    models_with_best_q = [0] * num_experiments
    last_best_q = 9
    test_q_vals_per_generation = [0] * num_generations
    testModel = Particle(input_dim)
    # Initialize global best position


    # PSO parameters
    inertia_weight = 0.7
    cognitive_coeff = 1.5
    social_coeff = 2.0


    # Main PSO loop
    for experiment_num in range(num_experiments):
        best_q_value_trial = 999999999
        population = initialize_population(population_size, input_dim)
        global_best_position = update_global_best_position(population)


        for generation in range(num_generations):
            generational_q = 9999999
            print(f"Generation {generation + 1}/{num_generations}")

            # Update each particle's velocity and position
            for particle in population:
                particle.update_velocity(inertia_weight, cognitive_coeff, social_coeff,
                                          particle.position, global_best_position)
                particle.update_position()
                particle.calculate_q(x_train, y_train, b1, b2)
                if particle.q < generational_q:
                    generational_q = particle.q
                    generational_model = particle

                if particle.q < best_q_value_trial:
                    best_q_value_trial = particle.q
                    print(f"\n****NEW BEST Q: {particle.q:.2f}****\n\n----------------------------------------------")
                    models_with_best_q[experiment_num] = particle

            if generation in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                best_q_values_generations[generation].append(best_q_value_trial)

            testModel.set_weights(generational_model.get_weights())
            testModel.calculate_q(X_test, y_test, 100, 100)
            test_q_vals_per_generation[generation] += testModel.q

            print(f"THIS GENERATION BEST Q = {generational_q:.2f}")
            print(f"GENERATION NUMBER {generation}")
            print(f"**TEST Q VAL {testModel.q}")
            print(f"EXPERIMENT NUMBER {experiment_num + 1}\n\n----------------------------------------------")
            # Update global best position
            global_best_position = update_global_best_position(population)

            # Print best fitness value of the current generation
            best_fitness = min(p.q for p in population)
            print(f"Best Fitness: {population[0].q}")


    return best_q_values_trial, best_q_values_generations, models_with_best_q, test_q_vals_per_generation



population_size = 30
best_qs_trial, q_selected_gens, models_with_best_q, test_q_vals = pso_algorithm(X_train, y_train, b1,
                                                                             b2, input_dim, population_size, generations)
print("--- %s seconds ---" % (time.time() - start_time))

file = open("output.txt", "w")


for i, model in enumerate(models_with_best_q):
    file.write((f"Experiment #{i+1} - Train"))
    conf_matrix = evaluate_model_conf_matrix(X_train, y_train, model)
    file.write(f"\n{conf_matrix}\n")

    file.write((f"Experiment #{i + 1} - Test"))
    conf_matrix = evaluate_model_conf_matrix(X_test, y_test, model)
    file.write(f"\n{conf_matrix}\n")


file.write("---------------------\nbest q's in trials\n")

for item in best_qs_trial:
    file.write(f"\n{item}\n")

file.write("---------------------\naverage test q's per generation\n")

outList = [item / num_experiments for item in test_q_vals]
for item in outList:
    file.write(str(item))
    file.write("\n")

file.write("---------------------\n")

file.write(json.dumps(q_selected_gens))

# # Display confusion matrices
# for i in range(num_experiments):
#     print(f"Confusion Matrix - Training Data (Experiment {i + 1}):")
#     print(confusion_matrices_train[i])
#     print(f"Confusion Matrix - Test Data (Experiment {i + 1}):")
#     print(confusion_matrices_test[i])
#
# # averaging q values per epoch
# q_train_final = []
# for x in q_train_avg:
#     q_train_final.append(x / num_experiments)
# q_test_final = []
# for x in q_test_avg:
#     q_test_final.append(x / num_experiments)
#
# plot the graph using matplotlib
#
# plt.figure(figsize=(10, 6))
# plt.plot(np.log(range(1, generations + 1)), q_train_final, label='Training Data')
# plt.plot(np.log(range(1, generations + 1)), q_test_final, label='Test Data')
# plt.xlabel('log(Number of Weight Updates)')
# plt.ylabel('Average q')
# plt.title('Average q vs. log(Number of Weight Updates)')
# plt.legend()
# plt.show()
#
