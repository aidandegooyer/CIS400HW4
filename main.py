import json

from individual import Individual
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# HYPERPARAMETERS
importdata = True
randomrelabel = False
generations = 260
num_experiments = 10
population_size = 22
num_parents = 4
mutation_rate = 0.5

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
        pop_list.append(Individual(input_dim))
        pop_list[i].get_random_weights()
        pop_list[i].calculate_q(X_train, y_train, b1, b2)
    pop_list.sort(key=lambda a: a.q)  # sorts population list by q value
    return pop_list


# evolve the population, returns a new generation sorted by q
def evolve_population(population, mutation_rate):
    offspring = []
    for j in range(population_size):
        parent1 = population[np.random.randint(0, high=4)].get_weights()  # get one of the best 4 parents
        parent2 = population[np.random.randint(0, high=4)].get_weights()
        child = []
        for i in range(len(parent1)):
            temp = np.mean(np.array([parent1[i], parent2[i]]), axis=0)  # average the parents weights and add  to child
            child.append(temp)
        offspring.append(Individual(input_dim))
        offspring[j].set_weights(child)
        offspring[j].mutate(mutation_rate)  # mutate child after recombination
        offspring[j].calculate_q(X_train, y_train, b1, b2)  # run calculations to give child's q
    # the above line could be run on the whole list in parallel to improve performance. take it out of the loop maybe?

    offspring.sort(key=lambda a: a.q)
    return offspring


def evaluate_model_conf_matrix(x_test, y_test, model):
    # not really useful except when needing confusion matricies for a report
    current_model = Sequential()
    current_model.add(Dense(units=input_dim, activation='sigmoid', input_dim=input_dim))
    current_model.add(Dense(units=input_dim * 2, activation='sigmoid'))
    current_model.add(Dense(units=1, activation='sigmoid'))
    current_model.set_weights(model)
    predictions = current_model.predict(x_test)
    rounded_predictions = np.rint(predictions)
    conf_matrix = confusion_matrix(y_test, rounded_predictions)  # needs help
    return conf_matrix


def evolutionary_strategies(X_train, y_train, mutation_rate, num_generations):
    best_q_values_trial = []
    best_q_values_generations = {1: [], 2: [], 4: [], 8: [], 16: [], 32: [], 64: [], 128: [], 256: []}
    models_with_best_q = [0] * num_experiments
    last_best_q = 9

    for experiment_num in range(num_experiments):
        population = initialize_population(population_size, input_dim)
        best_q_value_trial = 999999999

        for generation in range(num_generations):
            generational_q = 9999999
            offspring = evolve_population(population, mutation_rate)

            for individual in offspring:
                if individual.q < generational_q:
                    generational_q = individual.q

                if individual.q < best_q_value_trial:
                    best_q_value_trial = individual.q
                    print(
                        f"**********************************************NEW BEST Q: {individual.q}**********************************************")
                    models_with_best_q[experiment_num] = individual

            if generation in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                best_q_values_generations[generation].append(best_q_value_trial)

            population = offspring

            # Adapt mutation rate using the modified 1/5 rule
            if generation % 5 == 0:
                if abs(generational_q - last_best_q) < 1:
                    mutation_rate *= 2
                else:
                    mutation_rate *= 0.8
                last_best_q = generational_q
            if mutation_rate > 10: mutation_rate /= 10
            print(
                f"**********************************************LAST MUTATION EVAL BEST Q = {last_best_q}**********************************************")
            print(
                f"**********************************************THIS GENERATION BEST Q = {generational_q}**********************************************")
            print(
                f"**********************************************MUTATION RATE = {mutation_rate}**********************************************")
            print(
                f"**********************************************GENERATION NUMBER {generation}**********************************************")

        best_q_values_trial.append(best_q_value_trial)
        print(
            f"**********************************************EXPERIMENT NUMBER {experiment_num + 1} COMPLETE**********************************************")

    return best_q_values_trial, best_q_values_generations, models_with_best_q


population_size = input_dim
best_qs_trial, q_selected_gens, models_with_best_q = evolutionary_strategies(X_train, y_train, mutation_rate,
                                                                             generations)

file = open("output.txt", "w")

for model, q in models_with_best_q:
    conf_matrix = evaluate_model_conf_matrix(X_train, y_train, model)
    file.write(f"\n{conf_matrix}\n{q}\n")

file.write("---------------------")

for item in best_qs_trial:
    file.write(f"\n{item}\n")

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
