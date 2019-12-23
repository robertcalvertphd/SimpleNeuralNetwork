import numpy as np
import random


class NeuralNetwork:

    def __init__(self, number_of_inputs):
        np.random.seed(random.randint(1, 100))
        #set random weights for each neuron, values between -2 and 2.
        self.synaptic_weights = 2 * np.random.random((number_of_inputs+1, 1)) - 1

    @staticmethod
    def sigmoid(x):
        # there are other activation functions that may be used
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            learning_rate = .0001  # calibrate for accuracy so as to avoid over correcting when minimizing error
            output = self.apply_weights(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * NeuralNetwork.sigmoid_derivative(output)) * learning_rate
            #np.dot of (a,b), (c,d) would be ac + bd.
            self.synaptic_weights += adjustments

    def apply_weights(self, inputs):
        inputs = inputs.astype(float)
        output = NeuralNetwork.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


class TestObject:
    #generate random objects which follow an arbitrary rule pattern defined below.
    def __init__(self, rules):
        self.a = random.randint(0, 1)
        self.b = random.randint(0, 1)
        self.c = random.randint(0, 1)
        self.d = random.randint(0, 1)

        if rules == 0:
            self.score1 = self.applyRules0()
        elif rules == 1:
            self.score1 = self.applyRules1()
        elif rules == 2:
            self.score1 = self.applyRules2()
        elif rules == 3:
            self.score1 = self.applyRules3()
        elif rules == 4:
            self.score1 = self.applyRules4()
        elif rules == 5:
            self.score1 = self.applyRules5()
        else:
            print("error: rule set not defined")

    def applyRules0(self):
        if self.a:
            return 0
        if self.b:
            return 1
        if self.c:
            return 0
        if self.d:
            return 1
        return random.randint(0, 1)

    def applyRules1(self):
        score = 0
        if self.a:
            score += random.randint(5, 10)
        if self.b:
            score -= random.randint(5, 10)
        if self.c:
            score += random.randint(3, 5)
        if self.d:
            score -= random.randint(3, 5)
        if score > 0:
            return 1
        return 0

    def applyRules2(self):
        if self.a == 1 and self.b == 1:
            return 1
        if self.a == 1 and self.c == 1:
            return 0
        if self.b == 0 and self.c == 1:
            return 0
        if self.d == 1:
            return 1
        return 0

    def applyRules3(self):
        if self.a + self.b + self.c > 1:
            return 1
        if self.d == 0:
            return 1
        return 0

    def applyRules4(self):
        score = self.a * 10 - self.b * 10 + self.c * 5
        if score > 0:
            score = random.randint(1, score) + int(score / 2)
            if score > 5:
                return 1
        return 0

    def applyRules5(self):
        score = 0
        self.a = random.randint(1, 10)
        self.b = random.randint(1, 10)
        self.c = random.randint(1, 10)
        self.d = random.randint(1, 5)
        for i in range(self.d * 5):
            if random.randint(1, self.a + 1) == 1:
                score += 6
            if random.randint(1, self.b + 1) == 1:
                score += 4
            if self.c > 5:
                if random.randint(1, self.d + 1) == 1:
                    score -= 2
                else:
                    score += 1
        return score

    def getArray(self):
        return [self.a, self.b, self.c, self.d, 1]


def createInputAndOutputSets(n, rule):
    inputs = []
    outputs = []
    for i in range(n):
        t = TestObject(rule)
        inputs.append(t.getArray())
        outputs.append(t.score1)
    return [np.array(inputs), np.array([outputs]).T]  # .T transforms from many cols to many rows.


def trainNN(neural_network, rule):
    number_of_cases = 1000
    training_iterations = 10000
    data = createInputAndOutputSets(number_of_cases, rule)  # creates data from rule for both inputs and outputs
    training_inputs = data[0]  # data returns inputs then outputs, thus 0 then 1.
    training_outputs = data[1]
    neural_network.train(training_inputs, training_outputs, training_iterations)


def evaluate_neural_network(neural_network, rules):
    #applies weights from trained neural network to new data set and then calculates accuracy.
    correct = 0
    trials = 10000
    for i in range(trials):
        t = TestObject(rules)
        prediction = 0
        if neural_network.apply_weights(np.array(t.getArray())) > .5:
            prediction = 1
        if t.score1 == prediction:
            correct += 1

    print("rules:", rules, "correct:", str(round(correct / trials, 4) * 100) + "%")


def run():
    for i in range(5):
        neural_network = NeuralNetwork(4)
        trainNN(neural_network, i)
        evaluate_neural_network(neural_network, i)


run()
