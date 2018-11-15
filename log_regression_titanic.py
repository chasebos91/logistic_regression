import numpy as np
import matplotlib.pyplot as plt
import time



class MyLogisticReg:

    def __init__(self, gradient, regularizer, data_set, iterations, k, step):
        self.gradient = gradient

        if data_set == "mnist-train.csv":
            self.mnist_flag = True
        else: self.mnist_flag = False

        self.regularizer = float(regularizer)
        self.preprocess(data_set)
        self.k = k
        self.neta = 0
        self.num_instances = self.data_set[0:,0].size
        self.theta = np.transpose(np.full([self.num_features], 0.1, float))
        self.step = step
        self.iterations = iterations
        self.predictions = np.zeros(self.num_instances, float)
        self.hypothesis = np.zeros(self.num_features, float)
        self.cost_record = list()
        self.time_record = list()


    def train_test_split(self, split = .70):

        train_num = int(self.num_instances * split)
        train_X = self.data_set[:train_num, 1:]
        train_y = self.data_set[:train_num, 0]
        test_X = self.data_set[train_num:, 1:]
        test_y = self.data_set[train_num:, 0]

        return train_X, train_y, test_X, test_y

    def logistic_regression(self):


        train_X, train_y, test_X, test_y = self.train_test_split()
        self.fit(train_X,train_y)
        y_hat = self.predict(test_X)
        model = self.theta
        print("model accuracy:" , self.evaluate(y_hat, test_y))
        return model

    def preprocess(self, data_set):
        self.data_set = np.genfromtxt(data_set, delimiter=',')
        self.data_set = self.data_set[1:, ]
        if self.gradient == 'SGD':
            np.random.shuffle(self.data_set)

        b0_col = np.full([self.data_set.T.shape[1], 1], 1)
        self.data_set = np.hstack((self.data_set, b0_col))

        self.num_features = self.data_set[0].size - 1

        self.normalize()


    def normalize(self):
        for col in range(1, self.num_features):
            self.data_set[0:, col] = (self.data_set[0:, col] - self.data_set[0:, col].min())/ (self.data_set[0:, col].max() - self.data_set[0:, col].min())

    def k_fold_cross_validation(self, k):

        train_X = []
        train_y = []

        partition = self.num_instances / k
        next = partition
        start = 0
        score = 0

        for i in range(0,k):
            #test is what proportion of the fold?
            train_X.append(self.data_set[start:next, 1:])
            train_y.append(self.data_set[start:next, 0])
            start = next
            next += partition

        train_X_total = list(train_X)
        train_y_total = list(train_y)
        max_theta = np.zeros(self.theta.size)
        max_accuracy = 0
        l = list()

        for i in range(0,len(train_X_total)):

            test_X = train_X.pop(i)
            test_y = train_y.pop(i)
            train_X = np.vstack(train_X)
            train_y = np.vstack(train_y).flatten()
            self.fit(train_X, train_y)
            y_hat = self.predict(test_X)
            curr_accuracy = self.evaluate(y_hat, test_y)
            l.append(curr_accuracy)
            score += curr_accuracy

            if curr_accuracy > max_accuracy:
                max_accuracy = curr_accuracy
                max_theta = self.theta.copy()

            train_X = list(train_X_total)
            train_y = list(train_y_total)

            self.theta = np.transpose(np.full([self.num_features], 0.1, float))

        self.theta = max_theta.copy()
        stddev = np.std(l)
        score = score / k

        print "Average accuracy: ", score
        print "Maximum accuracy:", max_accuracy
        print "Standard dev", stddev
        return max_theta


    def calc_eta(self, X):

        y_hat = (np.dot(X, self.theta))

        return y_hat

    def calc_hypothesis(self, X):
        return self.sigmoid(self.calc_eta(X))

    def sigmoid(self, eta):
        # use scipy softplus or not?
        y_hat = 1 / (1 + np.exp(-(eta)))
        self.hypothesis = y_hat.copy()
        return y_hat

        #y_hat = np.where(eta > 30, np.multiply(Y, eta) - eta, np.multiply(Y, eta) - np.log(1 + np.exp(eta)))

    def diff(self, theta_1, theta_100):

        if np.absolute(sum(theta_100) - sum(theta_1)) < .0001:
            return True
        else: return False


    def cost(self, X, y):

        y_hat = self.calc_hypothesis(X)
        epsilon = 1e-5

        regulizer = ((self.regularizer/2) * np.square(np.linalg.norm(self.theta)))

        cost = np.mean(-y * np.log(y_hat + epsilon) - (1 - y) * np.log(1 - y_hat + epsilon)) + regulizer

        #cost = regulizer - np.sum((y * y_hat)) - np.sum(np.log((1 + np.exp(y_hat) + epsilon)))

        return cost

    def reshuffle(self,X,y):
        new_y = np.reshape(y, [len(y),1])
        data = np.hstack((new_y, X))
        np.random.shuffle(data)
        X = data[0:, 1:]
        y = data[0:, 0]
        return X, y

    def scalar_multiply(self, a, b):
        return a * b

    def GD(self, X, y):
        regulizer = ((self.regularizer)/2 * np.square(np.linalg.norm(self.theta)))
        scalar_m = np.vectorize(self.scalar_multiply)
        y_hat = self.calc_hypothesis(X)
        size = float(len(X))
        delta = np.dot (X.T, (y_hat - y)) + regulizer
        #X_thing = X.T
        #print y_hat
        #print np.dot(X_thing, (y - ((np.exp(y_hat) / (1 + np.exp(y_hat))))))
        #print regulizer
        #delta = np.dot(X_thing, (y - (np.exp(y_hat)/ (1+np.exp(y_hat))))) + regulizer
        scale = float((1.0/size))
        test = scalar_m((scale), (delta))

        return test

    def convergence_plot(self, x, y, title, label):
        plt.scatter(x, y, label = title)
        plt.xlabel(label)
        plt.ylabel("cost")
        plt.title("Objective Curve")
        plt.legend()
        plt.show()


    def fit(self, X, y):
        """fit model. this function trains model parameters with input train data X and y.
        Parameters:
        X: numpy.array, shape (n_samples, n_features)
            feature matrix of training instances
        y: numpy.array. shape (n_samples, )
            label vector of training instances

        """
        cost_record = list()
        theta_record = list()

        if self.gradient is "SGD":
            iter_val = 1
            flag = False
            prev = 0
            size = X.T.shape[1]
            remainder = size % 10
            size -= remainder
            next = 0
            start = time.time()

            for i in range(1, self.iterations):

                #if condition for diff function (calculating weights for stopping criteria)
                if flag == True:
                    break

                next += 10
                little_data = X[prev:next]
                little_test = y[prev:next]
                delta = self.GD(little_data, little_test)
                prev = next
                self.theta = self.theta - self.step * delta
                #print self.cost(X,y)
                self.time_record.append(time.time() - start)
                theta_record.append(self.theta)

                if next == size:
                    X, y = self.reshuffle(X, y)
                    next, prev = 0, 0

                #store a theta from iter 0 and from iter 100, run diff for stopping criteria for every 100 iters

                if iter_val == 1:
                    theta_1 = self.theta.copy()

                if i % 100 == 0:
                    theta_100 = self.theta.copy()
                    flag = self.diff(theta_1, theta_100)

                iter_val += 1
                cost_record.append(self.cost(X,y))


        if self.gradient is "GD":
            iter_val = 1
            flag = False
            loss_history = []
            start = time.time()
            for i in range(1, self.iterations):

                #if condition for diff function (calculating weights for stopping criteria)
                if flag == True:
                    break

                test = self.GD(X, y)

                self.theta = self.theta - self.step * test
                self.time_record.append(time.time() - start)

                theta_record.append(self.theta)

                if iter_val == 1:
                    theta_1 = self.theta.copy()

                if i % 100 == 0:
                    theta_100 = self.theta.copy()
                    flag = self.diff(theta_1, theta_100)

                iter_val += 1
                cost_record.append(self.cost(X, y))


        iter_list = range(len(cost_record))
        #self.convergence_plot(iter_list, cost_record, "Convergence by iteration", "iterations")
        #self.convergence_plot(self.time_record, cost_record, "Convergence by time", "time")

    def predict(self, X):
        """
        Predict using the logistic regression model
        :param X: np.array, shape (n_sampes, n_features)
            Test instances
        :return: y-pred: np. array, shape (n_sampes,)
            Returns prediceted values.

        """
        if self.mnist_flag == True:
            t_val = 8
            f_val = 9

        else:
            t_val = 1
            f_val = 0

        y_hat = self.calc_hypothesis(X)

        self.predictions = y_hat
        x = 0

        for pred in self.predictions:

            if pred >= .5:
                y_hat[x] = t_val
            else:
                y_hat[x] = f_val
            x += 1

        return y_hat


    def evaluate(self, y_test, y_pred):
        """evaluate the accuracy of predictions against true labels
            Params:
            y_test: np_array, shape (n_samples, )
                True labels
            y_pred: np.array, shape (n_sampes, )
            predicted labels.

            Returns:
                error_rate: a float in [0,1], the error rate of the prediciton.
            """
        error_rate = np.sum(np.equal(y_test, y_pred).astype(np.float)) / y_test.size
        return error_rate


#DRIVER

# hyperparams

titanic_iters = 100000
titanic_step = .001
titanic_k = 10
titanic_data = 'titanic_train.csv'
k = 10

GD = "GD"
SGD = "SGD"


t = MyLogisticReg(GD, 0, titanic_data , titanic_iters, titanic_k, titanic_step)
t_sgd = MyLogisticReg(SGD, 0, titanic_data , titanic_iters, titanic_k, titanic_step)
t_cross_val = MyLogisticReg(GD, 0, titanic_data , titanic_iters, titanic_k, titanic_step)
t_cross_val_sgd = MyLogisticReg(SGD, 0, titanic_data , titanic_iters, titanic_k, titanic_step)

print("Model with vanilla GD:")
t_model = t.logistic_regression()
print("Model with SGD:")
t_sgd_model = t_sgd.logistic_regression()
print("Cross Validated Model with vanilla GD:")
t_cv_model = t_cross_val.k_fold_cross_validation(k)
print("Cross Validated Model with  SGD:")
t_cv_sgd_model = t_cross_val_sgd.k_fold_cross_validation(k)





