import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Binarizer

def load_dataset():
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")
    test_passenger_id = test_data["PassengerId"]

    #Drop unnecessary collumns
    test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis = 1, inplace=True)
    train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis = 1, inplace=True)

    #Fill in NaN values
    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
    test_data["Embarked"] = train_data["Embarked"].fillna("S")
    test_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())

    #Convert sex to 1/0 value
    train_data = sex_to_int(train_data)
    test_data = sex_to_int(test_data)

    #Normalize values
    columns = ["Age"]
    train_data = normalize_age(train_data, columns)
    test_data = normalize_age(test_data, columns)

    #Convert embarked into one hot encoded columns
    columns = ["Embarked", "Pclass"]
    prefixes = ["Port", "Pclass"]
    train_data = dummy_data(train_data, columns, prefixes)
    test_data = dummy_data(test_data, columns, prefixes)
    return test_data, train_data, test_passenger_id

def sex_to_int(data):
    data['Sex'].replace('female', 0, inplace=True)
    data['Sex'].replace('male', 1, inplace=True)
    return data

#Encodes data in collumns to 0/1 collumns
def dummy_data(data, columns, prefixes):
    for column, prefix in zip(columns, prefixes):
        one_hot = pd.get_dummies(data[column], prefix = prefix)
        data = data.drop(column, axis = 1)
        data = data.join(one_hot)
    return data

def normalize_age(data, columns):
    for column in columns:
        max_value = data[column].max()
        min_value = data[column].min()
        data[column] = (data[column] - min_value) / (max_value - min_value)
    return data
    



def split_data(data, fraction = .8):

    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=1-fraction, random_state = 42)

    return train_x, valid_x, train_y, valid_y



def get_batch(data_x, data_y, batch_size = 32):
    batch_n=len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i*batch_size:(i+1)*batch_size]
        batch_y = data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y

def model(X_train, Y_train, X_valid, Y_valid, X_test, hidden_layer_size, 
            learning_rate = 0.0001, num_epochs = 200, minibatch_size = 32, lambd = 0.01,
            print_cost = True, keep_p = 1):
    
    
    tf.reset_default_graph() # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    my_seed = 1
    (m, n_x) = X_train.shape 
    n_y = Y_train.shape[1]                          
    costs = [] 
    weights = []                                     
    
    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, n_x])
    Y = tf.placeholder(tf.float32, [None, n_y])

    regularizer = tf.contrib.layers.l2_regularizer(lambd)


    #Forward propagation
    prev_layer = X
    for i in hidden_layer_size:
        layer = tf.contrib.layers.fully_connected(inputs = prev_layer, num_outputs = i, activation_fn = tf.nn.relu, weights_regularizer = regularizer)
        layer = tf.nn.dropout(layer, keep_prob, seed = my_seed)

        #TODO test this
        prev_layer = batch_norm = tf.contrib.layers.batch_norm(layer, activation_fn = tf.nn.relu)

    out_layer = tf.contrib.layers.fully_connected(inputs = prev_layer, num_outputs = 1, activation_fn = None, weights_regularizer = regularizer)
    out_layer = tf.cast(out_layer, tf.float32)

    #Back propagation
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = out_layer, labels = Y)) + lambd*tf.nn.l2_loss(out_layer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0. # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)

            for minibatch_X, minibatch_Y in get_batch(X_train, Y_train, minibatch_size):
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: keep_p})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 20 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        # lets save the parameters in a variable
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        predicted = tf.nn.sigmoid(out_layer)
        correct_pred = tf.equal(tf.round(predicted), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1}))
        print ("Test Accuracy:", accuracy.eval({X: X_valid, Y: Y_valid, keep_prob: 1}))

        predictions = sess.run(predicted, feed_dict={X: X_test, keep_prob: 1})
        predictions = np.nan_to_num(predictions) > .5
        return predictions
        



test_data, train_data, test_passenger_id = load_dataset()

train_x, valid_x, train_y, valid_y = split_data(train_data, fraction = 1)

hidden_layer_size = [16, 8]

predictions = model(train_x, train_y, valid_x, valid_y, test_data, hidden_layer_size, learning_rate = 0.01, lambd = 0.01, keep_p = .5)
predictions = predictions.astype(int)

evaluation = test_passenger_id.to_frame()
evaluation["Survived"] = predictions

evaluation.to_csv("submission.csv",index=False)
