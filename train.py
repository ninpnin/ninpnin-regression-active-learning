import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

from keras.models import Sequential
from keras import Input
from keras.layers import Dense

import tensorflow_probability as tfp
tfd = tfp.distributions

def custom_loss(y_true, y_pred):
    #print(y_pred.shape)
    mu_pred = y_pred[:,0]
    sigma_pred = tf.math.exp(y_pred[:, 1])
    #print(mu_pred)
    
    #print(sigma_pred)
    loss = 0.0
    no_of_examples = y_pred.shape[0]
    for ix in range(no_of_examples):
        dist = tfd.Normal(loc=mu_pred[ix], scale=sigma_pred[ix])
        loss += dist.log_prob(y_true[ix])
    #diff = mu_pred - y_true
    #loss = - tf.reduce_sum(tf.multiply(diff, diff))
    #print(loss)
    return - loss / no_of_examples

def define_model(args):
    model = Sequential(name="Model-with-One-Input")
    model.add(Input(shape=(1,), name='Input-Layer'))
    model.add(Dense(args.hidden, activation='relu', name='h1'))
    model.add(Dense(args.hidden, activation='relu', name='h2'))
    model.add(Dense(2, activation='linear', name='Output-Layer'))

    model.compile(optimizer='adam', loss=custom_loss, run_eagerly=True)
    return model

def main(args):
    # Generate data

    df = pd.read_csv("data.csv")


    model = define_model(args)
    print(model.summary())

    x, y = df["x"], df["y"]
    x, y = tf.constant(x), tf.constant(y)

    x_ds, y_ds = tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((x_ds, y_ds))

    dataset = dataset.shuffle(1000)
    #dataset = dataset.take(50)
    ds_len = len(dataset)
    train_ds_len = int(ds_len * 0.7)
    valid_ds_len = int(ds_len * 0.15)

    train_dataset = dataset.take(train_ds_len)
    valid_dataset = dataset.skip(train_ds_len).take(valid_ds_len)
    test_dataset = dataset.skip(train_ds_len + valid_ds_len)
    train_dataset = train_dataset.batch(args.batch_size)
    valid_dataset = valid_dataset.batch(args.batch_size)


    print(train_dataset)
    print(valid_dataset)
    #x_train, x_test, y_train, y_test = 
    #print(next(dataset))
    if args.early_stopping_patience is not None:
        es_callback = tf.keras.callbacks.EarlyStopping(patience=args.early_stopping_patience)
        model.fit(train_dataset, epochs=args.epochs, validation_data=valid_dataset, callbacks=[es_callback])
    else:
        model.fit(train_dataset, epochs=args.epochs, validation_data=valid_dataset)

    x = tf.range(-22.0, 22.0) / 11.0 * 2
    y_hat = model.predict(x)
    print(x.shape, y_hat.shape)
    mu_hat = y_hat[:, 0]
    sigma_hat = tf.math.exp(y_hat[:, 1])
    print(x.shape, y_hat.shape)


    d = {"x": x.numpy(), "mu_hat": mu_hat, "sigma_hat": sigma_hat}
    df = pd.DataFrame(d)
    df["y_up"] = df["mu_hat"] + df["sigma_hat"]
    df["y_low"] = df["mu_hat"] - df["sigma_hat"]

    fig, ax = plt.subplots()
    sns.set_theme()
    sns.lineplot(x='x', y='value', hue='variable', ax=ax,
             data=pd.melt(df[["x", "mu_hat", "y_up", "y_low"]], ['x']))

    train_dataset = train_dataset.unbatch()
    train_dataset = [(x_i,y_i) for x_i, y_i in train_dataset.as_numpy_iterator()]
    x_train = [x_i for x_i, y_i in train_dataset]
    y_train = [y_i for x_i, y_i in train_dataset]
    
    df2 = pd.DataFrame({"x": x_train, "y": y_train})
    print(df2)
    sns.regplot(data=df2, x='x', y='y', ax=ax, fit_reg=False)
    plt.show()
    plt.savefig('model.png')

if __name__ == '__main__':
    import argparse                                                                                                   
    parser = argparse.ArgumentParser()                                                                                
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-e", "--early_stopping_patience", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    main(args)