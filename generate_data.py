import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns

# Leaky linearized tanh
@tf.function
def mean(x, threshold=1.0, a=2.0, b=0.1):
    a_prime = a / threshold
    x_pos = tf.abs(x)
    
    y = tf.math.minimum(threshold, x_pos) * a_prime

    if x_pos > threshold:
        x_remainder = x_pos - threshold
        y += x_remainder * b

    if x < 0:
        y = -y

    return y

def main(args):
    # Generate data
    x = tf.range(-200.0, 200.0) / 50.0
    mu = tf.map_fn(mean, x)
    y = tf.random.normal(shape=[len(mu)],mean=0.0, stddev=args.sigma) + mu

    # Create dataframe
    d = {"x": x.numpy(), "y": y.numpy()}
    df = pd.DataFrame(d)
    print(df)

    # Plot data
    sns.set_theme()
    sns.relplot(                                                                                                      
        data=df,
        x="x", y="y", kind="scatter",
        facet_kws=dict(sharex=False),
    )
    df.to_csv("data.csv", index=False)
    plt.show()

if __name__ == '__main__':
    import argparse                                                                                                   
    parser = argparse.ArgumentParser()                                                                                
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    main(args)