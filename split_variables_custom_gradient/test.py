import argparse
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3)


class Test(tf.keras.Model):
    def __init__(self):
        super(Test, self).__init__()
        self.a = tf.Variable(10.0)
        self.b = tf.Variable(20.0)

    @tf.custom_gradient
    def forward(self, x):
        w1, w2 = tf.split(x, 2)
        y1 = self.a * w1
        y2 = self.b * w2
        y = tf.concat([y1, y2], 0)

        def backward(dy, variables=None):
            print("variables: ", variables)
            dw1, dw2 = tf.split(dy, 2)
            dx1 = - self.a * dw1 / 1000000
            dx2 = - 2 * self.b * dw2
            dx = tf.concat([dx1, dx2], 0)
            grads = [tf.reshape(dx1, []), tf.reshape(dx2, [])]
            print("dx: ", dx)
            print("grads: ", grads)

            return dx, grads

        return y, backward

    def call(self, x):
        return self.forward(x)


def main(epochs: int = 10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    model = Test()

    last_y = 0
    last_a = 0
    last_b = 0
    for i in range(epochs):
        with tf.GradientTape() as tape:
            x = tf.Variable([1.0, 2.0])
            y = model(x)

        grads = tape.gradient(y, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print("epoch: ", i + 1)
        print(f"  y: {y} ({y.numpy() - last_y})")
        print(f"  a: {model.a.numpy()} ({'{:.3f}'.format(model.a.numpy() - last_a)})")
        print(f"  b: {model.b.numpy()} ({'{:.3f}'.format(model.b.numpy() - last_b)})")
        print()

        last_y = y.numpy()
        last_a = model.a.numpy()
        last_b = model.b.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10)

    args = parser.parse_args()
    main(args.epochs)
