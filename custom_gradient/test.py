import tensorflow as tf
import argparse


class Test(tf.keras.Model):
    """
    custom_gradient のテスト
    backward で負の値を指定することで、伝搬した際に誤差を大きくしてみる
    """

    def __init__(self):
        super(Test, self).__init__()
        self.a = tf.Variable(1.0)
        self.b = tf.Variable(2.0)

    @tf.custom_gradient
    def forward(self, x):
        y = self.a * x + self.b

        def backward(dy, variables=None):
            print("variables: ", variables)
            dx = - self.a * dy
            return dx, [dx, 0]

        return y, backward

    def call(self, x):
        return self.forward(x)


def main(epochs: int = 10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = Test()

    last_y = 0
    last_a = 0
    last_b = 0
    for i in range(epochs):
        with tf.GradientTape() as tape:
            x = tf.Variable(1.0)
            y = model(x)

        grads = tape.gradient(y, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print("epoch: ", i + 1)
        print(f"  y: {y} ({'{:.3f}'.format(y - last_y)})")
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
