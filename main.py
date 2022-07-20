import tensorflow as tf
from keras.utils import normalize
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from wandb.keras import WandbCallback

from wandb import init

init(project="Handwriting Recognition")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

model = tf.keras.models.Sequential(
    [
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=36, kernel_size=(5, 5), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"],
)
model.fit(x_train, y_train, epochs=3, callbacks=[WandbCallback()])
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint/weights.ckpt", save_weights_only=True, save_best_only=True
)
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[cp_callback],
)

model.save("convolutional.h5")
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
