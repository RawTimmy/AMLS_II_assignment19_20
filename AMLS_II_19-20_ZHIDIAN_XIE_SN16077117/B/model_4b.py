import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt

class model_4b(object):
    def __init__(self):
        vocab_size = 20000

        model = models.Sequential()

        model.add(layers.Embedding(vocab_size, 200))
        model.add(layers.Dropout(0.1))

        # model.add(layers.LSTM(32)) # For comparison purpose only
        model.add(layers.Bidirectional(layers.LSTM(32)))
        model.add(layers.Dropout(0.1))

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

        self.model = model

class utils_4b(object):
    def __init__(self):
        self.lstm = model_4b()

    def process_4b(self, train_X, train_Y, test_X, test_Y):

        num_epochs, val_split = 10, 0.2

        # # Optional: Function to plot learning curves
        # def plot_curve(title, train_val, train_loss, val_val, val_loss, num_epoch, axes=None, ylim=None):
        #     num = np.linspace(1, num_epoch, num_epoch)
        #     axes[0].set_title(title)
        #     if ylim is not None:
        #         axes[0].set_ylim(*ylim)
        #     axes[0].set_xticks(num)
        #     axes[0].set_xlabel("Number of epochs")
        #     axes[0].set_ylabel("Score")
        #
        #     axes[0].grid()
        #     axes[0].plot(num, train_val, 'o-', color="r",
        #                  label="Training score")
        #     axes[0].plot(num, val_val, 'o-', color="g",
        #                  label="Cross-validation score")
        #     axes[0].legend(loc="best")
        #
        #     axes[1].grid()
        #     axes[1].plot(num, train_loss, 'o-', color="r",
        #                  label="Training loss")
        #     axes[1].plot(num, val_loss, 'o-', color="g",
        #                  label="Cross-validation loss")
        #     axes[1].legend(loc="best")
        #     axes[1].set_title("Training Loss v.s. Validation Loss")
        #     axes[1].set_xticks(num)
        #     axes[1].set_xlabel("Number of epochs")
        #     axes[1].set_ylabel("Loss")
        #
        #     plt.show()
        #     return plt

        self.lstm.model.compile(optimizer=optimizers.Adam(1e-4),
                          loss=losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])

        hist = self.lstm.model.fit(np.array(train_X), np.array(train_Y), epochs=num_epochs, validation_split=val_split, shuffle=True)

        # # Optional: Plot learning curves
        # fig, axes = plt.subplots(1,2, figsize=(20,5))
        # plot_curve("Learning Curve(Bi-LSTM)",
        #            hist.history['accuracy'],
        #            hist.history['loss'],
        #            hist.history['val_accuracy'],
        #            hist.history['val_loss'],
        #            num_epochs,
        #            axes, ylim=(0.0, 1.01))

        train_loss, train_acc = self.lstm.model.evaluate(np.array(train_X), np.array(train_Y), verbose=0)
        test_loss, test_acc = self.lstm.model.evaluate(np.array(test_X), np.array(test_Y),verbose=0)

        return train_acc, test_acc
