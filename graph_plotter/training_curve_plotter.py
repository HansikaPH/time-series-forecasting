import tensorflow as tf
import numpy as np

class CurvePlotter:
    def __init__(self, session, no_of_curves):
        self.__session = session
        self.__writer_train = tf.summary.FileWriter('./logs/plot_train')

        if no_of_curves == 2:
            self.__writer_val = tf.summary.FileWriter('./logs/plot_val')
        self.__loss_var = tf.Variable(0.0)
        tf.summary.scalar("loss", self.__loss_var)
        self.__write_op = tf.summary.merge_all()

    def plot_train(self, loss, epoch):
        summary = self.__session.run(self.__write_op, {self.__loss_var: np.mean(loss)})
        self.__writer_train.add_summary(summary, epoch)
        self.__writer_train.flush()

    def plot_val(self, loss, epoch):
        summary = self.__session.run(self.__write_op, {self.__loss_var: np.mean(np.mean(loss))})
        self.__writer_val.add_summary(summary, epoch)
        self.__writer_val.flush()