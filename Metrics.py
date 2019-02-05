import keras
import time
import matplotlib.pyplot as plt

class Metrics(keras.callbacks.Callback):

    def __init__(self, agent, env):
        keras.callbacks.Callback.__init__
        self.agent = agent
        self.env = env

    def BAKon_train_begin(self, logs={}):
        self.metrics = {key : [] for key in self.agent.metrics_names}

    def BAKon_step_end(self, episode_step, logs):
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            self.metrics[key].append(logs.get('metrics')[ordinal])

    def on_train_begin(self, logs={}):
        self.metrics = {}

    def on_step_end(self, episode_step, logs):
        timestamp = time.time()
        self.metrics[timestamp] = []
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            print(key)
            self.metrics[timestamp].append(logs.get('metrics')[ordinal])
        self.metrics[timestamp].append(self.env.reward)
        self.metrics[timestamp].append(self.env.env.score)

    def __str__(self):
        result = ""
        for key in self.metrics:
            result += "%s: %s\n" % (key, self.metrics[key])
        return result

    def export_figs(self):
        name_metrics = self.agent.metrics_names + ['reward' + 'score']
        for i, name_metric in enumerate(name_metrics):
            plt.figure()
            plt.plot([self.metrics[ts][i] for ts in self.metrics], alpha = .6)
            plt.title('model ' + name_metric)
            plt.ylabel(name_metric)
            plt.xlabel('epoch')
            #plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(name_metric + '.png')
        return
