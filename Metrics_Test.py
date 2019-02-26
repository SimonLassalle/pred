import keras
import time
import matplotlib.pyplot as plt

class Metrics(keras.callbacks.Callback):

    name_metrics = ['reward']

    def __init__(self, agent, env):
        keras.callbacks.Callback.__init__
        self.agent = agent
        self.env = env
        self.metrics = {}

    def on_step_end(self, episode_step, logs):
        timestamp = time.time()
        self.metrics[timestamp] = []
        print(logs)
        for name_metric in Metrics.name_metrics:
            self.metrics[timestamp].append(logs[name_metric])

    def export_to_text(self):
        result = ','.join(Metrics.name_metrics) + '\n'
        for i, key in enumerate(self.metrics) :
            result += str(i) + ',' + str(self.metrics[key]) + '\n'
        return result

    def cumulated_reward(self):
        sum = 0
        tabl = []
        for i, timestamp in enumerate(self.metrics.keys()) :
            sum += self.metrics[timestamp][0]
            tabl.append(sum)
        return tabl

    def export_figs(self, fileName):
        for i, name_metric in enumerate(Metrics.name_metrics):
            plt.figure()
            plt.plot([self.metrics[ts][i] for ts in self.metrics], alpha = .6)
            plt.title(fileName + ' ' + name_metric)
            plt.ylabel(name_metric)
            plt.xlabel('epoch')
            #plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('./output/' + fileName + '_' + name_metric + '.png')
        return
