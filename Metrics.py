import keras
import time
import matplotlib.pyplot as plt

class Metrics(keras.callbacks.Callback):

    def __init__(self, agent, env):
        keras.callbacks.Callback.__init__
        self.agent = agent
        self.env = env

    def on_train_begin(self, logs={}):
        """ This function is called at the beginning of the training phase. """
        self.metrics = {}

    def on_step_end(self, episode_step, logs):
        """ At each step end, we add the metrics in the Metric object. """
        timestamp = time.time()
        self.metrics[timestamp] = []
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            # We add the agent's metrics
            self.metrics[timestamp].append(logs.get('metrics')[ordinal])
        # Then we add our own metrics, which are the environment's reward and score
        self.metrics[timestamp].append(self.env.reward)
        self.metrics[timestamp].append(self.env.env.score)

    def __str__(self):
        """ Print the metrics. """
        result = ""
        for key in self.metrics:
            result += "%s: %s\n" % (key, self.metrics[key])
        return result

    def export_to_text(self):
        """ This function returns a csv like string with all the metrics data. """
        result = ','.join(['step'] + self.agent.metrics_names + ['reward', 'score']) + '\n'
        for i, key in enumerate(self.metrics) :
            result += str(i) + ',' + ','.join([str(x) for x in self.metrics[key]]) + '\n'
        return result

    def export_figs(self, fileName):
        """ Export the data into a figure and save it as a .png file. """
        name_metrics = self.agent.metrics_names + ['reward', 'score']
        for i, name_metric in enumerate(name_metrics):
            plt.figure()
            plt.plot([self.metrics[ts][i] for ts in self.metrics], alpha = .6)
            plt.title(fileName + ' ' + name_metric)
            plt.ylabel(name_metric)
            plt.xlabel('epoch')
            plt.savefig('./metrics/' + fileName + '_' + name_metric + '.png')
            plt.close('all')
        return
