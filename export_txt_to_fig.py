import matplotlib.pyplot as plt
import pandas

fileName = '1D_advanced_Sequential1000_BoltzmannQ_10000steps(0)'
csv = pandas.read_csv('./output/' + fileName + '.csv')

plt.plot(csv['step'], csv['loss'])
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.savefig('./metrics/' + fileName + '.png')
