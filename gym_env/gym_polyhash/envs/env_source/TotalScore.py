import sys
import os

if __name__ == '__main__':
    folder = sys.argv[1]

    score = 0

    for file in os.listdir(folder):
        if file.endswith('.score'):
            with open(os.path.join(folder,file),'r') as handle:
                line=handle.readline()
                score += int(line.split('/')[0])

    print('Total score:', score)