import re
import matplotlib.pyplot as plt

# takes the printed output from training in log_file and draws learning curves
log_file = 'log.txt'

it_pattern = re.compile('iteration: (\\d+)\n')
val_pattern = re.compile('mean reward: (nan|[\\-0-9\\.]+)\n')

iterations = []
values = []
for line in open(log_file):
    m = it_pattern.match(line)
    if m is not None:
        iterations.append(int(m.group(1)))
    m = val_pattern.match(line)
    if m is not None:
        values.append(float(m.group(1)))
assert len(iterations) == len(values)

plt.plot(iterations, values)
plt.show()
