filename = 'boneloss_test.csv'

with open(filename,'r') as f:
    lines = f.readlines()
lines = [line.replace(' ','_') for line in lines]
with open(filename,'w') as f:
    f.writelines(lines)
