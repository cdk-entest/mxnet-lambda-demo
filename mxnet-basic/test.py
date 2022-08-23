import matplotlib.pyplot as plt 

fig,axes = plt.subplots(1,1,figsize=(10,5))
axes.plot([1,2,3,4,5])
plt.savefig('test.png')

