import matplotlib.pyplot as plt
import numpy as np

labels = ['COAT','YAHOO']
x = np.arange(len(labels))
mf_naive = [0.5952,0]
rely = [1.6064,1]
rely2 = [1.5674,2]
mf_IPS_mar = [0.6195,0]
mf_IPS_manr = [0.6230,0]
width = 0.2       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.3, mf_naive, width, label='mf_naive')
rects2 = ax.bar(x ,mf_IPS_mar,width, label='mf_IPS_MAR')
rects3 = ax.bar(x + 0.3, mf_IPS_manr, width, label='MF_IPS_MNAR')
# rects4 = ax.bar(x,rely, width,bottom = mf_IPS_mar, label='improve',color = 'r')
# rects5 = ax.bar(x + 0.3,rely2, width,bottom = mf_IPS_manr, color = 'r')
# ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
#        label='Women')

# ax.xticks(x,labels)
ax.set_ylabel('MSE')
ax.set_title('MSE from different methods')
ax.set_ylim(0,1)
ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
ax.bar_label(rects2,labels=['4.08%','0%'], padding=3)
ax.bar_label(rects3,labels=['4.67%','0%'], padding=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
fig.tight_layout()

plt.show()