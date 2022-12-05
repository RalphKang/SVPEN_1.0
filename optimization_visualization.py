import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% import recording data
data_dir='output/network_recording.csv'
optimization_recording=pd.read_csv(data_dir)
recording_np=optimization_recording.to_numpy()
#%% plot the loss
plt.figure()
plt.subplot(2,2,1)
plt.plot(recording_np[:,0],recording_np[:,1])
plt.xlabel('epoch')
plt.ylabel('temperture')
plt.subplot(2,2,2)
plt.plot(recording_np[:,0],recording_np[:,2])
plt.xlabel('epoch')
plt.ylabel('mole fraction')
plt.subplot(2,2,3)
plt.plot(recording_np[:,0],recording_np[:,3],'k')
plt.plot(recording_np[:,0],recording_np[:,4],'r')
plt.legend(['Ground truth','Prediction'])
plt.xlabel('epoch')
plt.ylabel('Error')
plt.subplot(2,2,4)
plt.plot(recording_np[:,0],recording_np[:,5],'k')
plt.plot(recording_np[:,0],recording_np[:,6],'r')
plt.xlabel('epoch')
plt.ylabel('Error Loss')
plt.tight_layout()
plt.show()

