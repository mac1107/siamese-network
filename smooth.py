import pandas as pd
import numpy as np
import os
def smooth(csv_path,weight=0.98):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        if pd.isna(point):
            point=0
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val


    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    save.to_csv('smooth98_'+csv_path)


if __name__=='__main__':
    smooth('loss_batch10_lr0.001.csv')

