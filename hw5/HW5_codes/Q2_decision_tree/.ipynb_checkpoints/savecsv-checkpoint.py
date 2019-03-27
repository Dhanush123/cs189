import pandas as pd
import numpy as np
import datetime

def preds_to_csv(y,header=""):
    y = y.astype(int)
    df = pd.DataFrame({'Category': y})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv(str(datetime.datetime.now().time())+'_submission.csv', index_label='Id')
    print("saved predictions")