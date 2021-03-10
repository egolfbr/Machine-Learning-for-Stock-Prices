import pandas as pd
import numpy as np

# path ='C:/Users/bre13/OneDrive/Desktop/amzn_edited.xlsx'
# length = 5
# percent = 0.02
# incdec = 0
def fullScript(length,percent,incdec,path):
    df = pd.read_excel(path, index_col=0, usecols=[1,2,3,4], parse_dates=True)
    df['trading'] = df['Volume']/df['Shares Outstanding']
    SMA_12 = df.iloc[:, 0].rolling(12).mean()
    SMA_12.name = 'SMA 12 days'
    SMA_26 = df.iloc[:, 0].rolling(26).mean()
    SMA_26.name = 'SMA 26 days'
    EMA_12 = SMA_12.copy()
    EMA_12.iloc[11] = np.nan
    smoothing=0.074074074
    EMA_12.iloc[12] = smoothing * df.iloc[12, 0] + (1 - smoothing) * SMA_12.iloc[11]
    for i in range(13, len(EMA_12)):
        EMA_12.iloc[i] = smoothing * df.iloc[i, 0] + (1 - smoothing) * EMA_12.iloc[i-1]
    EMA_12.name = 'EMA 12 days'
    final = pd.concat([df, SMA_12, SMA_26, EMA_12], axis=1)

    ########################################################################

    final['new'] = 0
    # 



    final = final.to_numpy()
    start_index = length
    i = start_index
    z = len(final)
    # rv = final[387][:]
    # trv = final[387-length][:]


    # print("row vector ")
    # print(rv)
    # print("target")
    # print(trv)
    # print("#################################################################")
    # print("Shape and value of the last value in the row")
    # print(final.shape)
    # print(final[0][7])
    # print("##################################################################")
    # print("values that we are comparing")
    # print(rv[0],trv[0])
    # print("##################################################################")


    # if(rv[0] >= percent*trv[0] + trv[0]):
    #     rv[7] = 1
    #     final[389][:] = rv
    # else:
    #     rv[7] = 0
    #     final[389][:] = rv
    # print("Vector output")
    # print(final[389][:])
    # print("#########################################################")
    if incdec == 1:
        for i in range(z):
            rowVector = final[i][:]
            TargetRV = final[i-length][:]
            if(rowVector[0] >= percent*TargetRV[0] + TargetRV[0]):
                rowVector[7] = 1
                final[i][:] = rowVector
            else:
                rowVector[7] = 0
                final[i][:] = rowVector
    else:
        for i in range(z):
            rowVector = final[i][:]
            TargetRV = final[i-length][:]
            if(rowVector[0] <= percent*TargetRV[0] + TargetRV[0]):
                rowVector[7] = 1
                final[i][:] = rowVector
            else:
                rowVector[7] = 0
                final[i][:] = rowVector

    df = pd.DataFrame(final)
    updatedDf = df
    # print(df.head(50))
    # print("########################################")
    updatedDf = updatedDf.dropna()
    updatedDf = updatedDf.reset_index(drop=True)
    return updatedDf
