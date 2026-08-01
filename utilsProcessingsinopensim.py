# utilsProcessing.py — versión sin OpenSim
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Filtro pasa-bajos Butterworth en sos + filtfilt
def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    fs = 1/np.round(np.mean(np.diff(time)), 16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(int(order/2), wn, btype='low', output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)

# Utilidad para leer .mot/.sto sin OpenSim
def _storage_to_dataframe_noos(storage_path, headers=None):
    with open(storage_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    # detecta línea de columnas (empieza con 'time')
    header_idx = next(i for i,l in enumerate(lines) if l.strip().split()[0].lower()=='time' or l.lower().startswith('time'))
    cols = lines[header_idx].strip().split()
    data = np.loadtxt(lines[header_idx+1:], dtype=float)
    out = {c: data[:, j] for j, c in enumerate(cols)}
    if headers is not None:
        out = {k: out[k] for k in ['time'] + list(headers)}
    return out

# Segmentación de marcha (delegada a gait_analysis)
def segment_gait(session_id, trial_name, data_folder, gait_cycles_from_end=0):
    from gait_analysissinopensim import gait_analysis
    gait = gait_analysis(os.path.join(data_folder, session_id), trial_name, n_gait_cycles=-1)
    heelstrikeTimes = gait.gaitEvents['ipsilateralTime'][gait_cycles_from_end,(0,2)].tolist()
    return heelstrikeTimes, gait

# Segmentación de sentadillas usando pelvis_ty desde IK (o .mot/.sto)
def segment_squats(ikFilePath=None, pelvis_ty=None, timeVec=None, visualize=False, filter_pelvis_ty=True, cutoff_frequency=4, height=.2):
    if pelvis_ty is None or timeVec is None:
        ik = _storage_to_dataframe_noos(ikFilePath, headers={'pelvis_ty'})
        timeVec = ik['time']
        pelvis_ty = ik['pelvis_ty']
        if filter_pelvis_ty:
            pelvis_ty = lowPassFilter(timeVec, np.asarray(pelvis_ty).reshape(-1,1), cutoff_frequency).ravel()

    dt = timeVec[1] - timeVec[0]
    pelvSignal = np.array(-pelvis_ty - np.min(-pelvis_ty))
    pelvSignalPos = np.array(pelvis_ty - np.min(pelvis_ty))
    idxMinPelvTy,_ = signal.find_peaks(pelvSignal, distance=.7/dt, height=height)

    startFinishInds, minIdxOld = [], 0
    for i, minIdx in enumerate(idxMinPelvTy):
        nextIdx = idxMinPelvTy[i+1] if i < len(idxMinPelvTy)-1 else len(pelvSignalPos)
        startIdx = np.argmax(pelvSignalPos[minIdxOld:minIdx]) + minIdxOld
        endIdx = np.argmax(pelvSignalPos[minIdx:nextIdx]) + minIdx
        startFinishInds.append([startIdx,endIdx])
        minIdxOld = int(minIdx)

    startFinishTimes = [[float(timeVec[i0]), float(timeVec[i1])] for (i0,i1) in startFinishInds]

    if visualize:
        plt.figure(); plt.plot(-pelvSignal)
        for val in startFinishInds:
            plt.plot(val, -pelvSignal[val], 'o', mfc='k', mec='none', label='Squatting phase')
        plt.xlabel('Frames'); plt.ylabel('Position [m]'); plt.title('Vertical pelvis position'); plt.draw()
    return startFinishTimes

# Segmentación Sit-to-Stand

def segment_STS(ikFilePath=None, pelvis_ty=None, timeVec=None, velSeated=0.3, velStanding=0.15, visualize=False, filter_pelvis_ty=True, cutoff_frequency=4, delay=0.1):
    if pelvis_ty is None or timeVec is None:
        ik = _storage_to_dataframe_noos(ikFilePath, headers={'pelvis_ty'})
        timeVec = ik['time']
        pelvis_ty = ik['pelvis_ty']
        if filter_pelvis_ty:
            pelvis_ty = lowPassFilter(timeVec, np.asarray(pelvis_ty).reshape(-1,1), cutoff_frequency).ravel()

    dt = timeVec[1] - timeVec[0]
    pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
    pelvVel = np.diff(pelvSignal, append=0)/dt
    idxMaxPelvTy,_ = signal.find_peaks(pelvSignal, distance=.9/dt, height=.2, prominence=.2)

    startFinishInds, maxIdxOld = [], 0
    for maxIdx in idxMaxPelvTy:
        vels = pelvVel[maxIdxOld:maxIdx]
        velPeak, peakVals = signal.find_peaks(vels, distance=.9/dt, height=.2)
        velPeak = int(velPeak[np.argmax(peakVals['peak_heights'])] + maxIdxOld)
        velsLeft = np.flip(pelvVel[maxIdxOld:velPeak])
        velsRight = pelvVel[velPeak:]
        startIdx = int(velPeak - np.argwhere(velsLeft < velSeated)[0])
        endIdx = int(velPeak + np.argwhere(velsRight < velStanding)[0])
        startFinishInds.append([startIdx, endIdx])
        maxIdxOld = int(maxIdx)

    risingTimes = [[float(timeVec[i0]), float(timeVec[i1])] for (i0,i1) in startFinishInds]

    sf = 1/np.round(np.mean(np.round(timeVec[1:] - timeVec[:-1], 2)), 16)
    startFinishIndsDelay = [[i0 + int(delay*sf), i1] for (i0,i1) in startFinishInds]
    risingTimesDelayedStart = [[float(timeVec[i0]), float(timeVec[i1])] for (i0,i1) in startFinishIndsDelay]

    startFinishIndsDelayPeriodic = []
    for i0,i1 in startFinishIndsDelay:
        pelvVal_up = pelvSignal[i0]
        val_down = int(np.argwhere(pelvSignal[i0+1:] < pelvVal_up)[0][0] + (i0+1))
        if abs(pelvSignal[val_down] - pelvVal_up) > abs(pelvSignal[val_down-1] - pelvVal_up):
            val_down -= 1
        startFinishIndsDelayPeriodic.append([i0, val_down])
    risingSittingTimesDelayedStartPeriodicEnd = [[float(timeVec[i0]), float(timeVec[i1])] for (i0,i1) in startFinishIndsDelayPeriodic]


    if visualize:
        plt.figure(); plt.plot(pelvSignal)
        for (i0,i1),(d0,_),(p0,p1) in zip(startFinishInds, startFinishIndsDelay, startFinishIndsDelayPeriodic):
            plt.plot([i0,i1], [pelvSignal[i0], pelvSignal[i1]], 'o', mfc='k', mec='none', label='Rising phase')
            plt.plot(d0, pelvSignal[d0], 'o', mfc='r', mec='none', label='Delayed start')
            plt.plot(p1, pelvSignal[p1], 'o', mfc='g', mec='none', label='Periodic end')
        plt.xlabel('Frames'); plt.ylabel('Position [m]'); plt.title('Vertical pelvis position'); plt.tight_layout(); plt.draw()


    return (risingTimes, risingTimesDelayedStart, risingSittingTimesDelayedStartPeriodicEnd)