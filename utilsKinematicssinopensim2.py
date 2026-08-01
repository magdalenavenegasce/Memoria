# utilsKinematics.py — backend sin OpenSim, interfaz compatible
import os, copy
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.transform import Rotation


from utilsProcessingsinopensim import lowPassFilter
from utilsTRCsinopensim import trc_2_dict

class kinematics:
    def __init__(self, sessionDir, trialName, modelName=None, lowpass_cutoff_frequency_for_coordinate_values=-1):
        self.lowpass_cutoff_frequency_for_coordinate_values = lowpass_cutoff_frequency_for_coordinate_values

        # ===== Cargar .mot (coordenadas) =====
        motionPath = os.path.join(sessionDir, 'OpenSimData', 'Kinematics', f'{trialName}.mot')
        if not os.path.exists(motionPath):
            raise FileNotFoundError(f'No existe archivo de cinemática: {motionPath}')
        with open(motionPath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        header_idx = next(i for i,l in enumerate(lines) if l.strip().lower().startswith('time'))
        cols = lines[header_idx].strip().split()
        data = np.loadtxt(lines[header_idx+1:], dtype=float)
        df = pd.DataFrame(data, columns=cols)

        self.time = df['time'].to_numpy()
        self.columnLabels = [c for c in df.columns if c != 'time']

        # Heurística: DOFs rotacionales (no *_tx/_ty/_tz) vienen en grados
        rot_mask = np.array([not (('_tx' in c) or ('_ty' in c) or ('_tz' in c)) for c in self.columnLabels])
        Qs = df[self.columnLabels].to_numpy().copy()
        Qs[:, rot_mask] = np.deg2rad(Qs[:, rot_mask])

        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency_for_coordinate_values)
        self.Qs = Qs

        # Derivadas numéricas con splines (suaves y robustas)
        self.Qds = np.zeros_like(self.Qs)
        self.Qdds = np.zeros_like(self.Qs)
        for i in range(self.Qs.shape[1]):
            spl = InterpolatedUnivariateSpline(self.time, self.Qs[:, i], k=3)
            self.Qds[:, i] = spl.derivative(1)(self.time)
            self.Qdds[:, i] = spl.derivative(2)(self.time)

        self.idxColumnRotLabels = np.where(rot_mask)[0].tolist()
        self.idxColumnTrLabels = np.where(~rot_mask)[0].tolist()
        self.coordinates = self.columnLabels

        # Internos para lazy-loading
        self._stateTrajectory = None # compat

    # ===== Marcadores y rotaciones =====
    def get_marker_dict(self, session_dir, trial_name, lowpass_cutoff_frequency=-1):
        trcFilePath = os.path.join(session_dir, 'MarkerData', f'{trial_name}.trc')
        if not os.path.exists(trcFilePath):
            raise FileNotFoundError(f'No existe archivo TRC: {trcFilePath}')
        markerDict = trc_2_dict(trcFilePath)
        if lowpass_cutoff_frequency > 0:
            markerDict['markers'] = {
                name: lowPassFilter(markerDict['time'], arr, lowpass_cutoff_frequency)
                for name, arr in markerDict['markers'].items()
            }
        # Guardamos para COM aproximado
        self._last_markerDict_for_com = markerDict
        return markerDict
    
    def rotate_marker_dict(self, markerDict, euler_angles):
        rotation = Rotation.from_euler(''.join(euler_angles.keys()), list(euler_angles.values()), degrees=True)
        out = copy.deepcopy(markerDict)
        out['markers'] = {k: rotation.apply(v) for k, v in markerDict['markers'].items()}
        return out


    def rotate_com(self, comValues, euler_angles):
        rotation = Rotation.from_euler(''.join(euler_angles.keys()), list(euler_angles.values()), degrees=True)
        xyz = comValues[['x','y','z']].to_numpy()
        rot = rotation.apply(xyz)
        return pd.DataFrame({'time': comValues['time'], 'x': rot[:,0], 'y': rot[:,1], 'z': rot[:,2]})
    
    # ===== Coordenadas =====
    def get_coordinate_values(self, in_degrees=True, lowpass_cutoff_frequency=-1):
        Q = self.Qs.copy()
        if in_degrees:
            Q[:, self.idxColumnRotLabels] = np.rad2deg(Q[:, self.idxColumnRotLabels])
        if lowpass_cutoff_frequency > 0:
            Q = lowPassFilter(self.time, Q, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: estás filtrando coordenadas por segunda vez.")
        return pd.DataFrame(np.column_stack([self.time, Q]), columns=['time'] + self.columnLabels)
    
    def get_coordinate_speeds(self, in_degrees=True, lowpass_cutoff_frequency=-1):
        Qd = self.Qds.copy()
        if in_degrees:
            Qd[:, self.idxColumnRotLabels] = np.rad2deg(Qd[:, self.idxColumnRotLabels])
        if lowpass_cutoff_frequency > 0:
            Qd = lowPassFilter(self.time, Qd, lowpass_cutoff_frequency)
        return pd.DataFrame(np.column_stack([self.time, Qd]), columns=['time'] + self.columnLabels)


    def get_coordinate_accelerations(self, in_degrees=True, lowpass_cutoff_frequency=-1):
        Qdd = self.Qdds.copy()
        if in_degrees:
            Qdd[:, self.idxColumnRotLabels] = np.rad2deg(Qdd[:, self.idxColumnRotLabels])
        if lowpass_cutoff_frequency > 0:
            Qdd = lowPassFilter(self.time, Qdd, lowpass_cutoff_frequency)
        return pd.DataFrame(np.column_stack([self.time, Qdd]), columns=['time'] + self.columnLabels)


    # ===== COM aproximado sin modelo =====
    def get_center_of_mass_values(self, lowpass_cutoff_frequency=-1):
        if not hasattr(self, '_last_markerDict_for_com'):
            raise RuntimeError('Llama a get_marker_dict() antes para estimar COM sin modelo.')

        M = self._last_markerDict_for_com['markers']
        pelvis_names = ['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study']
        pelvis = np.mean(np.stack([M[n] for n in pelvis_names], axis=0), axis=0)
        com_src = pelvis.copy()

        # Time base del TRC (source)
        t_src = np.asarray(self._last_markerDict_for_com.get("time", None), dtype=float)
        if t_src is None:
            raise RuntimeError("markerDict no tiene 'time' para COM.")

        # Time base del MOT / kinematics (target)
        t_tgt = np.asarray(self.time, dtype=float)

        # 1) Interpolar com al time target si es necesario
        mismatch = (len(t_src) != len(t_tgt)) or (np.nanmax(np.abs(t_src[:min(len(t_src),len(t_tgt))] - t_tgt[:min(len(t_src),len(t_tgt))])) > 1e-9)
        if mismatch:
            com_use = np.zeros((len(t_tgt), 3), dtype=float)
            for k in range(3):
                com_use[:, k] = np.interp(t_tgt, t_src, com_src[:, k])
        else:
            com_use = com_src

        # 2) Filtrar (ya en el mismo time base)
        if lowpass_cutoff_frequency > 0:
            com_use = lowPassFilter(t_tgt, com_use, lowpass_cutoff_frequency)

        return pd.DataFrame({
            'time': t_tgt,
            'x': com_use[:, 0],
            'y': com_use[:, 1],
            'z': com_use[:, 2],
        })


    # ===== Placeholders dependientes de OpenSim =====
    def get_muscle_tendon_lengths(self, *_, **__):
        raise NotImplementedError('Sin OpenSim: longitudes músculo-tendón no disponibles.')
    def get_moment_arms(self, *_, **__):
        raise NotImplementedError('Sin OpenSim: brazos de momento no disponibles.')