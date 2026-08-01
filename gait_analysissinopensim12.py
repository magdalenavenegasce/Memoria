import sys
sys.path.append('../')

import numpy as np
import copy
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from utilsProcessingsinopensim import lowPassFilter

from utilsKinematicssinopensim2 import kinematics


class gait_analysis(kinematics):

    def __init__(self, session_dir, trial_name, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=-1, gait_style='auto', trimming_start=0,
                 trimming_end=0, treadmill_speed_override=None):

        # Inherit init from kinematics class (offline)
        super().__init__(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values
        )

        # Trimming manual opcional
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end

        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(
            session_dir, trial_name,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values
        )

        # Coordinate values (offline: provisto por utilsKinematicssinopensim)
        self.coordinateValues = self.get_coordinate_values()

        # Trims
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start, 6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:, :]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'], 6) <=
                                         np.round(self.markerDict['time'][-1] - self.trimming_end, 6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end, :]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]

        # Rotate marker data so x is forward
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Segment gait cycles
        self.gaitEvents = self.segment_walking(n_gait_cycles=n_gait_cycles, leg=leg)
        self.nGaitCycles = np.shape(self.gaitEvents['ipsilateralIdx'])[0]

        # Treadmill speed (0 si overground)
        self.treadmillSpeed, _ = self.compute_treadmill_speed(gait_style=gait_style)

        # Sobreescribir velocidad de caminadora si se provee manualmente
        if treadmill_speed_override is not None and treadmill_speed_override > 0:
            self.treadmillSpeed = float(treadmill_speed_override)

        # Lazy vars
        self._comValues = None
        self._comValuesRotatedPerGaitCycle = None
        self._comValuesRotated = None
        self._R_world_to_gait = None
        self._leg_length = None

        # Rotaciones por ciclo
        self.markerDictRotatedPerGaitCycle = self.rotate_vector_into_gait_frame()

    # ------------------- COM -------------------
    def comValues(self, rotate=None, filt_freq=-1):
        if rotate is None:
            if self._comValues is None or filt_freq != -1:
                self._comValues = self.get_center_of_mass_values(lowpass_cutoff_frequency=filt_freq)
                if self.trimming_start > 0:
                    self._comValues = self._comValues.iloc[self.idx_trim_start:]
                if self.trimming_end > 0:
                    self._comValues = self._comValues.iloc[:self.idx_trim_end]
            return self._comValues

        if rotate == 'gaitCycle':
            if self._comValuesRotatedPerGaitCycle is None or filt_freq != -1:
                comUnrotated = self.comValues(filt_freq=filt_freq)
                comRotated = self.rotate_vector_into_gait_frame(comUnrotated[['x', 'y', 'z']].to_numpy())
                self._comValuesRotatedPerGaitCycle = pd.DataFrame(
                    data=np.concatenate(
                        (np.expand_dims(comUnrotated['time'].to_numpy(), axis=1), comRotated), axis=1
                    ),
                    columns=['time', 'x', 'y', 'z']
                )
                if self.trimming_start > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[self.idx_trim_start:]
                if self.trimming_end > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[:self.idx_trim_end]
            return self._comValuesRotatedPerGaitCycle

        if rotate == 'y':
            if self._comValuesRotated is None or filt_freq != -1:
                self._comValuesRotated = self.rotate_com(self.comValues(filt_freq=filt_freq), {'y': self.rotation_about_y})
                if self.trimming_start > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[self.idx_trim_start:]
                if self.trimming_end > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[:self.idx_trim_end]
            return self._comValuesRotated

    # ------------------- Gait frame -------------------
    def R_world_to_gait(self):
        if self._R_world_to_gait is None:
            self._R_world_to_gait = self.compute_gait_frame()
        return self._R_world_to_gait

    def get_gait_events(self):
        return self.gaitEvents

    # --------- ROTACIÓN GLOBAL ----------
    def rotate_x_forward(self):
        psis_midpoint = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2
        asis_midpoint = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2
        heading_vector = asis_midpoint - psis_midpoint
        angle = np.unwrap(np.arctan2(heading_vector[:, 2], heading_vector[:, 0]))
        n_frames = len(self.markerDict['time'])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))
        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {'y': angle})
        return angle, marker_dict_rotated

    # --------- Longitud de pierna ----------
    def leg_length(self):
        if self._leg_length is None:
            leg, contLeg = self.get_leg()
            # ipsi
            kjc = (self.markerDict['markers'][leg + '_knee_study'] +
                   self.markerDict['markers'][leg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][leg + '_ankle_study'] +
                   self.markerDict['markers'][leg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][leg.upper() + 'HJC_study']
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))
            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))
            _leg_length = {'ipsilateral': femur_length + tibia_length}
            # contra
            kjc = (self.markerDict['markers'][contLeg + '_knee_study'] +
                   self.markerDict['markers'][contLeg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][contLeg + '_ankle_study'] +
                   self.markerDict['markers'][contLeg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][contLeg.upper() + 'HJC_study']
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))
            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))
            _leg_length['contralateral'] = femur_length + tibia_length
        return _leg_length

    # --------- SCALARS ----------
    def compute_scalars(self, scalarNames, return_all=False):
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]

        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return

        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in gait_analysis class.')

        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
             scalarDict[scalarName]['units']) = thisFunction(return_all=return_all)
        return scalarDict

    def compute_stride_length(self, return_all=False):
        leg, _ = self.get_leg()
        calc_position = self.markerDictRotatedPerGaitCycle['markers'][leg + '_calc_study']

        geom = (
            - calc_position[self.gaitEvents['ipsilateralIdx'][:, :1], 0] +
            calc_position[self.gaitEvents['ipsilateralIdx'][:, 2:3], 0]
        )
        dt = np.diff(self.gaitEvents['ipsilateralTime'][:, (0, 2)])
        belt = self.treadmillSpeed * dt
        strideLengths = geom + belt

        #print("[DEBUG stride] geom=", geom.ravel(),
        #    " dt=", dt.ravel(),
        #    " treadmill=", self.treadmillSpeed,
        #    " belt=", belt.ravel(),
        #    " total=", strideLengths.ravel())

        strideLength = np.mean(strideLengths)
        units = 'm'
        if return_all:
            return strideLengths, units
        else:
            return strideLength, units

    def compute_step_length(self, return_all=False):
        leg, contLeg = self.get_leg()
        step_lengths = {}
        step_lengths[contLeg.lower()] = (
            - self.markerDictRotated['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:, :1], 0] +
            self.markerDictRotated['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:, 1:2], 0] +
            self.treadmillSpeed * (self.gaitEvents['contralateralTime'][:, 1:2] -
                                   self.gaitEvents['ipsilateralTime'][:, :1])
        )
        step_lengths[leg.lower()] = (
            self.markerDictRotated['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:, 2:], 0] -
            self.markerDictRotated['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:, 1:2], 0] +
            self.treadmillSpeed * (-self.gaitEvents['contralateralTime'][:, 1:2] +
                                   self.gaitEvents['ipsilateralTime'][:, 2:])
        )
        step_length = {key: np.mean(values) for key, values in step_lengths.items()}
        units = 'm'
        if return_all:
            return step_lengths, units
        else:
            return step_length, units

    def compute_step_length_symmetry(self, return_all=False):
        step_lengths, units = self.compute_step_length(return_all=True)
        step_length_symmetry_all = step_lengths['r'] / step_lengths['l'] * 100
        step_length_symmetry = np.mean(step_length_symmetry_all)
        units = '% (R/L)'
        if return_all:
            return step_length_symmetry_all, units
        else:
            return step_length_symmetry, units

    def compute_gait_speed(self, return_all=False):
        comValuesArray = np.vstack((self.comValues()['x'], self.comValues()['y'], self.comValues()['z'])).T
        com_component = (
            np.linalg.norm(
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:, :1]] -
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:, 2:3]], axis=2
            ) /
            np.diff(self.gaitEvents['ipsilateralTime'][:, (0, 2)])
        )
        gait_speeds = com_component + self.treadmillSpeed
        #print("[DEBUG gait_speed] com_component=", com_component.ravel(),
        #    " treadmill=", self.treadmillSpeed,
        #    " result=", gait_speeds.ravel())
        gait_speed = np.mean(gait_speeds)
        units = 'm/s'
        if return_all:
            return gait_speeds, units
        else:
            return gait_speed, units

    def compute_cadence(self, return_all=False):
        cadence_all = 60 * 2 / np.diff(self.gaitEvents['ipsilateralTime'][:, (0, 2)])
        cadence = np.mean(cadence_all)
        units = 'steps/min'
        if return_all:
            return cadence_all, units
        else:
            return cadence, units

    def compute_treadmill_speed(self, overground_speed_threshold=0.3,
                                gait_style='auto', return_all=False):
        """
        Estima la velocidad de la caminadora usando el movimiento del tobillo
        durante la fase de apoyo.
        Para sobresuelo ('overground' o 'auto' con velocidad < umbral), devuelve 0.
        """
        if gait_style in ('auto', 'treadmill'):
            leg, _ = self.get_leg()
            foot_position = self.markerDict['markers'][leg + '_ankle_study']

            # Duración de stride en frames (HS→HS) — ventana robusta para caminadora
            stance_len = (
                self.gaitEvents['ipsilateralIdx'][:, 2].astype(float) -
                self.gaitEvents['ipsilateralIdx'][:, 0].astype(float)
            )

            start_idx = np.round(
                self.gaitEvents['ipsilateralIdx'][:, 0] + 0.1 * stance_len
            ).astype(int)
            end_idx = np.round(
                self.gaitEvents['ipsilateralIdx'][:, 2] - 0.3 * stance_len
            ).astype(int)

            dt = np.nanmedian(np.diff(self.markerDict['time']))
            nframes = len(foot_position)

            treadmill_speeds = np.full(self.nGaitCycles, np.nan)
            for i in range(self.nGaitCycles):
                s, e = int(start_idx[i]), int(end_idx[i])
                if e <= s + 1 or s < 0 or e > nframes:
                    continue
                seg = foot_position[s:e, :]
                if len(seg) < 2:
                    continue
                treadmill_speeds[i] = np.linalg.norm(
                    np.mean(np.diff(seg, axis=0), axis=0)
                ) / dt

            treadmill_speed = float(np.nanmean(treadmill_speeds))

            if not np.isfinite(treadmill_speed) or treadmill_speed <= 0:
                treadmill_speed = 0.0
                treadmill_speeds = np.zeros(self.nGaitCycles)
            else:
                # Rellenar NaN individuales con la media
                treadmill_speeds = np.where(
                    np.isnan(treadmill_speeds), treadmill_speed, treadmill_speeds
                )
                # Solo anular si 'auto' confirma sobresuelo (velocidad baja)
                if gait_style == 'auto' and treadmill_speed < overground_speed_threshold:
                    treadmill_speed = 0.0
                    treadmill_speeds = np.zeros(self.nGaitCycles)

        else:  # 'overground' explícito
            treadmill_speed = 0.0
            treadmill_speeds = np.zeros(self.nGaitCycles)

        units = 'm/s'
        if return_all:
            return treadmill_speeds, units
        else:
            return treadmill_speed, units

    def compute_step_width(self, return_all=False):
        """
        Anchura de paso calculada en el gait frame (eje lateral Z).

        Markers: ankle + mankle (centro articular del tobillo) — correcto para
        ambos modos.
        La proyección al gait frame garantiza que el eje lateral sea siempre Z,
        independientemente de la dirección de marcha en el laboratorio (crítico
        para sobresuelo).
        """
        leg, contLeg = self.get_leg()

        # Centro articular del tobillo (ankle + medial ankle)
        ankle_position_ips = (
            self.markerDict['markers'][leg + '_ankle_study'] +
            self.markerDict['markers'][leg + '_mankle_study']) / 2
        ankle_position_cont = (
            self.markerDict['markers'][contLeg + '_ankle_study'] +
            self.markerDict['markers'][contLeg + '_mankle_study']) / 2

        ips_stance_length = np.diff(
            self.gaitEvents['ipsilateralIdx'][:, [0, 1]], axis=1
        ).squeeze()

        cont_stance_length = (self.gaitEvents['contralateralIdx'][:, 0] -
                              self.gaitEvents['ipsilateralIdx'][:, 0] +
                              self.gaitEvents['ipsilateralIdx'][:, 2] -
                              self.gaitEvents['contralateralIdx'][:, 1])

        n_frames = len(ankle_position_ips)
        ankle_vector = np.zeros((self.nGaitCycles, 3))

        for i in range(self.nGaitCycles):
            # --- ipsilateral midstance (40–60 % de stance) ---
            lo_ips = int(self.gaitEvents['ipsilateralIdx'][i, 0] +
                         np.round(0.4 * ips_stance_length[i]))
            hi_ips = int(self.gaitEvents['ipsilateralIdx'][i, 0] +
                         np.round(0.6 * ips_stance_length[i]))
            lo_ips = max(0, min(lo_ips, n_frames - 1))
            hi_ips = max(lo_ips + 1, min(hi_ips + 1, n_frames))

            # --- contralateral midstance ---
            base_cont = int(min(self.gaitEvents['contralateralIdx'][i, 1],
                                self.gaitEvents['ipsilateralIdx'][i, 2] - 1))
            lo_cont = base_cont + int(np.round(0.4 * cont_stance_length[i]))
            hi_cont = base_cont + int(np.round(0.6 * cont_stance_length[i]))
            lo_cont = max(0, min(lo_cont, n_frames - 1))
            hi_cont = max(lo_cont + 1, min(hi_cont + 1, n_frames))

            if lo_ips >= hi_ips or lo_cont >= hi_cont:
                ankle_vector[i, :] = np.nan
                continue

            ankle_vector[i, :] = (
                np.mean(ankle_position_cont[lo_cont:hi_cont, :], axis=0) -
                np.mean(ankle_position_ips[lo_ips:hi_ips, :], axis=0)
            )

        # Proyectar al gait frame y tomar el eje lateral (Z)
        # Nota: self.R_world_to_gait() con () — corrige bug silencioso de v6
        ankle_vector_in_gait_frame = np.array([
            np.dot(ankle_vector[i, :], self.R_world_to_gait()[i, :, :])
            for i in range(self.nGaitCycles)
        ])

        step_widths = np.abs(ankle_vector_in_gait_frame[:, 2])

        # Sustituir NaN por la media de ciclos válidos
        valid_mask = np.isfinite(step_widths)
        if np.any(valid_mask):
            step_widths[~valid_mask] = np.nanmean(step_widths[valid_mask])
        else:
            step_widths[:] = 0.0

        step_width = np.mean(step_widths)
        units = 'm'
        if return_all:
            return step_widths, units
        else:
            return step_width, units

    def compute_stance_time(self, return_all=False):
        stanceTimes = np.diff(self.gaitEvents['ipsilateralTime'][:, :2])
        stanceTime = np.mean(stanceTimes)
        units = 's'
        if return_all:
            return stanceTimes, units
        else:
            return stanceTime, units

    def compute_swing_time(self, return_all=False):
        swingTimes = np.diff(self.gaitEvents['ipsilateralTime'][:, 1:])
        swingTime = np.mean(swingTimes)
        units = 's'
        if return_all:
            return swingTimes, units
        else:
            return swingTime, units

    def compute_single_support_time(self, return_all=False):
        double_support_time, _ = self.compute_double_support_time(return_all=True)
        singleSupportTimes = 100 - double_support_time
        singleSupportTime = np.mean(singleSupportTimes)
        units = '%'
        if return_all:
            return singleSupportTimes, units
        else:
            return singleSupportTime, units

    def compute_double_support_time(self, return_all=False):
        doubleSupportTimes = (
            (np.diff(self.gaitEvents['ipsilateralTime'][:, :2]) -
             np.diff(self.gaitEvents['contralateralTime'][:, :2])) /
            np.diff(self.gaitEvents['ipsilateralTime'][:, (0, 2)])
        ) * 100
        doubleSupportTime = np.mean(doubleSupportTimes)
        units = '%'
        if return_all:
            return doubleSupportTimes, units
        else:
            return doubleSupportTime, units

    def compute_double_support_custom(self, return_all=False):
        ips = self.gaitEvents["ipsilateralTime"]
        cont = self.gaitEvents["contralateralTime"]

        stance_ips = np.diff(ips[:, :2], axis=1).squeeze()
        stance_cont = np.diff(cont[:, :2], axis=1).squeeze()
        stride_time = np.diff(ips[:, (0, 2)], axis=1).squeeze()

        double_support_times = ((stance_ips - stance_cont) / stride_time) * 100
        double_support_time = np.mean(double_support_times)
        units = "%"

        if return_all:
            return double_support_times, units
        else:
            return double_support_time, units

    def compute_midswing_dorsiflexion_angle(self, return_all=False):
        to_1_idx = self.gaitEvents['ipsilateralIdx'][:, 1]
        hs_2_idx = self.gaitEvents['ipsilateralIdx'][:, 2]
        leg, contLeg = self.get_leg()
        ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] -
                       self.markerDict['markers'][contLeg + '_ankle_study'])
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait()[i, :, :])
             for i in range(self.nGaitCycles)]
        )
        swingDfAngles = np.zeros((to_1_idx.shape))
        for i in range(self.nGaitCycles):
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[i, to_1_idx[i]:hs_2_idx[i], 0])) + to_1_idx[i]
            swingDfAngles[i] = np.mean(self.coordinateValues['ankle_angle_' +
                                     self.gaitEvents['ipsilateralLeg']].to_numpy()[idx_midSwing])
        swingDfAngle = np.mean(swingDfAngles)
        units = 'deg'
        if return_all:
            return swingDfAngles, units
        else:
            return swingDfAngle, units

    def compute_midswing_ankle_heigh_dif(self, return_all=False):
        to_1_idx = self.gaitEvents['ipsilateralIdx'][:, 1]
        hs_2_idx = self.gaitEvents['ipsilateralIdx'][:, 2]
        leg, contLeg = self.get_leg()
        ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] -
                       self.markerDict['markers'][contLeg + '_ankle_study'])
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait()[i, :, :])
             for i in range(self.nGaitCycles)]
        )
        swingAnkleHeighDiffs = np.zeros((to_1_idx.shape))
        for i in range(self.nGaitCycles):
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[i, to_1_idx[i]:hs_2_idx[i], 0])) + to_1_idx[i]
            swingAnkleHeighDiffs[i] = ankleVector_inGaitFrame[i, idx_midSwing, 1]
        swingAnkleHeighDiff = np.mean(swingAnkleHeighDiffs)
        units = 'm'
        if return_all:
            return swingAnkleHeighDiffs, units
        else:
            return swingAnkleHeighDiff, units

    def compute_peak_angle(self, dof, start_idx, end_idx, return_all=False):
        peakAngles = np.zeros((self.nGaitCycles))
        for i in range(self.nGaitCycles):
            peakAngles[i] = np.max(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        peakAngle = np.mean(peakAngles)
        units = 'deg'
        if return_all:
            return peakAngles, units
        else:
            return peakAngle, units

    def compute_rom(self, dof, start_idx, end_idx, return_all=False):
        roms = np.zeros((self.nGaitCycles))
        for i in range(self.nGaitCycles):
            roms[i] = np.ptp(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        rom = np.mean(roms)
        units = 'deg'
        if return_all:
            return roms, units
        else:
            return rom, units

    def compute_correlations(self, cols_to_compare=None, visualize=False, return_all=False):
        leg, contLeg = self.get_leg(lower=True)
        correlations_all_cycles = []
        mean_correlation_all_cycles = np.zeros((self.nGaitCycles, 1))
        for i in range(self.nGaitCycles):
            hs_ind_1 = self.gaitEvents['ipsilateralIdx'][i, 0]
            hs_ind_cont = self.gaitEvents['contralateralIdx'][i, 1]
            hs_ind_2 = self.gaitEvents['ipsilateralIdx'][i, 2]
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            if cols_to_compare is None:
                cols_to_compare = df1.columns
            for col in self.coordinateValues.columns:
                if col.endswith('_' + leg):
                    df1[col] = self.coordinateValues[col][hs_ind_1:hs_ind_2]
                elif col.endswith('_' + contLeg):
                    df2[col] = np.concatenate((self.coordinateValues[col][hs_ind_cont:hs_ind_2],
                                               self.coordinateValues[col][hs_ind_1:hs_ind_cont]))
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
            df1_interpolated = df1.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
            df2_interpolated = df2.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
            correlations = {}
            total_weighted_correlation = 0
            for col1 in df1_interpolated.columns:
                if any(col1.startswith(col_compare) for col_compare in cols_to_compare):
                    if col1.endswith('_r'):
                        corresponding_col = col1[:-2] + '_l'
                    elif col1.endswith('_l'):
                        corresponding_col = col1[:-2] + '_r'
                    else:
                        continue
                    if corresponding_col in df2_interpolated.columns:
                        signal1 = df1_interpolated[col1]
                        signal2 = df2_interpolated[corresponding_col]
                        max_range_signal1 = np.ptp(signal1)
                        max_range_signal2 = np.ptp(signal2)
                        max_range = max(max_range_signal1, max_range_signal2) if max(max_range_signal1, max_range_signal2) != 0 else 1.0
                        mean_abs_error = np.mean(np.abs(signal1 - signal2)) / max_range
                        correlation = signal1.corr(signal2)
                        weight = 1 - mean_abs_error
                        weighted_correlation = correlation * weight
                        correlations[col1] = weighted_correlation
                        total_weighted_correlation += weighted_correlation
                        if visualize:
                            plt.figure(figsize=(8, 5))
                            plt.plot(signal1, label='df1')
                            plt.plot(signal2, label='df2')
                            plt.title(f"Comparison between {col1} and {corresponding_col} with weighted correlation {weighted_correlation}")
                            plt.legend()
                            plt.show()
            mean_correlation_all_cycles[i] = total_weighted_correlation / max(1, len(correlations))
            correlations_all_cycles.append(correlations)
        if not return_all and len(correlations_all_cycles) > 0:
            mean_correlation_all_cycles = np.mean(mean_correlation_all_cycles)
            correlations_all_cycles = {key: sum(d[key] for d in correlations_all_cycles) /
                                        len(correlations_all_cycles) for key in correlations_all_cycles[0]}
        return correlations_all_cycles, mean_correlation_all_cycles

    # ------------------- Gait frame -------------------
    def compute_gait_frame(self):
        pelvisMarkerNames = ['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study']
        pelvisMarkers = [self.markerDict['markers'][mkr] for mkr in pelvisMarkerNames]
        pelvisCenter = np.mean(np.array(pelvisMarkers), axis=0)

        leg = self.gaitEvents['ipsilateralLeg']
        if leg == 'l':
            leg = 'L'
        anklePos = self.markerDict['markers'][leg + '_ankle_study']

        asisMarkerNames = ['L.ASIS_study', 'r.ASIS_study']
        asisMarkers = [self.markerDict['markers'][mkr] for mkr in asisMarkerNames]
        asisVector = np.squeeze(np.diff(np.array(asisMarkers), axis=0))

        if self.treadmillSpeed == 0:
            x = np.diff(pelvisCenter[self.gaitEvents['ipsilateralIdx'][:, (0, 2)], :], axis=1)[:, 0, :]
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        else:
            x = np.zeros((self.nGaitCycles, 3))
            for i in range(self.nGaitCycles):
                x[i, :] = (anklePos[self.gaitEvents['ipsilateralIdx'][i, 2]] -
                            anklePos[self.gaitEvents['ipsilateralIdx'][i, 1]])
            x = x / np.linalg.norm(x, axis=1, keepdims=True)

        z_temp = np.zeros((self.nGaitCycles, 3))
        for i in range(self.nGaitCycles):
            z_temp[i, :] = np.mean(asisVector[self.gaitEvents['ipsilateralIdx'][i, 0]:
                                              self.gaitEvents['ipsilateralIdx'][i, 2]], axis=0)
        z_temp = z_temp / np.linalg.norm(z_temp, axis=1, keepdims=True)

        y = np.cross(z_temp, x)
        z = np.cross(x, y)
        R_lab_to_gait = np.stack((x.T, y.T, z.T), axis=1).transpose((2, 0, 1))
        return R_lab_to_gait

    def rotate_vector_into_gait_frame(self, vectorArray=None):
        def rotate_vec(vec, R):
            return np.dot(vec, R)
        if vectorArray is None:
            markerDict_rotated_per_step = copy.deepcopy(self.markerDict)
            for marker_name, marker in markerDict_rotated_per_step['markers'].items():
                for i in range(self.nGaitCycles):
                    markerDict_rotated_per_step['markers'][marker_name][
                        self.gaitEvents['ipsilateralIdx'][i, 0]: self.gaitEvents['ipsilateralIdx'][i, 2], :] = rotate_vec(
                        marker[self.gaitEvents['ipsilateralIdx'][i, 0]: self.gaitEvents['ipsilateralIdx'][i, 2], :],
                        self.R_world_to_gait()[i, :, :])
            return markerDict_rotated_per_step
        else:
            for i in range(self.nGaitCycles):
                vectorArray[self.gaitEvents['ipsilateralIdx'][i, 0]: self.gaitEvents['ipsilateralIdx'][i, 2], :] = rotate_vec(
                    vectorArray[self.gaitEvents['ipsilateralIdx'][i, 0]: self.gaitEvents['ipsilateralIdx'][i, 2], :],
                    self.R_world_to_gait()[i, :, :])
            return vectorArray

    def get_leg(self, lower=False):
        if self.gaitEvents['ipsilateralLeg'] == 'r':
            leg = 'r'; contLeg = 'L'
        else:
            leg = 'L'; contLeg = 'r'
        if lower:
            return leg.lower(), contLeg.lower()
        else:
            return leg, contLeg

    def get_coordinates_normalized_time(self):
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)
        coordValuesNorm = []
        for i in range(self.nGaitCycles):
            coordValues = data[self.gaitEvents['ipsilateralIdx'][i, 0]: self.gaitEvents['ipsilateralIdx'][i, 2] + 1]
            coordValuesNorm.append(np.stack([np.interp(np.linspace(0, 100, 101),
                                   np.linspace(0, 100, len(coordValues)), coordValues[:, i])
                                   for i in range(coordValues.shape[1])], axis=1))
        coordinateValuesTimeNormalized = {}
        coordVals_mean = np.mean(np.array(coordValuesNorm), axis=0)
        coordinateValuesTimeNormalized['mean'] = pd.DataFrame(data=coordVals_mean, columns=colNames)
        if self.nGaitCycles > 2:
            coordVals_sd = np.std(np.array(coordValuesNorm), axis=0)
            coordinateValuesTimeNormalized['sd'] = pd.DataFrame(data=coordVals_sd, columns=colNames)
        else:
            coordinateValuesTimeNormalized['sd'] = None
        coordinateValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]
        return coordinateValuesTimeNormalized

    # ------------------- Segmentación -------------------
    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False):

        # ===== parámetros que puedes afinar =====
        EVENT_FILT_HZ = 8.0
        MIN_PEAK_DIST_S = 0.20
        PROMINENCES = [0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]
        # ========================================

        t = np.asarray(self.markerDict['time'], dtype=float)
        dt = np.nanmedian(np.diff(t))
        fs = 1.0 / dt
        min_dist = max(1, int(round(MIN_PEAK_DIST_S * fs)))

        def _lp1d(x, cutoff_hz):
            if cutoff_hz is None or cutoff_hz <= 0:
                return x
            x2 = np.asarray(x, dtype=float).reshape(-1, 1)
            x2f = lowPassFilter(t, x2, cutoff_hz)
            return x2f[:, 0]

        def detect_gait_peaks(r_calc_rel_x, l_calc_rel_x, r_toe_rel_x, l_toe_rel_x, prominence=0.3):
            r_calc_f = _lp1d(r_calc_rel_x, EVENT_FILT_HZ)
            l_calc_f = _lp1d(l_calc_rel_x, EVENT_FILT_HZ)
            r_toe_f  = _lp1d(r_toe_rel_x,  EVENT_FILT_HZ)
            l_toe_f  = _lp1d(l_toe_rel_x,  EVENT_FILT_HZ)
            rHS, _ = find_peaks(r_calc_f, prominence=prominence, distance=min_dist)
            lHS, _ = find_peaks(l_calc_f, prominence=prominence, distance=min_dist)
            rTO, _ = find_peaks(-r_toe_f, prominence=prominence, distance=min_dist)
            lTO, _ = find_peaks(-l_toe_f, prominence=prominence, distance=min_dist)
            return rHS, lHS, rTO, lTO

        def detect_correct_order(rHS, rTO, lHS, lTO, tolerance=0.15):
            expectedOrder = {'rHS': 'lTO', 'lTO': 'lHS', 'lHS': 'rTO', 'rTO': 'rHS'}
            all_events = (
                [(t, 'rHS') for t in rHS] +
                [(t, 'lTO') for t in lTO] +
                [(t, 'lHS') for t in lHS] +
                [(t, 'rTO') for t in rTO]
            )
            if len(all_events) < 4:
                return False
            all_events.sort(key=lambda x: x[0])
            sequence = [e[1] for e in all_events]
            correct = 0
            total = 0
            for j in range(len(sequence) - 1):
                cur = sequence[j]
                nxt = sequence[j + 1]
                if cur in expectedOrder:
                    total += 1
                    if expectedOrder[cur] == nxt:
                        correct += 1
            if total == 0:
                return False
            return (correct / total) >= 0.65

        # Posición relativa al sacro
        r_calc_rel = self.markerDict['markers']['r_calc_study'] - self.markerDict['markers']['r.PSIS_study']
        r_toe_rel  = self.markerDict['markers']['r_toe_study']  - self.markerDict['markers']['r.PSIS_study']
        l_calc_rel = self.markerDict['markers']['L_calc_study'] - self.markerDict['markers']['L.PSIS_study']
        l_toe_rel  = self.markerDict['markers']['L_toe_study']  - self.markerDict['markers']['L.PSIS_study']

        mid_psis = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2
        mid_asis = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2
        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:, 1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor, axis=1, keepdims=True)

        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor, r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor, l_calc_rel)
        r_toe_rel_x  = np.einsum('ij,ij->i', mid_dir_floor, r_toe_rel)
        l_toe_rel_x  = np.einsum('ij,ij->i', mid_dir_floor, l_toe_rel)

        for i, prom in enumerate(PROMINENCES):
            rHS, lHS, rTO, lTO = detect_gait_peaks(
                r_calc_rel_x=r_calc_rel_x,
                l_calc_rel_x=l_calc_rel_x,
                r_toe_rel_x=r_toe_rel_x,
                l_toe_rel_x=l_toe_rel_x,
                prominence=prom
            )
            #print(f"  prom={prom:.3f} → rHS={len(rHS)} lHS={len(lHS)} rTO={len(rTO)} lTO={len(lTO)}")

            if not detect_correct_order(rHS=rHS, rTO=rTO, lHS=lHS, lTO=lTO):
                if prom == PROMINENCES[-1]:
                    raise ValueError('The ordering of gait events is not correct. Consider trimming your trial using the trimming_start and trimming_end options.')
                else:
                    print('The gait events were not in the correct order. Trying peak detection again with prominence = ' + str(PROMINENCES[i + 1]) + '.')
            else:
                break

        if visualize:
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'], r_toe_rel_x, label='toe')
            plt.plot(self.markerDict['time'], r_calc_rel_x, label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()
            plt.figure(2)
            plt.plot(self.markerDict['time'], l_toe_rel_x, label='toe')
            plt.plot(self.markerDict['time'], l_calc_rel_x, label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        if leg == 'auto':
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'

        if leg == 'r':
            hsIps, toIps, hsCont, toCont = rHS, rTO, lHS, lTO
        else:
            hsIps, toIps, hsCont, toCont = lHS, lTO, rHS, rTO

        # Deduplicar TOs: máximo 1 por ventana inter-HS (mejora robustez en caminadora)
        def deduplicate_events(events, hs_array, keep='first'):
            if len(hs_array) < 2:
                return events
            keep_mask = np.ones(len(events), dtype=bool)
            for k in range(len(hs_array) - 1):
                w_start = hs_array[k]
                w_end   = hs_array[k + 1]
                in_win  = np.where((events > w_start) & (events < w_end))[0]
                if len(in_win) > 1:
                    discard = in_win[1:] if keep == 'first' else in_win[:-1]
                    keep_mask[discard] = False
            return events[keep_mask]

        toIps  = deduplicate_events(toIps,  hsIps, keep='first')
        toCont = deduplicate_events(toCont, hsCont if len(hsCont) > 1 else hsIps, keep='first')

        if len(hsIps) - 1 < n_gait_cycles:
            print('You requested {} gait cycles, but only {} were found. Proceeding with this number.'.format(n_gait_cycles, len(hsIps) - 1))
            n_gait_cycles = len(hsIps) - 1
        if n_gait_cycles == -1:
            n_gait_cycles = len(hsIps) - 1
        if n_gait_cycles < 1:
            raise ValueError(
                "The ordering of gait events is not correct. "
                "Consider trimming your trial using the trimming_start and trimming_end options."
            )

        gaitEvents_ips = np.zeros((n_gait_cycles, 3), dtype=int)
        gaitEvents_cont = np.zeros((n_gait_cycles, 2), dtype=int)

        for i in range(n_gait_cycles):
            gaitEvents_ips[i, 0] = hsIps[-i - 2]
            gaitEvents_ips[i, 2] = hsIps[-i - 1]

            win_start = gaitEvents_ips[i, 0]
            win_end   = gaitEvents_ips[i, 2]
            mid       = (win_start + win_end) // 2

            # TO ipsilateral: primer TO dentro de la ventana
            toIps_in_win = toIps[(toIps > win_start) & (toIps < win_end)]
            if len(toIps_in_win) > 0:
                gaitEvents_ips[i, 1] = toIps_in_win[0]

            # TO contralateral: el más cercano al punto medio
            toCont_in_win = toCont[(toCont > win_start) & (toCont < win_end)]
            if len(toCont_in_win) > 0:
                gaitEvents_cont[i, 0] = toCont_in_win[np.argmin(np.abs(toCont_in_win - mid))]
                toContFound = True
            else:
                toContFound = False

            # HS contralateral: el más cercano al punto medio
            hsCont_in_win = hsCont[(hsCont > win_start) & (hsCont < win_end)]
            if len(hsCont_in_win) > 0:
                gaitEvents_cont[i, 1] = hsCont_in_win[np.argmin(np.abs(hsCont_in_win - mid))]
                hsContFound = True
            else:
                hsContFound = False

            if not toContFound or not hsContFound:
                print('Could not find contralateral gait event within ipsilateral gait event range '
                      + str(i + 1) + ' steps until the end. Skipping this step.')
                gaitEvents_cont[i, :] = -1

        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]

        gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
        gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]

        gaitEvents = {'ipsilateralIdx': gaitEvents_ips,
                      'contralateralIdx': gaitEvents_cont,
                      'ipsilateralTime': gaitEventTimes_ips,
                      'contralateralTime': gaitEventTimes_cont,
                      'eventNamesIpsilateral': ['HS', 'TO', 'HS'],
                      'eventNamesContralateral': ['TO', 'HS'],
                      'ipsilateralLeg': leg}
        return gaitEvents
