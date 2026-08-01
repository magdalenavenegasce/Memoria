# utils.py — versión sin dependencia de OpenSim

import os
import requests
import urllib.request
import shutil
import numpy as np
import pandas as pd
import yaml
import pickle
import glob
import zipfile
import platform

from utilsAPI import get_api_url
from utilsAuthentication import get_token
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

API_URL = get_api_url()
API_TOKEN = get_token()


# ----------------------------- Utilidades básicas I/O -----------------------------

def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def import_metadata(filePath):
    with open(filePath, "r", encoding="utf-8") as myYamlFile:
        parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    return parsedYamlFile


# ----------------------------- API (sesiones, trials) -----------------------------

def get_session_json(session_id):
    resp = requests.get(
        API_URL + f"sessions/{session_id}/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    )

    if resp.status_code == 500:
        raise Exception('No server response. Likely not a valid session id.')

    sessionJson = resp.json()
    if 'trials' not in sessionJson.keys():
        raise Exception('This session is not in your username, nor is it public. You do not have access.')

    # Ordena trials por fecha de creación.
    sessionJson['trials'].sort(key=lambda t: t['created_at'])
    return sessionJson


def get_user_sessions():
    return requests.get(
        API_URL + "sessions/valid/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    ).json()


def get_user_sessions_all(user_token=API_TOKEN):
    return requests.get(
        API_URL + "sessions/",
        headers={"Authorization": f"Token {user_token}"}
    ).json()


def get_user_subjects(user_token=API_TOKEN):
    return requests.get(
        API_URL + "subjects/",
        headers={"Authorization": f"Token {user_token}"}
    ).json()


def get_subject_sessions(subject_id, user_token=API_TOKEN):
    return requests.get(
        API_URL + f"subjects/{subject_id}/",
        headers={"Authorization": f"Token {user_token}"}
    ).json()['sessions']


def get_trial_json(trial_id):
    return requests.get(
        API_URL + f"trials/{trial_id}/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    ).json()


def get_neutral_trial_id(session_id):
    session = get_session_json(session_id)
    neutral_ids = [t['id'] for t in session['trials'] if t['name'] == 'neutral']
    if len(neutral_ids) > 0:
        neutralID = neutral_ids[-1]
    elif session['meta'].get('neutral_trial'):
        neutralID = session['meta']['neutral_trial']['id']
    else:
        raise Exception('No neutral trial in session.')
    return neutralID


def get_calibration_trial_id(session_id):
    session = get_session_json(session_id)
    calib_ids = [t['id'] for t in session['trials'] if t['name'] == 'calibration']
    if len(calib_ids) > 0:
        calibID = calib_ids[-1]
    elif session['meta'].get('sessionWithCalibration'):
        calibID = get_calibration_trial_id(session['meta']['sessionWithCalibration']['id'])
    else:
        raise Exception('No calibration trial in session.')
    return calibID


def get_trial_id(session_id, trial_name):
    session = get_session_json(session_id)
    trial_ids = [t['id'] for t in session['trials'] if t['name'] == trial_name]
    if not trial_ids:
        raise ValueError(f"Trial '{trial_name}' no encontrado en la sesión {session_id}.")
    return trial_ids[0]


# ----------------------------- Descarga de archivos de sesión/trial -----------------------------

def get_camera_mapping(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)
    trial = get_trial_json(calibration_id)
    resultTags = [res['tag'] for res in trial['results']]
    mappingPath = os.path.join(session_path, 'Videos', 'mappingCamDevice.pickle')
    os.makedirs(os.path.join(session_path, 'Videos'), exist_ok=True)
    if not os.path.exists(mappingPath):
        mappingURL = trial['results'][resultTags.index('camera_mapping')]['media']
        download_file(mappingURL, mappingPath)


def get_model_and_metadata(session_id, session_path):
    neutral_id = get_neutral_trial_id(session_id)
    trial = get_trial_json(neutral_id)
    resultTags = [res['tag'] for res in trial['results']]

    # Metadata
    metadataPath = os.path.join(session_path, 'sessionMetadata.yaml')
    if not os.path.exists(metadataPath):
        metadataURL = trial['results'][resultTags.index('session_metadata')]['media']
        os.makedirs(session_path, exist_ok=True)
        download_file(metadataURL, metadataPath)

    # Modelo (para compatibilidad con otros módulos se descarga si está disponible)
    modelURL = trial['results'][resultTags.index('opensim_model')]['media']
    modelName = modelURL[modelURL.rfind('-') + 1:modelURL.rfind('?')]
    modelFolder = os.path.join(session_path, 'OpenSimData', 'Model')
    modelPath = os.path.join(modelFolder, modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)
    return modelName


def get_model_name_from_metadata(sessionFolder, appendText='_scaled'):
    metadataPath = os.path.join(sessionFolder, 'sessionMetadata.yaml')
    if not os.path.exists(metadataPath):
        raise Exception('Session metadata not found, could not identify OpenSim model.')
    metadata = import_metadata(metadataPath)
    modelName = metadata['openSimModel'] + appendText + '.osim'
    return modelName


def get_main_settings(session_folder, trial_name):
    settings_path = os.path.join(session_folder, 'MarkerData', 'Settings', f'settings_{trial_name}.yaml')
    return import_metadata(settings_path)


def get_motion_data(trial_id, session_path):
    trial = get_trial_json(trial_id)
    trial_name = trial['name']
    resultTags = [res['tag'] for res in trial['results']]

    # Marker data (.trc)
    if 'marker_data' in resultTags:
        markerFolder = os.path.join(session_path, 'MarkerData')
        markerPath = os.path.join(markerFolder, trial_name + '.trc')
        os.makedirs(markerFolder, exist_ok=True)
        if not os.path.exists(markerPath):
            markerURL = trial['results'][resultTags.index('marker_data')]['media']
            download_file(markerURL, markerPath)

    # IK data (.mot)
    if 'ik_results' in resultTags:
        ikFolder = os.path.join(session_path, 'OpenSimData', 'Kinematics')
        ikPath = os.path.join(ikFolder, trial_name + '.mot')
        os.makedirs(ikFolder, exist_ok=True)
        if not os.path.exists(ikPath):
            ikURL = trial['results'][resultTags.index('ik_results')]['media']
            download_file(ikURL, ikPath)

    # Main settings
    if 'main_settings' in resultTags:
        settingsFolder = os.path.join(session_path, 'MarkerData', 'Settings')
        settingsPath = os.path.join(settingsFolder, 'settings_' + trial_name + '.yaml')
        os.makedirs(settingsFolder, exist_ok=True)
        if not os.path.exists(settingsPath):
            settingsURL = trial['results'][resultTags.index('main_settings')]['media']
            download_file(settingsURL, settingsPath)


def get_geometries(session_path, modelName='LaiUhlrich2022_scaled'):
    geometryFolder = os.path.join(session_path, 'OpenSimData', 'Model', 'Geometry')
    try:
        os.makedirs(geometryFolder, exist_ok=True)
        if 'Lai' in modelName:
            modelType = 'LaiArnold'
            vtpNames = [
                'capitate_lvs','capitate_rvs','hamate_lvs','hamate_rvs',
                'hat_jaw','hat_ribs_scap','hat_skull','hat_spine','humerus_lv',
                'humerus_rv','index_distal_lvs','index_distal_rvs',
                'index_medial_lvs','index_medial_rvs','index_proximal_lvs',
                'index_proximal_rvs','little_distal_lvs','little_distal_rvs',
                'little_medial_lvs','little_medial_rvs','little_proximal_lvs',
                'little_proximal_rvs','lunate_lvs','lunate_rvs','l_bofoot',
                'l_femur','l_fibula','l_foot','l_patella','l_pelvis','l_talus',
                'l_tibia','metacarpal1_lvs','metacarpal1_rvs',
                'metacarpal2_lvs','metacarpal2_rvs','metacarpal3_lvs',
                'metacarpal3_rvs','metacarpal4_lvs','metacarpal4_rvs',
                'metacarpal5_lvs','metacarpal5_rvs','middle_distal_lvs',
                'middle_distal_rvs','middle_medial_lvs','middle_medial_rvs',
                'middle_proximal_lvs','middle_proximal_rvs','pisiform_lvs',
                'pisiform_rvs','radius_lv','radius_rv','ring_distal_lvs',
                'ring_distal_rvs','ring_medial_lvs','ring_medial_rvs',
                'ring_proximal_lvs','ring_proximal_rvs','r_bofoot','r_femur',
                'r_fibula','r_foot','r_patella','r_pelvis','r_talus','r_tibia',
                'sacrum','scaphoid_lvs','scaphoid_rvs','thumb_distal_lvs',
                'thumb_distal_rvs','thumb_proximal_lvs','thumb_proximal_rvs',
                'trapezium_lvs','trapezium_rvs','trapezoid_lvs','trapezoid_rvs',
                'triquetrum_lvs','triquetrum_rvs','ulna_lv','ulna_rv'
            ]
        else:
            raise ValueError("Geometries not available for this model")
        for vtpName in vtpNames:
            url = f'https://mc-opencap-public.s3.us-west-2.amazonaws.com/geometries_vtp/{modelType}/{vtpName}.vtp'
            filename = os.path.join(geometryFolder, f'{vtpName}.vtp')
            download_file(url, filename)
    except Exception:
        pass


def download_kinematics(session_id, folder=None, trialNames=None):
    if folder is None:
        folder = os.getcwd()
    os.makedirs(folder, exist_ok=True)

    neutral_id = get_neutral_trial_id(session_id)
    get_motion_data(neutral_id, folder)

    modelName = get_model_and_metadata(session_id, folder)
    modelName = modelName.replace('.osim', '')

    sessionJson = get_session_json(session_id)
    sessionTrialNames = [t['name'] for t in sessionJson['trials']]

    if trialNames is not None:
        for t in trialNames:
            if t not in sessionTrialNames:
                print(t + ' not in session trial names.')

    loadedTrialNames = []
    for trialDict in sessionJson['trials']:
        if trialNames is not None and trialDict['name'] not in trialNames:
            continue
        trial_id = trialDict['id']
        get_motion_data(trial_id, folder)
        loadedTrialNames.append(trialDict['name'])

    loadedTrialNames = [i for i in loadedTrialNames if i not in ('neutral', 'calibration')]
    get_geometries(folder, modelName=modelName)
    return loadedTrialNames, modelName


def download_trial(trial_id, folder, session_id=None):
    trial = get_trial_json(trial_id)
    if session_id is None:
        session_id = trial['session_id']
    os.makedirs(folder, exist_ok=True)
    get_model_and_metadata(session_id, folder)
    get_motion_data(trial_id, folder)
    return trial['name']


# ----------------------------- Lectores .sto/.mot (sin OpenSim) -----------------------------

def storage_to_numpy(storage_file, excess_header_entries=0):
    """
    Lee un archivo OpenSim Storage (.sto/.mot) a structured array.
    Detecta 'endheader' y usa la línea siguiente como nombres de columnas.
    """
    with open(storage_file, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = False
        column_names = None
        line_number_of_line_containing_endheader = None
        for i, line in enumerate(f):
            if header_line:
                column_names = line.split()
                break
            if 'endheader' in line:
                line_number_of_line_containing_endheader = i + 1
                header_line = True
        if line_number_of_line_containing_endheader is None:
            # Fallback: intenta detectar línea que inicia con 'time'
            f.seek(0)
            for i, line in enumerate(f):
                if line.strip().lower().startswith('time'):
                    column_names = line.split()
                    line_number_of_line_containing_endheader = i
                    break

    if column_names is None:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        if excess_header_entries == 0:
            names = True
            skip_header = line_number_of_line_containing_endheader
        else:
            names = column_names[:-excess_header_entries]
            skip_header = line_number_of_line_containing_endheader + 1

    data = np.genfromtxt(storage_file, names=names, skip_header=skip_header, dtype=float)
    return data


def storage_to_dataframe(storage_file, headers):
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])
    return out


def load_storage(file_path, outputFormat='numpy'):
    """
    Carga .sto/.mot SIN OpenSim.
    Devuelve (data, headers) si outputFormat='numpy' o un DataFrame si 'dataframe'.
    - data: ndarray de shape (n_rows, n_cols)
    - headers: lista de nombres de columnas (incluye 'time')
    """
    data_struct = storage_to_numpy(file_path)
    headers = list(data_struct.dtype.names) if hasattr(data_struct, 'dtype') else None
    if headers is None:
        raise ValueError(f"No se pudieron determinar columnas en {file_path}")
    # Asegurar que 'time' sea la primera columna
    if 'time' in headers:
        headers = ['time'] + [h for h in headers if h != 'time']
    data_matrix = np.column_stack([data_struct[h] for h in headers])
    if outputFormat == 'numpy':
        return data_matrix, headers
    elif outputFormat == 'dataframe':
        return pd.DataFrame(data_matrix, columns=headers)
    else:
        return None


def numpy_to_storage(labels, data, storage_file, datatype=None):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    with open(storage_file, 'w', encoding='utf-8') as f:
        if datatype is None:
            f.write(f'name {storage_file}\n')
            f.write(f'datacolumns {data.shape[1]}\n')
            f.write(f'datarows {data.shape[0]}\n')
            f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
            f.write('endheader \n')
        else:
            if datatype == 'IK':
                f.write('Coordinates\n')
            elif datatype == 'ID':
                f.write('Inverse Dynamics Generalized Forces\n')
            elif datatype == 'GRF':
                f.write(f'{storage_file}\n')
            elif datatype == 'muscle_forces':
                f.write('ModelForces\n')
            f.write('version=1\n')
            f.write('nRows=%d\n' % data.shape[0])
            f.write('nColumns=%d\n' % data.shape[1])
            if datatype == 'IK':
                f.write('inDegrees=yes\n\n')
                f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
                f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
            elif datatype == 'ID':
                f.write('inDegrees=no\n')
            elif datatype == 'GRF':
                f.write('inDegrees=yes\n')
            elif datatype == 'muscle_forces':
                f.write('inDegrees=yes\n\n')
                f.write('This file contains the forces exerted on a model during a simulation.\n\n')
                f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
                f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
                f.write('Angles are in degrees.\n\n')
            f.write('endheader \n')
        for lab in labels:
            f.write(f'{lab}\t')
        f.write('\n')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f.write('%20.8f\t' % data[i, j])
            f.write('\n')


# ----------------------------- Videos y sincronización -----------------------------

def download_videos_from_server(session_id, trial_id,
                                isCalibration=False, isStaticPose=False,
                                trial_name=None, session_path=None):
    if session_path is None:
        data_dir = os.getcwd()
        session_path = os.path.join(data_dir, 'Data', session_id)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)

    resp = requests.get(f"{API_URL}trials/{trial_id}/",
                        headers={"Authorization": f"Token {API_TOKEN}"})
    trial = resp.json()
    if trial_name is None:
        trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')
    print(f"\nDownloading {trial_name}")

    videos_dir = os.path.join(session_path, "Videos")
    os.makedirs(videos_dir, exist_ok=True)

    mapping_path = os.path.join(videos_dir, 'mappingCamDevice.pickle')
    mappingCamDevice = {}
    if not os.path.exists(mapping_path):
        for k, video in enumerate(trial.get("videos", [])):
            cam_dir = os.path.join(videos_dir, f"Cam{k}", "InputMedia", trial_name)
            os.makedirs(cam_dir, exist_ok=True)
            video_path = os.path.join(cam_dir, f"{trial_name}.mov")
            download_file(video["video"], video_path)
            mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
        with open(mapping_path, 'wb') as handle:
            pickle.dump(mappingCamDevice, handle)
    else:
        with open(mapping_path, 'rb') as handle:
            mappingCamDevice = pickle.load(handle)
            # normaliza keys a upper
            for dID in list(mappingCamDevice.keys()):
                mappingCamDevice[dID.upper()] = mappingCamDevice.pop(dID)
        for video in trial.get("videos", []):
            k = mappingCamDevice[video["device_id"].replace('-', '').upper()]
            videoDir = os.path.join(videos_dir, f"Cam{k}", "InputMedia", trial_name)
            os.makedirs(videoDir, exist_ok=True)
            video_path = os.path.join(videoDir, f"{trial_name}.mov")
            if not os.path.exists(video_path) and video.get('video'):
                download_file(video["video"], video_path)

    return trial_name


def get_calibration(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)
    resp = requests.get(f"{API_URL}trials/{calibration_id}/",
                        headers={"Authorization": f"Token {API_TOKEN}"})
    trial = resp.json()
    calibResultTags = [res['tag'] for res in trial['results']]
    videoFolder = os.path.join(session_path, 'Videos')
    os.makedirs(videoFolder, exist_ok=True)
    if trial['status'] != 'done':
        return
    mapURL = trial['results'][calibResultTags.index('camera_mapping')]['media']
    mapLocalPath = os.path.join(videoFolder, 'mappingCamDevice.pickle')
    download_and_switch_calibration(session_id, session_path, calibTrialID=calibration_id)
    if len(glob.glob(mapLocalPath)) == 0:
        download_file(mapURL, mapLocalPath)


def download_and_switch_calibration(session_id, session_path, calibTrialID=None):
    if calibTrialID is None:
        calibTrialID = get_calibration_trial_id(session_id)
    resp = requests.get(f"https://api.opencap.ai/trials/{calibTrialID}/",
                        headers={"Authorization": f"Token {API_TOKEN}"})
    trial = resp.json()
    calibURLs = {t['device_id']: t['media'] for t in trial['results'] if t['tag'] == 'calibration_parameters_options'}
    calibImgURLs = {t['device_id']: t['media'] for t in trial['results'] if t['tag'] == 'calibration-img'}
    _, imgExtension = os.path.splitext(calibImgURLs[list(calibImgURLs.keys())[0]])
    lastIdx = imgExtension.find('?')
    if lastIdx > 0:
        imgExtension = imgExtension[:lastIdx]
    if 'meta' in trial.keys() and trial['meta'] is not None and 'calibration' in trial['meta'].keys():
        calibDict = trial['meta']['calibration']
        calibImgFolder = os.path.join(session_path, 'CalibrationImages')
        os.makedirs(calibImgFolder, exist_ok=True)
        for cam, calibNum in calibDict.items():
            camDir = os.path.join(session_path, 'Videos', cam)
            os.makedirs(camDir, exist_ok=True)
            file_name = os.path.join(camDir, 'cameraIntrinsicsExtrinsics.pickle')
            img_fileName = os.path.join(calibImgFolder, 'calib_img' + cam + imgExtension)
            if calibNum == 0:
                download_file(calibURLs[cam + '_soln0'], file_name)
                download_file(calibImgURLs[cam], img_fileName)
            elif calibNum == 1:
                download_file(calibURLs[cam + '_soln1'], file_name)
                download_file(calibImgURLs[cam + '_altSoln'], img_fileName)


def post_file_to_trial(filePath, trial_id, tag, device_id):
    files = {'media': open(filePath, 'rb')}
    data = {"trial": trial_id, "tag": tag, "device_id": device_id}
    requests.post(f"{API_URL}results/", files=files, data=data,
                  headers={"Authorization": f"Token {API_TOKEN}"})
    files["media"].close()


def post_video_to_trial(filePath, trial_id, device_id, parameters):
    files = {'video': open(filePath, 'rb')}
    data = {"trial": trial_id, "device_id": device_id, "parameters": parameters}
    requests.post(f"{API_URL}videos/", files=files, data=data,
                  headers={"Authorization": f"Token {API_TOKEN}"})
    files["video"].close()


def delete_video_from_trial(video_id):
    requests.delete(f"{API_URL}videos/{video_id}/",
                    headers={"Authorization": f"Token {API_TOKEN}"})


def delete_results(trial_id, tag=None, resultNum=None):
    if resultNum is not None:
        resultNums = [resultNum]
    elif tag is not None:
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results'] if r['tag'] == tag]
    else:
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results']]
    for rNum in resultNums:
        requests.delete(API_URL + f"results/{rNum}/",
                        headers={"Authorization": f"Token {API_TOKEN}"})


def set_trial_status(trial_id, status):
    if status not in ['done', 'error', 'stopped', 'reprocess']:
        raise ValueError('Invalid status. Available statuses: done, error, stopped, reprocess')
    requests.patch(API_URL + f"trials/{trial_id}/", data={'status': status},
                   headers={"Authorization": f"Token {API_TOKEN}"})


def set_session_subject(session_id, subject_id):
    requests.patch(API_URL + f"sessions/{session_id}/", data={'subject': subject_id},
                   headers={"Authorization": f"Token {API_TOKEN}"})


def get_syncd_videos(trial_id, session_path):
    trial = requests.get(f"{API_URL}trials/{trial_id}/",
                         headers={"Authorization": f"Token {API_TOKEN}"}).json()
    trial_name = trial['name']
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'video-sync':
                url = result['media']
                cam, suff = os.path.splitext(url[url.rfind('_') + 1:])
                lastIdx = suff.find('?')
                if lastIdx > 0:
                    suff = suff[:lastIdx]
                syncVideoPath = os.path.join(session_path, 'Videos', cam, 'InputMedia', trial_name,
                                             trial_name + '_sync' + suff)
                download_file(url, syncVideoPath)


def download_session(session_id, sessionBasePath=None,
                     zipFolder=False, writeToDB=False, downloadVideos=True):
    print(f'\nDownloading {session_id}')
    if sessionBasePath is None:
        sessionBasePath = os.path.join(os.getcwd(), 'Data')

    session = get_session_json(session_id)
    session_path = os.path.join(sessionBasePath, 'OpenCapData_' + session_id)

    # Archivos base
    os.makedirs(session_path, exist_ok=True)
    try:
        get_camera_mapping(session_id, session_path)
    except Exception:
        pass
    try:
        get_model_and_metadata(session_id, session_path)
    except Exception:
        pass

    # Descarga de datos y (opcional) videos
    for trial in session['trials']:
        tr_id = trial['id']
        get_motion_data(tr_id, session_path)
        if downloadVideos:
            try:
                download_videos_from_server(session_id, tr_id, session_path=session_path)
            except Exception:
                pass

    # Opcional: zip
    if zipFolder:
        zip_name = os.path.join(sessionBasePath, f'OpenCapData_{session_id}.zip')
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(session_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, sessionBasePath)
                    zf.write(abs_path, rel_path)

    return session_path