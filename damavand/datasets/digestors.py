import numpy as np
import pandas as pd
import os
import scipy.io as sio
import gc
from damavand.damavand.utils import *

class KAIST:
	def __init__(self, base_directory, files, channels = list(range(4))):
		"""
		Parameters
		----------
		base_directory : str
		The home directory of the dataset.
		files : list of str
		The list of files to include during the mining process.
		channels : list of int, optional
		The list of channels to include; 0, 1, 2 and 3 correspond to x direction - housing A, y direction - housing A, x direction - housing B and y direction - housing B, respectively. Default value is [0, 1, 2, 3].
		
		Attributes
		----------
		base_dir : str
		The home directory of the dataset.
		files : list of str
		The list of files to include during the mining process.
		data : dict of list of pd.DataFrame
		Mined data is organized as a python dictonary whose keys are elements of the `channels`; corresponding values are lists of `pd.DataFrame` objects.
		"""
		self.base_dir = base_directory
		self.files = files
		self.data = {key: [] for key in channels}

	def mine(self, mining_params):
		"""
		Mine the data in the dataset based on mining parameters.

		Parameters
		----------
		mining_params : dict
		A dictionary containing the mining parameters; its keys are `win_len` and `hop_len` - the length of the window and the hop length for the window, respectively.

		Returns
		-------
		None
		"""
		for file in self.files:
			file_name_split = file.split('_')
			if len(file_name_split) > 2:
				load = list(file_name_split[0])[0]
				state = file_name_split[1]
				severity = file_name_split[2].split('.')[0]
			else:
				load = list(file_name_split[0])[0]
				state = file_name_split[1].split('.')[0]
				severity = '-'

			mat_contents = sio.loadmat(self.base_dir + file)

			for key in self.data.keys():
				temp_df = splitter(mat_contents['Signal'][0][0][1][0][0][0][:, key], mining_params['win_len'], mining_params['hop_len'])
				temp_df['load'], temp_df['state'], temp_df['severity'] = load, state, severity
				self.data[key].append(temp_df)

class MFPT:
	def __init__(self, base_directory, files):
		self.base_dir = base_directory
		self.files = files

		self.data = {
			97656: [],
			48828: [], 
		}

	def mine(self, mining_params):
		"""
		Mines the dataset by processing files in specified folders and extracts relevant data.

		Parameters
		----------
		mining_params : dict
			A dictionary containing the mining parameters with keys as sampling frequencies 
			and values as dictionaries with 'win_len' (window length) and 'hop_len' (hop length) 
			for signal processing.

		Processing Details
		------------------
		- Iterates over each folder and file in the specified base directory.
		- Loads '.mat' files and extracts data based on the file's prefix:
		- 'baseline': Normal state data is extracted.
		- 'OuterRaceFault': Data indicating outer race fault is extracted.
		- 'InnerRaceFault': Data indicating inner race fault is extracted.
		- Extracts relevant information such as sampling frequency (Fs), load, rotational speed, 
		and state from the file data.
		- Uses a splitter function to divide the data into windows based on the specified 
		window length and hop length.
		- Appends the processed data, along with its attributes, into the corresponding 
		lists in the data attribute.

		Returns
		-------
		None
		"""

		for file in self.files:
			if file.endswith('.mat'):
				file_path = os.path.join(self.base_dir, file)
				mat_data = sio.loadmat(file_path)
				bearing_data = mat_data['bearing'][0][0]

				if file.startswith('baseline'):
					Fs, load, rot_speed, signal = bearing_data[0][0][0], bearing_data[2][0][0], bearing_data[3][0][0], bearing_data[1]
					state = 'Normal'
				elif file.startswith('OuterRaceFault'):
					Fs, load, rot_speed, signal = bearing_data[3][0][0], bearing_data[1][0][0], bearing_data[0][0][0], bearing_data[2]
					state = 'OR'
				elif file.startswith('InnerRaceFault'):
					Fs, load, rot_speed, signal = bearing_data[3][0][0], bearing_data[1][0][0], bearing_data[0][0][0], bearing_data[2]
					state = 'IR'
				else:
					continue

				temp_df = splitter(signal.reshape(-1), mining_params[Fs]['win_len'], mining_params[Fs]['hop_len'])
				temp_df['Fs'], temp_df['load'], temp_df['rot_speed'], temp_df['state'] = Fs, load, rot_speed, state
				self.data[Fs].append(temp_df)



class CWRU:
	def __init__(self, base_directory, channels = ['FE', 'DE']):
		"""
		Parameters
		----------
		base_directory : str
			The directory where the dataset files are stored.
		channels : list of str, optional
			A list of channel identifiers to include. Default is ['FE', 'DE'].

		Attributes
		----------
		base_dir : str
			Stores the base directory path for the dataset files.
		channels : list of str
			Stores the list of channels to be processed.
		data : dict
			A nested dictionary where keys are channels and values are dictionaries with sampling frequencies as keys and empty lists as values, initialized for storing processed data.
		"""

		self.base_dir = base_directory
		self.channels = channels

		self.data = {channel:{Fs:[] for Fs in set([a.split('.')[0].split('_')[-1] for a in os.listdir(self.base_dir)])} for channel in self.channels}

	def mine(self, mining_params, synchronous_only=False):
		for file in os.listdir(self.base_dir):
			if file.endswith('.mat'):
				file_parts = file.split('.mat')[0].split('_')
				if len(file_parts) == 3:
					state, rot_speed, fs = file_parts
					severity = '-'
					defected_bearing = '-'
				else:
					defected_bearing, state, severity, rot_speed, fs = file_parts

				mat_data = sio.loadmat(self.base_dir + file)
				available_channels = {key.split('_')[1]: key for key in mat_data.keys() if key.split('_')[-1] == 'time'}

				if synchronous_only:
					if set(available_channels.keys()) >= set(self.channels):
						for channel in self.channels:
							if channel in available_channels.keys():
								processed_data = splitter(
									mat_data[available_channels[channel]].reshape(-1),
									mining_params[fs]['win_len'],
									mining_params[fs]['hop_len']
								)
								processed_data['state'], processed_data['defected_bearing'], processed_data['severity'], processed_data['rot_speed'], processed_data['fs'] = (
									state, defected_bearing, severity, rot_speed, fs
								)
								self.data[channel][fs].append(processed_data)
				else:
					for channel in self.channels:
						if channel in available_channels.keys():
							processed_data = splitter(
								mat_data[available_channels[channel]].reshape(-1),
								mining_params[fs]['win_len'],
								mining_params[fs]['hop_len']
							)
							processed_data['state'], processed_data['defected_bearing'], processed_data['severity'], processed_data['rot_speed'], processed_data['fs'] = (
								state, defected_bearing, severity, rot_speed, fs
							)
							self.data[channel][fs].append(processed_data)

class SEU:
	def __init__(self, base_directory, channels = list(range(8))):
		"""
		Parameters
		----------
		base_directory : str
		The home directory of the dataset.
		channels : list of int, optional
		The list of channels to include; 0, 1, 2, 3, 4, 5, 6 and 7 correspond to 8 accelerometers. Default value is [0, 1, 2, 3, 4, 5, 6, 7].

		Attributes
		----------
		base_dir : str
		The home directory of the dataset.
		channels : list of int
		The list of channels to include.
		data : dict of list of pd.DataFrame
		Mined data is organized as a python dictonary whose keys are elements of the `channels`; corresponding values are lists of `pd.DataFrame` objects.
		"""
		self.base_dir = base_directory
		self.channels = channels

		self.data = {key:[] for key in self.channels}

	def mine(self, mining_params):
		"""
		Mine the data in the dataset based on mining parameters.

		Parameters
		----------
		mining_params : dict
		A dictionary containing the mining parameters; its keys are `win_len` and `hop_len` - the length of the window and the hop length for the window, respectively.

		Returns
		-------
		None
		"""
		for sub_directory in os.listdir(self.base_dir):
			for file in os.listdir(self.base_dir + sub_directory):
				if file.endswith('.csv'):
					file_split = file.split('.csv')[0].split('_')
					test_bed = sub_directory
					state = file_split[0]
					rot_speed = file_split[1]
					with open('SEU/' + sub_directory + '/' + file, 'r', encoding='gb18030', errors='ignore') as f:
						content=f.readlines()
						if file == "ball_20_0.csv":
							arr = np.array([i.split(',')[:-1] for i in content[16:]]).astype(float)
						else:
							arr = np.array([i.split('\t')[:-1] for i in content[16:]]).astype(float)

					print('Mining: ', file)
					for key in self.data.keys():
						temp_df = splitter(arr[:, key], mining_params['win_len'], mining_params['hop_len'])
						temp_df['test_bed'] = test_bed
						temp_df['state'] = state
						temp_df['rot_speed'] = rot_speed
						self.data[key].append(temp_df)
						gc.collect()


class MaFauldDa:
	def __init__(self, base_directory, folders, channels = list(range(8))):
		"""
		Parameters
		----------
		base_directory : str
			The home directory of the dataset.
		folders : list of str
			The list of folders to include during the mining process.
		channels : list of int, optional
			The list of channels to include; valid values range from 0 to 7, corresponding to various sensors or input channels. Default is [0, 1, 2, 3, 4, 5, 6, 7].

		Attributes
		----------
		base_dir : str
			Stores the base directory path for the dataset files.
		folders : list of str
			Stores the list of folders to be processed.
		channels : list of int
			Stores the list of channels to be processed.
		data : dict of list
			A dictionary where keys are channel numbers and values are lists initialized for storing processed data.
		"""
		self.base_dir = base_directory
		self.folders = folders
		self.channels = channels

		self.data = {key: [] for key in self.channels}

	def mine(self, mining_params):
		"""
		Mines the dataset by processing files in specified folders and extracts relevant data.

		Parameters
		----------
		mining_params : dict
			A dictionary containing the mining parameters with keys 'win_len' 
			(window length) and 'hop_len' (hop length) for signal processing.

		Processing Details
		------------------
		- Iterates over each folder in the 'folders' attribute.
		- For 'normal' folder, processes files and extracts data with 'normal' state 
		and no severity.
		- For 'underhang' and 'overhang' folders, processes subfolders and files to 
		extract data, appending state and severity based on directory structure.
		- For other folders, assumes folder name as state and processes files to 
		extract data appending state and severity.
		- Uses a splitter function to divide the data into windows based on the 
		specified window length and hop length.
		- Appends the processed data, along with its attributes, into the 
		corresponding lists in the 'data' attribute.

		Returns
		-------
		None
		"""

		for folder in self.folders:
			if folder == 'normal':
				state = 'normal'
				sev = '_'
				for file in os.listdir(self.base_dir + folder):
					df = pd.read_csv(self.base_dir + folder + '/' +  file, header = None)
					for key in self.data.keys():
						temp_df = splitter(df[key].values, mining_params['win_len'], mining_params['hop_len'])
						temp_df['state'] = state
						temp_df['severity'] = sev
						self.data[key].append(temp_df)

			elif folder in ['underhang', 'overhang']:
				for subfolder in os.listdir(self.base_dir + folder + '/'):
					state = folder + '_' + subfolder
					for sev in os.listdir(self.base_dir + folder + '/' + subfolder + '/'):
						for file in os.listdir(self.base_dir + folder + '/' + subfolder + '/' + sev + '/'):
							df = pd.read_csv(self.base_dir + folder + '/' + subfolder + '/' + sev + '/' + file, header = None)
							for key in self.data.keys():
								temp_df = splitter(df[key].values, mining_params['win_len'], mining_params['hop_len'])
								temp_df['state'] = state
								temp_df['severity'] = sev
								self.data[key].append(temp_df)
				
			else:
				state = folder
				for sev in os.listdir(self.base_dir + folder + '/'):
					for file in os.listdir(self.base_dir + folder + '/' + sev):
						df = pd.read_csv(self.base_dir + folder + '/' + sev + '/' + file, header = None)
						for key in self.data.keys():
							temp_df = splitter(df[key].values, mining_params['win_len'], mining_params['hop_len'])
							temp_df['state'] = state
							temp_df['severity'] = sev
							self.data[key].append(temp_df)

class MUET():
	def __init__(self, base_directory, folders, channels = list(range(1,4))):
		"""
		Parameters
		----------
		base_directory : str
			The home directory of the dataset.
		folders : list of str
			The list of folders to include during the mining process.
		channels : list of int, optional
			The list of channels to include. Default is [1, 2, 3].

		Attributes
		----------
		base_dir : str
			The home directory of the dataset.
		channels : list of int
			The list of channels to include.
		data : dict of list of pd.DataFrame
			Mined data is organized as a python dictonary whose keys are elements of the `channels`; corresponding values are lists of `pd.DataFrame` objects.
		"""
		self.base_dir = base_directory
		self.channels = channels
		self.folders = folders

		self.data = {key: [] for key in self.channels}

	def mine(self, mining_params):
		"""
		Mine the data in the dataset based on mining parameters.

		Parameters
		----------
		mining_params : dict
			A dictionary containing the mining parameters; its keys are `win_len` and `hop_len` - the length of the window and the hop length for the window, respectively.

		Returns
		-------
		None
		"""
		for folder in self.folders:
			if folder.startswith('Healthy'):
				state = 'healthy'
				severity = '-'
				for file in os.listdir(self.base_dir + folder + '/'):
					if file.endswith('.csv'):
						load = file.split(' ')[1] + ' ' + file.split(' ')[1].split('.')[0]
						df = pd.read_csv(self.base_dir + folder + '/' + file)
						for i in self.data.keys():
							temp_df = splitter(df.iloc[:, i].to_numpy(), mining_params['win_len'], mining_params['hop_len'])
							temp_df['state'], temp_df['severity'], temp_df['load'] = state, severity, load
							self.data[i].append(temp_df)
			else:
				for file in os.listdir(self.base_dir + folder + '/'):
					if file.endswith('.csv'):
						df = pd.read_csv(self.base_dir + folder + '/' + file)
						for i in self.data.keys():
							temp_df = splitter(df.iloc[:, i].to_numpy(), mining_params['win_len'], mining_params['hop_len'])
							severity = folder.split('-')[0]
							state =''.join(list(file.split('-')[0])[3:])
							load = file.split('-')[1].split('.')[0]
							temp_df['state'], temp_df['severity'], temp_df['load'] = state, severity, load
							self.data[i].append(temp_df)

class UoO():
	def __init__(self, base_directory, channels = ['Channel_1', 'Channel_2'], reps = list(range(1,4))):
		"""
		Parameters
		----------
		base_directory : str
			The directory where the dataset files are stored.
		channels : list of str, optional
			A list of channel identifiers to include. Default is ['Channel_1', 'Channel_2'].
		reps : list of int, optional
			A list of repetition identifiers to include. Default is [1, 2, 3].

		Attributes
		----------
		base_dir : str
			Stores the base directory path for the dataset files.
		channels : list of str
			Stores the list of channels to be processed.
		reps : list of int
			Stores the list of repetition identifiers to be processed.
		data : dict
			A dictionary where keys are channel names and values are lists initialized for storing processed data.
		"""

		self.base_dir = base_directory
		self.channels = channels
		self.reps = reps

		self.data = {key: [] for key in self.channels}

	def mine(self, mining_params):
		"""
		Mine the data in the dataset based on mining parameters.

		Parameters
		----------
		mining_params : dict
			A dictionary containing the mining parameters; its keys are `win_len` and `hop_len` - the length of the window and the hop length for the window, respectively.

		Returns
		-------
		None
		"""
		
		for file in os.listdir(self.base_dir):
			if file.endswith('.mat'):
				rep = int(file.split('.')[0].split('-')[-1])
				if rep in self.reps:
					state = file.split('.')[0].split('-')[:-1][0]
					loading = file.split('.')[0].split('-')[:-1][1]
					mat_data = sio.loadmat(self.base_dir + file)
					for channel in self.data.keys():
						temp_df = splitter(mat_data[channel].reshape((-1)), mining_params['win_len'], mining_params['hop_len'])
						temp_df['state'], temp_df['loading'], temp_df['rep'] = state, loading, rep
						self.data[channel].append(temp_df)


class PU():
	def __init__(self, base_directory, folders, channels = ['CP1', 'CP2', 'Vib'], reps = list(range(1, 21))):
		"""
		Parameters
		----------
		base_directory : str
			The home directory of the dataset.
		folders : list of str
			The list of folders to include during the mining process.
		channels : list of str, optional
			A list of channel identifiers to include. Default is ['CP1', 'CP2', 'Vib'].
		reps : list of int, optional
			A list of repetition identifiers to include. Default is [1, 2, ..., 20].

		Attributes
		----------
		base_dir : str
			Stores the base directory path for the dataset files.
		folders : list of str
			Stores the list of folders to be processed.
		channels : list of str
			Stores the list of channels to be processed.
		reps : list of int
			Stores the list of repetition identifiers to be processed.
		data : dict of list
			A dictionary where keys are channel names and values are lists initialized for storing processed data.
		"""

		self.base_dir = base_directory
		self.folders = folders
		self.channels = channels
		self.reps = reps
		self.data = {key: [] for key in self.channels}

	def mine(self, mining_params):
		self.corrupted_files = {}
		for folder in self.folders:
			for file in os.listdir(self.base_dir + folder):
				if file.endswith('.mat'):
					if int(file.split('.')[0].split('_')[-1]) in self.reps:
						rot_speed, load_torque, radial_force, code, rep = file.split('.')[0].split('_')
						try:
							mat_data = sio.loadmat(self.base_dir + folder + '/' + file)

							if 'CP1' in self.channels:
								temp_df = splitter(mat_data[file.split('.')[0]]['Y'][0][0][0][1][2].reshape((-1)), mining_params['win_len'], mining_params['hop_len'])
								temp_df['rot_speed'] = rot_speed
								temp_df['load_torque'] = load_torque
								temp_df['radial_force'] = radial_force
								temp_df['code'] = code
								temp_df['rep'] = rep
								self.data['CP1'].append(temp_df)

							if 'CP2' in self.channels:
								temp_df = splitter(mat_data[file.split('.')[0]]['Y'][0][0][0][2][2].reshape((-1)), mining_params['win_len'], mining_params['hop_len'])
								temp_df['rot_speed'] = rot_speed
								temp_df['load_torque'] = load_torque
								temp_df['radial_force'] = radial_force
								temp_df['code'] = code
								temp_df['rep'] = rep
								self.data['CP2'].append(temp_df)

							if 'Vib' in self.channels:
								temp_df = splitter(mat_data[file.split('.')[0]]['Y'][0][0][0][6][2].reshape((-1)), mining_params['win_len'], mining_params['hop_len'])
								temp_df['rot_speed'] = rot_speed
								temp_df['load_torque'] = load_torque
								temp_df['radial_force'] = radial_force
								temp_df['code'] = code
								temp_df['rep'] = rep
								self.data['Vib'].append(temp_df)
						except Exception as e:
							self.corrupted_files[self.base_dir + folder + '/' + file] = e

