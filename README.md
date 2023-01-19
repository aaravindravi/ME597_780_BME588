# ME 597/BME 588/ME 780 Brain Computer Interfaces Lab

This lab will cover topics in electroencephalography (EEG) based brain computer interfaces. Some of the steps in BCI pipeline such as: data handling, preprocessing, feature extraction and classification will be covered. This repository uses a publicly available offline dataset collected during a Steady-state visual evoked potentials (SSVEP) BCI experiment. This tutorial is designed as part of a lab for the Neural and Rehabilitation Engineering course at University of Waterloo (ME 597/BME 588/ME 780).

Requires Python 3.7.1. Note: When installing python on windows, add Python to environment variables. 

## Installing this repo

### Step 1: Install Git or Github Desktop
- Git download link and install: https://git-scm.com/downloads or https://gitforwindows.org/

#### Modifying PATH on Windows 10

1. In the Start Menu or taskbar search, search for "environment variable".
2. Select "Edit the system environment variables".
3. Click the "Environment Variables" button at the bottom.
4. Double-click the "Path" entry under "System variables".
5. With the "New" button in the PATH editor, add C:\Program Files\Git\bin\ and C:\Program Files\Git\cmd\ to the end of the list.
6. Close and re-open your console (Windows Powershell).

#### Clone this repository using Git

```bash
git clone https://github.com/aaravindravi/ME597_780_BME588.git
```
#### Alternate option is directly download the zip folder from the top right corner of this interface
- Download zip folder: https://github.com/aaravindravi/ME597_780_BME588/archive/refs/heads/main.zip

### Step 2: Creating a virtual environment
Run the following command in the Terminal/Powershell from inside a folder of your choice.
```bash
python -m venv bcilab
```

### Step 3: Activate the environment
Windows Power Shell
```bash
bcilab/Scripts/activate
```
macOS or Linux
```bash
source bcilab/bin/activate
```

### Step 4: Installing the requirements and dependencies
Navigate to the base folder of the project: 
```bash
   cd C:{your_working_directory}\ME597_780_BME588
```
Upgrade your pip library
```bash
python -m pip install --upgrade pip
```
Install the requirements and dependencies
```bash
pip install -r requirements.txt
```
Check the list of installed libraries
```bash
pip list
```

### Now you are ready to run the jupyter notebooks in the notebooks/ folder!
#### Navigate to your working directory and launch jupyter lab
```bash
cd C:{your_working_directory}\ME597_780_BME588
```
```bash
jupyter lab
```
This will open the jupyter lab in your browser. Navigate to the browser and 
## Helpful Download Links
- Python: https://www.python.org/downloads/
- Visual Studio Code: https://code.visualstudio.com/download
- Git download link and install: https://git-scm.com/downloads or https://gitforwindows.org/

## Note
- Windows based systems can use Windows Power Shell to run the above installation.

## Dataset Reference
12-Class publicly available SSVEP EEG Dataset
Dataset URL: https://github.com/mnakanishi/12JFPM_SSVEP/tree/master/data

## Dataset Description

Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject.
Please see the reference paper for the detail.

[Number of targets, Number of channels, Number of sampling points, Number of trials] = size(eeg)

Number of targets : 12
Number of channels : 8
Number of sampling points : 1114
Number of trials : 15
Sampling rate [Hz] : 256

The order of the stimulus frequencies in the EEG data:
[9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz
(e.g., eeg(1,:,:,:) and eeg(5,:,:,:) are the EEG data while a subject was gazing at the visual stimuli flickering at 9.25 Hz and 11.75Hz, respectively.)

The onset of visual stimulation is at 39th sample point, which means there are redundant data for 0.15 [s] before stimulus onset.