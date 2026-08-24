"""
===============================================
Compute all-to-all connectivity in sensor space
===============================================

Computes the Phase Lag Index (PLI) between all gradiometers and shows the
connectivity in 3D using the helmet geometry. The left visual stimulation data
are used which produces strong connectvitiy in the right occipital sensors.
"""

# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

# %%

import os.path as op

import mne
from mne.datasets import sample

from examples.connectivity_classes import sfreq
from mne_connectivity import spectral_connectivity_epochs

import numpy as np


print(__doc__)

# %%

# Set parameters
data_path = sample.data_path()
raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")
event_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw-eve.fif")

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# Add a bad channel
raw.info["bads"] += ["MEG 2443"]

# Pick MEG gradiometers
picks = mne.pick_types(
    raw.info, meg="grad", eeg=False, stim=False, eog=True, exclude="bads"
)

# Create epochs for the visual condition
event_id, tmin, tmax = 3, -0.2, 1.5  # want a long enough epoch for 5 cycles
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    reject=dict(grad=4000e-13, eog=150e-6),
)
epochs.load_data().pick("grad")  # just keep MEG and no EOG now

# Compute Fourier coefficients for the epochs (returns an EpochsSpectrum object)
# (storing Fourier coefficients in EpochsSpectrum objects requires MNE >= 1.8)
tmin = 0.0  # exclude the baseline period
#spectrum = epochs.compute_psd(method="multitaper", tmin=tmin, output="complex")

# Compute connectivity for the frequency band containing the evoked response
# (passing EpochsSpectrum objects as data requires MNE-Connectivity >= 0.8)
fmin, fmax = 4.0, 9.0
#con = spectral_connectivity_epochs(
#    data=spectrum, method="pli", fmin=fmin, fmax=fmax, faverage=False, n_jobs=1
#)

freqs = np.linspace(fmin, fmax, 10)
con_epochs = spectral_connectivity_epochs(
    data=epochs, method="coh", sfreq=sfreq, mode="cwt_morlet", cwt_freqs=freqs, fmin=fmin, fmax=fmax, faverage=True, n_jobs=1
)

test = con_epochs.to_networkx()
print(test)