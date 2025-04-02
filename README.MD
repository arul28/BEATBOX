# Collecting and organizing data

## cells 1 - 9

Unified all the vocal percussion data into one master dataset (master_dataset.csv). Created separate files for AVP (avp_dataset.csv) and LVT (lvt_dataset.csv) data first, since they needed different handling. Main differences: LVT used different names for drums (like "Kick" instead of "kd", "Snare" instead of "sd", and "HH" instead of "hhc"), and had a different folder structure (Frase/Improviso folders instead of Personal).The final dataframe contains every vocal percussion sound from both datasets, with timing, instrument type, phonemes, and file paths all organized by participant.

Dataset Structure Overview
   - AVP Dataset: 28 participants, located in Personal directory
   - LVT Dataset: 20 participants, split between Frase and Improviso directories

Data Standardization Steps (HH in lvt was made hhc to match avp naming, Kick in lvt was made kd, etc.)

      - hhc (hi-hat closed) ← HH
      - kd (kick drum) ← Kick
      - sd (snare drum) ← Snare


DataFrame Structure

   - onset_time: When the sound occurs (seconds)
   - instrument_label: Standardized instrument name (hhc, kd, sd)
   - onset_phoneme: Starting sound in IPA notation
   - coda_phoneme: Ending sound in IPA notation
   - dataset: Source dataset ("AVP" or "LVT")
   - participant_id: Unique participant identifier
   - subset: Data subset ("personal", "Frase", or "Improviso")
   - csv_file_path: Source CSV file location
   - wav_file_path: Corresponding audio file location

Data Organization

   - Entries are grouped by participant
   - Within each participant, entries are grouped by source file
   - Within each file, entries are chronologically ordered by onset_time

# Audio Segmentation

## cells 10 - 11

Using the master dataset's onset times, cut each continuous audio recording into individual "boxemes" (isolated vocal percussion sounds). Basically took each wav file, which could have had multiple drum sounds, and extracted each indivual drum sound using the onset times. Each segment is 0.5 seconds long, following the papers research that this duration typically captures a complete amateur vocal hit without including adjacent sounds. Created a segments folder containing individual WAV files for each sound. Also generated segment_info.csv that maps each WAV segment back to its original metadata. The model we will train needs to have similar input, or else it cant learn propery, so this step makes all the input wav files the same length.

# Data Augmentation

The paper states this drastically improved results, can come back to this later since short on time

# MFCC Feature Extraction

## cells 12 - 13

The code takes the projectFiles/EDA/segment_info.csv file and looks through it and takes every single segment, and calculates the mfcc feature vector for it. The librosa library does pretty much all the work for it. Then we have X, which is a large table where each row is 13 coeffecients describing each sound, and the label y is the drum type (hhc, hho, kd, sd). Now we have features and labels. The paper did some extra stuff, adding extra coeffecients (like max derivative, temporal centroid), but I have skipped those for now, as with data augmentation above, we might come back to it if performance is mid. 





