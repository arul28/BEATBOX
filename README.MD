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

In our data augmentation process, we take each original vocal percussion sound (boxeme) and create five variations of it using two key audio transformation techniques: pitch shifting and time stretching. For pitch shifting, we randomly adjust the pitch up or down within a range of -1.5 to +1.5 semitones, which creates subtle variations in the sound's frequency while maintaining its essential character. For time stretching, we randomly modify the duration of the sound by a factor between 0.8 and 1.2 (meaning the sound can be up to 20% faster or slower) without affecting its pitch. To add more variability, we randomly alternate the order of these transformations - sometimes applying pitch shift first, other times applying time stretch first. This process is applied five times to each original segment, resulting in five unique variations per original sound. For example, if we start with a hi-hat sound, we might create one version that's slightly higher pitched and slower, another that's lower pitched and faster, and so on. The original segments are also copied to the augmented directory, bringing our total from 5,714 original segments to 34,284 segments (5,714 originals + 28,570 augmented versions). This augmentation helps simulate natural variations in how different people might perform the same drum sound, potentially improving our model's ability to recognize these sounds across different performers and styles.


# MFCC Feature Extraction

## cells 12 - 13

The code takes the projectFiles/EDA/segment_info.csv file and looks through it and takes every single segment, and calculates the mfcc feature vector for it. The librosa library does pretty much all the work for it. Then we have X, which is a large table where each row is 13 coeffecients describing each sound, and the label y is the drum type (hhc, hho, kd, sd). Now we have features and labels. The paper did some extra stuff, adding extra coeffecients (like max derivative, temporal centroid), but I have skipped those for now, as with data augmentation above, we might come back to it if performance is mid. 

# Unsupervised Learning and Visualizing Results

## cells 14 - 17

Run super simple k-means model with 4 clusters. Performance is pretty bad. Visualizing results using PCA and 2d anda 3d plots. 

need add cv, test train split, hyperparemter tuning to increase performance, and also try out more algos like dbscan and stuff

# Evaluating K-Means

## cells 18 - 19

Evaluaitng results using multiple metrics, perofrmance is pretty bad


