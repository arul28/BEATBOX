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