import pandas as pd
import os

def modify_segment_info():
    """Modify segment_info.csv to use base names for LVT participants and save as new file"""
    print("\n=== Starting Segment Info Modification ===")
    
    # Read the original segment info file
    input_path = 'projectFiles/segment_info/segment_info.csv'
    output_path = 'projectFiles/segment_info/segment_info_base_names.csv'
    
    print(f"Reading from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Get original unique participants
    original_participants = sorted(df['participant_id'].unique())
    print(f"\nOriginal unique participants ({len(original_participants)}):")
    for p in original_participants:
        print(f"  {p}")
    
    # Create a mapping for LVT participants to their base names
    # For each LVT participant (not starting with 'P'), remove the last character (I or P)
    lvt_mask = ~df['participant_id'].str.startswith('P')
    lvt_participants = df.loc[lvt_mask, 'participant_id'].unique()
    
    # Special case mapping for JSoI and JoSP
    special_cases = {
        'JSoI': 'JoS',
        'JoSP': 'JoS'
    }
    
    print("\nLVT participant name changes:")
    for p in sorted(lvt_participants):
        if p in special_cases:
            base_name = special_cases[p]
            print(f"  {p} -> {base_name} (special case)")
            df.loc[df['participant_id'] == p, 'participant_id'] = base_name
        else:
            base_name = p[:-1]  # Remove last character (I or P)
            print(f"  {p} -> {base_name}")
            df.loc[df['participant_id'] == p, 'participant_id'] = base_name
    
    # Save the modified file
    print(f"\nSaving modified file to: {output_path}")
    df.to_csv(output_path, index=False)
    
    # Get final unique participants
    final_participants = sorted(df['participant_id'].unique())
    print(f"\nFinal unique participants ({len(final_participants)}):")
    for p in final_participants:
        print(f"  {p}")
    
    # Count AVP and LVT participants
    avp_count = sum(1 for p in final_participants if p.startswith('P'))
    lvt_count = sum(1 for p in final_participants if not p.startswith('P'))
    
    print(f"\nFinal counts:")
    print(f"  AVP participants: {avp_count}")
    print(f"  LVT participants: {lvt_count}")
    print(f"  Total participants: {len(final_participants)}")
    
    if len(final_participants) != 48:
        print(f"\nWARNING: Expected 48 participants, but found {len(final_participants)}")
    else:
        print("\n✓ Successfully consolidated to 48 participants")

if __name__ == "__main__":
    modify_segment_info() 