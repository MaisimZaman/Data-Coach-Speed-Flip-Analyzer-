from columns import relevant_columns

def timestamp_to_seconds(timestamp):
    """Converts a timestamp in MM:SS format to total seconds."""
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds

def build_training_dataframe(df, playerName, flip_df):
    speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] == "Speed Flip", "TimeStamp"].tolist()
    timestamps_sec = list(map(timestamp_to_seconds, speed_flip_timestamps))
    
    df_filtered = df[relevant_columns]
    
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
    
    # ✅ Keep only the first occurrence of each SecondsRemaining value
    #df_player_data = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    
    #df_filtered_postions = df_player_data.dropna(subset=['CarPositionX']).copy()
    
    df_filtered_postions = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    
    df_filtered_postions = df_player_data.dropna(subset=['CarPositionX']).copy()
    
    
    
    df_filtered_postions["SpeedFlip"] = df_filtered_postions["SecondsRemaining"].apply(lambda x: 1 if x in timestamps_sec else 0)
    
    df_filtered_postions.to_csv("training_data.csv", index=False)
