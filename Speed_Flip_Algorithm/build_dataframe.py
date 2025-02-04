from columns import relevant_columns
import pandas as pd

def timestamp_to_seconds(timestamp):
    """Converts a timestamp in MM:SS format to total seconds."""
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds

def build_training_dataframe(df, playerName, flip_df, no_speed_flips=False):
    speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] == "Speed Flip", "TimeStamp"].tolist()
    timestamps_sec = list(map(timestamp_to_seconds, speed_flip_timestamps))
    
    
    df_filtered = df[relevant_columns]
    
    
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
    
    
    # ✅ Keep only the first occurrence of each SecondsRemaining value
    #df_player_data = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    
    df_filtered_postions = df_player_data.dropna(subset=['CarPositionX']).copy()
    
    
    df_filtered_postions = df_filtered_postions.dropna(subset=['CarPositionX']).copy()
    
    
    if no_speed_flips:
        df_filtered_postions["SpeedFlip"] = df_filtered_postions["SecondsRemaining"].apply(lambda x: 0)
    else:
        df_filtered_postions["SpeedFlip"] = df_filtered_postions["SecondsRemaining"].apply(lambda x: 1 if x in timestamps_sec else 0)
    
    if no_speed_flips:
        df_filtered_postions.to_csv("training_data_2.csv", index=False)
    else:
        df_filtered_postions.to_csv("training_data.csv", index=False)
    
flip_df = pd.read_csv("Speedflip_excel.csv")

file_path = "replay_parquets/game_replay.parquet"
file_path2 = "replay_parquets/data_source.parquet"

df = pd.read_parquet(file_path)
df2 = pd.read_parquet(file_path2)

players = df['PlayerName'].unique().tolist()
players2 = df2['PlayerName'].unique().tolist()

player = players[0]
player2 = players2[0]


build_training_dataframe(df, player, flip_df)
build_training_dataframe(df2, player2, flip_df, no_speed_flips=True)

