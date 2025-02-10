from columns import relevant_columns
from columns import mean_columns
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
    
    
    df_filtered_postions = df_player_data.copy()
    
    
    averaged_data = df_filtered_postions.groupby('SecondsRemaining').agg(mean_columns)
    
    averaged_data['CarDodgeActive'] = df.groupby('SecondsRemaining')['CarDodgeActive'].any().astype(int)

    # Reset index to make 'SecondsRemaining' a column again
    averaged_data.reset_index(inplace=True)

    # Sort the DataFrame by 'SecondsRemaining' in descending order
    averaged_data = averaged_data.sort_values(by='SecondsRemaining', ascending=False)

    averaged_data = averaged_data.fillna(0)
    


   
    
    if no_speed_flips:
        averaged_data["SpeedFlip"] = averaged_data["SecondsRemaining"].apply(lambda x: 0)
        averaged_data.to_csv("Training_data/training_data_2.csv", index=False)
    else:
        averaged_data["SpeedFlip"] = averaged_data["SecondsRemaining"].apply(lambda x: 1 if x in timestamps_sec else 0)
        averaged_data.to_csv("Training_data/training_data.csv", index=False)
    
    
flip_df = pd.read_csv("Speedflip_mapping/Speedflip_excel.csv")

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

print("Data frames built")

