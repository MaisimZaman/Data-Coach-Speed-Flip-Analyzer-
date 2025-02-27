from columns import relevant_columns
from columns import mean_columns
import pandas as pd
from pathlib import Path

def timestamp_to_seconds(timestamp):
    """Converts a timestamp in MM:SS format to total seconds."""
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds


def timestamp_sec_for_player(playerName, map_df):
    filtered_timestamps = map_df.loc[map_df["PLAYER"] ==  playerName, "TIMESTAMP"].tolist()
    timestamp_sec = list(map(timestamp_to_seconds, filtered_timestamps))

    return timestamp_sec




def build_training_dataframe(df, playerName, map_df, num=1, multi_player=False, no_speed_flips=False):
    if multi_player:
        timestamps_sec = timestamp_sec_for_player(playerName, map_df)
    else:
        speed_flip_timestamps = map_df.loc[map_df["Flip Type"] == "Speed Flip", "TimeStamp"].tolist()
        timestamps_sec = list(map(timestamp_to_seconds, speed_flip_timestamps))
    
    
    df_filtered = df[relevant_columns]
    
  
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
    
    
    df_filtered_postions = df_player_data.copy()
    
    
    averaged_data = df_filtered_postions.groupby('SecondsRemaining').agg(mean_columns)
    
    averaged_data['CarDodgeActive'] = df_filtered_postions.groupby('SecondsRemaining')['CarDodgeActive'].any().astype(int)
    averaged_data['CarJumpActive'] = df_filtered_postions.groupby('SecondsRemaining')['CarJumpActive'].any().astype(int)

    # Reset index to make 'SecondsRemaining' a column again
    averaged_data.reset_index(inplace=True)

    # Sort the DataFrame by 'SecondsRemaining' in descending order
    averaged_data = averaged_data.sort_values(by='SecondsRemaining', ascending=False)

    averaged_data = averaged_data.fillna(0)
    


   
    
    if no_speed_flips:
        averaged_data["SpeedFlip"] = averaged_data["SecondsRemaining"].apply(lambda x: 0)
        averaged_data.to_csv(f"Training_data/training_data_{num}.csv", index=False)
    else:
        averaged_data["SpeedFlip"] = averaged_data["SecondsRemaining"].apply(lambda x: 1 if x in timestamps_sec else 0)
        averaged_data.to_csv(f"Training_data/training_data_{num}.csv", index=False)
        
def build_multiplayer_dataframe(players, df, map_df, step=0):
    
    for p in range(len(players)):
        build_training_dataframe(df, players[p], map_df, num=step+p+1, multi_player=True)
        
        
def build_training_files(parquet_path, map_path):

    parquet_files = list(Path(parquet_path).rglob('*'))  # Assuming parquet_path is a Path object
    map_files = list(Path(map_path).rglob('*'))  # Assuming map_path is a Path object

    player_count = 0

    # Ensure both lists have the same length and corresponding files are in order
    if len(parquet_files) != len(map_files):
        print("Error: The number of parquet files and map files does not match.")
        return

    for parquet_file, map_file in zip(parquet_files, map_files):
        if parquet_file.is_file() and map_file.is_file(): 
            p_df = pd.read_parquet(parquet_file)
            map_df = pd.read_csv(map_file)

            players = p_df['PlayerName'].unique().tolist()
        
            build_multiplayer_dataframe(players, p_df, map_df, step=player_count)
            player_count += len(players)




parquet_path = 'replay_parquets'
map_path = 'Speedflip_mapping'

    
map_df = pd.read_csv("Testing_mapping/Speedflip_excel1.csv")

file_path = "Testing_parquets/game_replay.parquet"
file_path1 = "Testing_parquets/data_source.parquet"

df = pd.read_parquet(file_path)
df2 = pd.read_parquet(file_path1)

players = df['PlayerName'].unique().tolist()
players2 = df2['PlayerName'].unique().tolist()


player = players[0]
player2 = players2[0]


build_training_dataframe(df, player, map_df, num=0)

build_training_files(parquet_path, map_path)




print("Data frames built")

