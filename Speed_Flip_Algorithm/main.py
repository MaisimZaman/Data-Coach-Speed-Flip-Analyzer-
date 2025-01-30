import pandas as pd
from columns import relevant_columns
import math




flip_df = pd.read_csv("Speedflip_excel.csv")

speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] == "Speed Flip", "TimeStamp"].tolist()
not_speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] != "Speed Flip", "TimeStamp"].tolist()
    
def timestamp_to_seconds(timestamp):
    """Converts a timestamp in MM:SS format to total seconds."""
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds

timestamps_sec = list(map(timestamp_to_seconds, speed_flip_timestamps))
not_timestamps_sec = list(map(timestamp_to_seconds, not_speed_flip_timestamps))

print(timestamps_sec)
print(not_timestamps_sec)
def is_speed_flip(data):
    pass 

def build_training_dataframe(df, playerName, timestamps_sec):
    df_filtered = df[relevant_columns]
    
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
    filtered_seconds = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    df_filtered_postions = df_player_data.dropna(subset=['CarPositionX'])
    
    df_filtered_postions["SpeedFlip"] = df_filtered_postions["SecondsRemaining"].apply(lambda x: 1 if x in timestamps_sec else 0)
    
    df_filtered_postions.to_csv("training_data.csv", index=False)
    
    
    
    
    


    
    

def convert_data_to_list(data):
    cleaned_data_list = []
    for index, row in data.iterrows():
        row_dict = row.to_dict()
        cleaned_data_list.append(row_dict)
        
    return cleaned_data_list

def filter_data_for_speed_flip(df, playerName):
    df_filtered = df[relevant_columns]
    speed_flip_timestamps = df_filtered[df_filtered["SecondsRemaining"].isin(timestamps_sec)]
    df_player_data = speed_flip_timestamps[speed_flip_timestamps['PlayerName'] == playerName]
    filtered_seconds = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    df_filtered_postions = df_player_data.dropna(subset=['CarPositionX'])
    
    return df_filtered_postions

def filter_data_for_not_speed_flip(df, playerName):
    df_filtered = df[relevant_columns]
    not_speed_flip_timestamps = df_filtered[df_filtered["SecondsRemaining"].isin(not_timestamps_sec)]
    df_player_data = not_speed_flip_timestamps[not_speed_flip_timestamps['PlayerName'] == playerName]
    filtered_seconds = df_player_data.drop_duplicates(subset='SecondsRemaining', keep='first')
    df_filtered_postions = df_player_data.dropna(subset=['CarPositionX'])
    
    return df_filtered_postions


def get_all_speed_flips(p_data):
    speed_flips = []
    for data in convert_data_to_list(p_data):
        if is_speed_flip(data):
            if not any(d.get('SecondsRemaining') == data['SecondsRemaining'] for d in speed_flips):
                speed_flips.append(data)
    
    return speed_flips


#returns list of formatted time for when all the speed_flips occur based on SecondsRemaining
def speed_flip_times(df, playerName):
    p_data = filter_data_for_speed_flip(df, playerName)
    speed_flips = get_all_speed_flips(p_data)
    
    formatted_speed_flip_times = map(lambda d: f"{int(divmod(d['SecondsRemaining'], 60)[0])}:{int(divmod(d['SecondsRemaining'], 60)[1]):02}" , speed_flips)
    
    return list(formatted_speed_flip_times)
    
    

file_path = "game_replay.parquet"


df = pd.read_parquet(file_path)

pd.set_option('display.max_columns', None)

players = df['PlayerName'].unique().tolist()
player = players[0]


speed_flip_df = filter_data_for_speed_flip(df, player)
not_speed_flip_df = filter_data_for_not_speed_flip(df, player)
#speed_flip_df.to_csv("speed_flip_stats.csv", index=False) 
#not_speed_flip_df.to_csv("not_speed_flip_stats.csv")
build_training_dataframe(df, player, timestamps_sec)

#To find when speed flips occur pass in the dataframe and the playername to find when they do speed flips in Seconds Remaining
#print(f"Player: {player}")
#print(speed_flip_times(df, player))
#print(len(speed_flip_times(df, player)))







