import pandas as pd
import math

file_path = "model_result.parquet"

def seconds_to_minute_second(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}:{int(remaining_seconds):02}"


def filter_speed_flips(player_name):
    filtered_player = df[df['PlayerName'] == player_name]
    filtered_speed_flip = filtered_player[filtered_player['PrecedingMechanicsEvents'] == "Speed flip"]
    time_remaining_list = filtered_speed_flip['TimeRemainingInSeconds']
    
    formated_split_flips = list(map(lambda x: seconds_to_minute_second(x), time_remaining_list.tolist()))
    
    return formated_split_flips
    

df = pd.read_parquet(file_path)


players = df['PlayerName'].unique().tolist()
player = players[2]
print(player)
print(filter_speed_flips(player))
print(len(filter_speed_flips(player)))