import pandas as pd
from columns import relevant_columns, mean_columns
import math
import joblib
from build_dataframe import build_training_dataframe



rf_model = joblib.load("ML_Models/rf_speed_flip_model.pkl")
xgb_model = joblib.load("ML_Models/xgb_speed_flip_model.pkl")

curent_model = xgb_model
flip_df = pd.read_csv("Speedflip_mapping/Speedflip_excel.csv")


speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] == "Speed Flip", "TimeStamp"].tolist()
not_speed_flip_timestamps = flip_df.loc[flip_df["Flip Type"] != "Speed Flip", "TimeStamp"].tolist()


    
print("Actual Speed Flip Timestamps:")
print(speed_flip_timestamps)
print(f"{len(speed_flip_timestamps)} total real speed flips")




def is_speed_flip(data, model):
    car_steer = data['CarSteer']
    car_rotation_x = data['CarRotationX']
    car_rotation_y = data['CarRotationY']
    car_rotation_z = data['CarRotationZ']
    car_rotation_w = data['CarRotationW']
    car_linear_velocity_x = data['CarLinearVelocityX']
    car_linear_velocity_y = data['CarLinearVelocityY']
    car_linear_velocity_z = data['CarLinearVelocityZ']
    car_angular_velocity_x = data['CarAngularVelocityX']
    car_angular_velocity_y = data['CarAngularVelocityY']
    car_angular_velocity_z = data['CarAngularVelocityZ']
    car_speed = data['CarSpeed']
    car_dodge_active = data['CarDodgeActive']
    
    input_data = {
    "CarSteer": car_steer,
    "CarRotationX": car_rotation_x,
    "CarRotationY": car_rotation_y,
    "CarRotationZ": car_rotation_z,
    "CarRotationW": car_rotation_w,
    "CarLinearVelocityX": car_linear_velocity_x,
    "CarLinearVelocityY": car_linear_velocity_y,
    "CarLinearVelocityZ": car_linear_velocity_z,
    "CarAngularVelocityX": car_angular_velocity_x,
    "CarAngularVelocityY": car_angular_velocity_y,
    "CarAngularVelocityZ": car_angular_velocity_z,
    "CarSpeed": car_speed,
    "CarDodgeActive": car_dodge_active
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        return True
    return False


        

def convert_data_to_list(data):
    cleaned_data_list = []
    for index, row in data.iterrows():
        row_dict = row.to_dict()
        cleaned_data_list.append(row_dict)
        
    return cleaned_data_list

def filter_data_for_player(df, playerName):
    df_filtered = df[relevant_columns]
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
    df_filtered_postions = df_player_data.copy()
    averaged_data = df_filtered_postions.groupby('SecondsRemaining').agg(mean_columns)
    
    averaged_data['CarDodgeActive'] = df.groupby('SecondsRemaining')['CarDodgeActive'].any().astype(int)

    # Reset index to make 'SecondsRemaining' a column again
    averaged_data.reset_index(inplace=True)

    # Sort the DataFrame by 'SecondsRemaining' in descending order
    averaged_data = averaged_data.sort_values(by='SecondsRemaining', ascending=False)
    
    return averaged_data




def get_all_speed_flips(p_data):
    speed_flips = []
    for data in convert_data_to_list(p_data):
        if is_speed_flip(data, curent_model):
            if not any(d.get('SecondsRemaining') == data['SecondsRemaining'] for d in speed_flips):
                speed_flips.append(data)
    
    return speed_flips


#returns list of formatted time for when all the speed_flips occur based on SecondsRemaining
def speed_flip_times(df, playerName):
    p_data = filter_data_for_player(df, playerName)
    speed_flips = get_all_speed_flips(p_data)
    
    formatted_speed_flip_times = map(lambda d: f"{int(divmod(d['SecondsRemaining'], 60)[0])}:{int(divmod(d['SecondsRemaining'], 60)[1]):02}" , speed_flips)
    
    return list(formatted_speed_flip_times)
    
    

file_path = "replay_parquets/game_replay.parquet"

file_path2 = "replay_parquets/data_source.parquet"

file_path3 = "replay_parquets/parquet1.parquet"

file_path3 = "replay_parquets/parquet2.parquet"


df = pd.read_parquet(file_path)

pd.set_option('display.max_columns', None)

players = df['PlayerName'].unique().tolist()
player = players[0]


print(" ")
print(f"Speed Flip timestamps predicted by ML model:")
print(speed_flip_times(df, player))
speed_flip_count = len(speed_flip_times(df, player))
print(f"{speed_flip_count} predicted speed flips")
print(player)








