import pandas as pd
from columns import relevant_columns
import math

players = ["Rockatanskye", "blooort", "YoLookItsOprah", "NottheCaveman", "gingerflipz", "JackoaTrades01"]


def is_speed_flip(data):
    # Extract data
    player_name = data['PlayerName']
    car_position_x = data['CarPositionX']
    car_position_y = data['CarPositionY']
    car_position_z = data['CarPositionZ']
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

    # Thresholds and ranges
    SPEED_THRESHOLD = 40.0                      # Speed above 40 units per second
    HEIGHT_THRESHOLD = 30.0                     # Minimum height indicating a jump
    VERTICAL_VELOCITY_THRESHOLD = 180.0         # Slightly increased vertical velocity threshold
    ANGULAR_VELOCITY_LIMIT = 2000.0             # Reduced limit for angular velocity
    FLEXIBLE_ANGLE_RANGE = 120.0                # Tightened range for roll/pitch angles (120 to 180 degrees)

    # Map boundary limits
    MAP_X_LIMIT = 4000.0                        # Approximate X map boundary
    MAP_Y_LIMIT = 5000.0                        # Approximate Y map boundary
    EDGE_MARGIN = 200.0                         # Margin for proximity to edges

    # Convert quaternion to roll (rotation around X-axis) and pitch (rotation around Y-axis)
    roll = math.atan2(2 * (car_rotation_w * car_rotation_x + car_rotation_y * car_rotation_z),
                      1 - 2 * (car_rotation_x**2 + car_rotation_y**2)) * (180 / math.pi)

    pitch = math.asin(2 * (car_rotation_w * car_rotation_y - car_rotation_z * car_rotation_x)) * (180 / math.pi)

    # Check if the car is close to upside down or tilted diagonally
    is_close_to_upside_down = (
        abs(roll) >= FLEXIBLE_ANGLE_RANGE or
        abs(pitch) >= FLEXIBLE_ANGLE_RANGE
    )

    # Speed condition
    is_speed_high = car_speed > SPEED_THRESHOLD

    # Height condition
    is_height_increasing = car_position_z > HEIGHT_THRESHOLD

    # Vertical velocity condition
    is_vertical_velocity_high = car_linear_velocity_z > VERTICAL_VELOCITY_THRESHOLD

    # Angular velocity condition: allow rapid rotations but with a reduced limit
    is_angular_velocity_high = (
        abs(car_angular_velocity_x) > 400.0 or
        abs(car_angular_velocity_y) > 400.0 or
        abs(car_angular_velocity_z) > 400.0
    )

    # Edge detection
    is_near_edge = (
        abs(car_position_x) > MAP_X_LIMIT - EDGE_MARGIN or
        abs(car_position_y) > MAP_Y_LIMIT - EDGE_MARGIN
    )

    # Final decision: flexible conditions for speed flips
    return (
        is_speed_high and
        is_height_increasing and
        is_vertical_velocity_high and
        (is_angular_velocity_high or is_close_to_upside_down) and
        not is_near_edge
    )
    
    

def convert_data_to_list(data):
    cleaned_data_list = []
    for index, row in data.iterrows():
        row_dict = row.to_dict()
        cleaned_data_list.append(row_dict)
        
    return cleaned_data_list

def filter_data_for_player(df, playerName):
    df_filtered = df[relevant_columns]
    df_player_data = df_filtered[df_filtered['PlayerName'] == playerName]
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
    p_data = filter_data_for_player(df, playerName)
    speed_flips = get_all_speed_flips(p_data)
    
    formatted_speed_flip_times = map(lambda d: f"{int(divmod(d['SecondsRemaining'], 60)[0])}:{int(divmod(d['SecondsRemaining'], 60)[1]):02}" , speed_flips)
    
    return list(formatted_speed_flip_times)
    
    

file_path = "data_source.parquet"


df = pd.read_parquet(file_path)

pd.set_option('display.max_columns', None)


#To find when speed flips occur pass in the dataframe and the playername to find when they do speed flips in Seconds Remaining
print(speed_flip_times(df, players[0]))







