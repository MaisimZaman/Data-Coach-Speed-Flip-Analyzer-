relevant_columns = [
    #Player Name
    "PlayerName",
    #time
    "SecondsRemaining",

    #car steer
    "CarSteer",

    #Car positions
    "CarThrottle",

    # Car Orientation (Rotation in Quaternion format)
    "CarRotationX",
    "CarRotationY",
    "CarRotationZ",
    "CarRotationW",
    
    # Car Linear Velocity (Movement Speed)
    "CarLinearVelocityX",
    "CarLinearVelocityY",
    "CarLinearVelocityZ",
    
    # Car Angular Velocity (Rotation Speed)
    "CarAngularVelocityX",
    "CarAngularVelocityY",
    "CarAngularVelocityZ",
    
    # Car Speed
    "CarSpeed",
    
    # Player Inputs
   # "CarThrottle",
    "CarBoostAmount",
    #"CarBoostActive",
    "CarDodgeActive",
    "CarJumpActive",
   # "CarDodgeImpulseX",
   # "CarDodgeImpulseY",
    #"CarDodgeImpulseZ"
]

mean_columns = {
    'CarSteer': 'mean',
    'CarThrottle': 'mean',
    'CarRotationX': 'mean',
    'CarRotationY': 'mean',
    'CarRotationZ': 'mean',
    'CarRotationW': 'mean',
    'CarLinearVelocityX': 'mean',
    'CarLinearVelocityY': 'mean',
    'CarLinearVelocityZ': 'mean',
    'CarAngularVelocityX': 'mean',
    'CarAngularVelocityY': 'mean',
    'CarAngularVelocityZ': 'mean',
    'CarSpeed': 'mean',
    }