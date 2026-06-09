#pragma once
#include <array>


namespace chassis_defines
{
    // Defines/constants for V3 chassis - based on Traxxas 4-tec 3.0 (Corvette Stingray body)

    // PWM signals configuration - DO NOT TOUCH, hardware compatibility with Traxxas ESC and Steering Servo
    const int TRAXXAX_PWM_MICROSECONDS_MIN = 1000;
    const int TRAXXAS_PWM_MICROSECONDS_MAX = 2000;
    const int TRAXXAS_PWM_MICROSECONDS_ZERO = 1500;

    // Watchdogs configuration (when serial communication stops working)
    const int WATCHDOG_MAX_TIME_BETWEEN_COMMANDS_IN_MILISECONDS = 1500;

    // RC receiver input configuration
    const int RC_INPUT_VALID_MIN_MICROSECONDS = 900;
    const int RC_INPUT_VALID_MAX_MICROSECONDS = 2100;
    const int RC_INPUT_NEUTRAL_MICROSECONDS = 1500;
    const int RC_THROTTLE_MANUAL_DEADBAND_MICROSECONDS = 60;
    const int RC_STEERING_MOVEMENT_THRESHOLD_MICROSECONDS = 10;
    const int RC_STEERING_ASSIST_HOLD_MILLISECONDS = 400;
    const int RC_INPUT_SIGNAL_TIMEOUT_MILLISECONDS = 100;
    const bool RC_INPUT_INVERT_MOTOR = false;
    const bool RC_INPUT_INVERT_STEERING = false;

    // Motors (ESC) configuration
    const int MOTOR_OUT_PIN = 9; // PIN to ESC - controls the speed
    const int MOTOR_IN_PIN = 6; // PIN from radio controller - read user input
    const float MOTOR_POWER_LEVEL_DEFAULT_CONSTRAINT_FORWARD = 1.0f; 
    const float MOTOR_POWER_LEVEL_DEFAULT_CONSTRAINT_BACKWARD = -1.0f; 


    // Steering servo configuration
    const int STEERING_OUT_PIN = 23; // PIN to servo - controls steering 
    const int STEERING_IN_PIN = 22; // PIN from radio controller - read user input
    const float STEERING_SWING_DEFAULT_CONSTRAINT_LEFT = -1.0f; // constrain left turn
    const float STEERING_SWING_DEFAULT_CONSTRAINT_RIGHT = 1.0f; // constrain right turn
    const int STEERING_SERVO_TRIM_IN_MILISECONDS = 85; // set TRIM (miliseconds) - wheels are not straight in platforms


    // Camera servos configuration
    const int SERVO_PITCH_PIN = 5; // UP - DOWN
    const int SERVO_YAW_PIN = 4; // LEFT - RIGHT
    // Set constraints for servos movement
    const float SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_UP = 1.0f; 
    const float SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_DOWN = -1.0f;
    const int SERVO_PITCH_DEFAULT_POSITION_MICROSECONDS = 1500;
    const int SERVO_YAW_DEFAULT_POSITION_MICROSECONDS = 1420;
    const float SERVO_YAW_SWING_DEFAULT_CONSTRAINT_LEFT = -1.0f;
    const float SERVO_YAW_SWING_DEFAULT_CONSTRAINT_RIGHT = 1.0f;


    // Set up ultrasonic sensors (distance sensors)
    const int ULTRASONIC_NUMBER_OF_SENSORS = 3; // Number of sensors.
    const std::array<uint8_t, ULTRASONIC_NUMBER_OF_SENSORS> ultrasonicTriggerPins = {11, 9, 24};
    const std::array<uint8_t, ULTRASONIC_NUMBER_OF_SENSORS> ultrasonicEchoPins = {10, 8, 12};

    const std::array<std::array<int, 3>, ULTRASONIC_NUMBER_OF_SENSORS> sensorOffsetsXYZFromOriginInMilimeters = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    const std::array<String, ULTRASONIC_NUMBER_OF_SENSORS> sensorDescriptions = {"Left Front", "Right Front", "Center Rear"}; // Configuration 1
    // const std::array<String, ULTRASONIC_NUMBER_OF_SENSORS> sensorDescriptions = {"Left Front", "Center Front", "Right Front"}; // Configuration 2

    const int ULTRASONNIC_MAX_DISTANCE = 200; // Maximum distance (in cm) to ping.
    const int ULTRASONIC_PING_INTERVAL = 30; // Milliseconds between sensor pings (29ms is about the min to avoid cross-sensor echo).

}

