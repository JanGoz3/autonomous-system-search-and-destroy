// Definition of a chassis
// Collection of steering, motors and sensors
#include <cstdlib>
#include <cstring>
#include <Motor.hpp>
#include <RCReceiver.hpp>
#include <Steering.hpp>
#include <ServosCamera.hpp>
#include "TeensyTimerTool.h"
#include <SimpleSerialCommunicator.hpp>

#include <chrono>

// TODO: 
// 1. Read configuration from a txt file on an SD card
// 2. Test watchdog timers (communication); try out software vs hardware timers under long computations
// 

class Chassis
{
private:
    Motor motor;
    RCReceiver rcReceiver;
    Steering steering;
    ServosCamera servosCamera;
    SimpleSerialCommunicator serialComm;
    TeensyTimerTool::OneShotTimer m_CommunicationWatchdogTimer;
    // TeensyTimerTool::PeriodicTimer m_watchdogTimer;

    std::chrono::milliseconds m_maxTimeBetweenCommands{chassis_defines::WATCHDOG_MAX_TIME_BETWEEN_COMMANDS_IN_MILISECONDS};

    bool m_cameraServosDecoupledFromWatchdog = true;
    volatile bool m_watchdogEventPending = false;
    volatile bool m_programCommandExpired = false;
    bool m_lastReportedOverrideState = false;

    float m_programSpeedCommand = 0.0f;
    float m_programSteeringCommand = 0.0f;
    bool m_programMotionCommandActive = false;


public:

public:
    Chassis()
    {

        
    }

    void Initialize()
    {
        motor.Initialize(chassis_defines::MOTOR_OUT_PIN,
                        chassis_defines::MOTOR_POWER_LEVEL_DEFAULT_CONSTRAINT_FORWARD, chassis_defines::MOTOR_POWER_LEVEL_DEFAULT_CONSTRAINT_BACKWARD);
        rcReceiver.Initialize(chassis_defines::MOTOR_IN_PIN, chassis_defines::STEERING_IN_PIN);
        steering.Initialize(chassis_defines::STEERING_OUT_PIN,
                            chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_LEFT, chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_RIGHT);
        servosCamera.Initialize(chassis_defines::SERVO_PITCH_PIN, chassis_defines::SERVO_YAW_PIN, 
                                chassis_defines::SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_UP, chassis_defines::SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_DOWN, chassis_defines::SERVO_PITCH_DEFAULT_POSITION_MICROSECONDS, 
                                chassis_defines::SERVO_YAW_SWING_DEFAULT_CONSTRAINT_LEFT, chassis_defines::SERVO_YAW_SWING_DEFAULT_CONSTRAINT_RIGHT, chassis_defines::SERVO_YAW_DEFAULT_POSITION_MICROSECONDS);
        
        serialComm.Initialize(115200);
        
        InitializeWatchdogTimers();
    }

    void SetNeutral()
    {
        m_programSpeedCommand = 0.0f;
        m_programSteeringCommand = 0.0f;
        m_programMotionCommandActive = false;
        StopMotion();
        servosCamera.SetNeutralSwing();
    }

    void StopMotion()
    {
        motor.StopMotor();
        steering.SetNeutralSwing();
        ResetWatchdogTimers();
    }

    void SetSpeed(float speed)
    {
        motor.SetSpeed(speed);
        ResetWatchdogTimers();
    }

    void SetSpeed(int speed)
    {
        motor.SetSpeed(speed);
        ResetWatchdogTimers();
    }

    void SetSteering(float swing)
    {
        steering.SetSteering(swing);
        ResetWatchdogTimers();
    }

    void SetSteering(int swing)
    {
        steering.SetSteering(swing);
        ResetWatchdogTimers();
    }

    void SetCameraServos(float pitch, float yaw)
    {
        servosCamera.SetPitchYaw(pitch, yaw);

        if (m_cameraServosDecoupledFromWatchdog == false) {
            ResetWatchdogTimers();
        }
        // SUGGESTION - DO NOT reset watchdog timer for camera servos, otherwise when there are 
        // problems with sterring/motor packets the chassis can "run away" if there are still servo packets
    }


    void Update()
    {
        if (m_programCommandExpired) {
            m_programCommandExpired = false;
            m_programSpeedCommand = 0.0f;
            m_programSteeringCommand = 0.0f;
            m_programMotionCommandActive = false;
        }

        if (m_watchdogEventPending) {
            m_watchdogEventPending = false;
            serialComm.SendCommand("watchdog", "1");
        }

        rcReceiver.Update();
        PublishOverrideStateIfNeeded();

        // Process incoming serial data
        serialComm.Update();

        SimpleSerialCommunicator::Message input = {};
        while (serialComm.GetData(input)) {
            processCommand(input);
        }

        ApplySelectedControl();

    }


protected:

    static bool TryParseFloat(const char* text, float& parsedValue)
    {
        if (text == nullptr || *text == '\0') {
            return false;
        }

        char* endPointer = nullptr;
        parsedValue = strtof(text, &endPointer);
        return endPointer != text && endPointer != nullptr && *endPointer == '\0';
    }

    static bool TryParseServoPayload(const char* text, float& pitch, float& yaw)
    {
        if (text == nullptr) {
            return false;
        }

        char buffer[SimpleSerialCommunicator::MAX_DATA_LENGTH + 1] = {0};
        const size_t inputLength = strlen(text);
        if (inputLength > SimpleSerialCommunicator::MAX_DATA_LENGTH) {
            return false;
        }

        memcpy(buffer, text, inputLength + 1);

        char* comma = strchr(buffer, ',');
        if (comma == nullptr || strchr(comma + 1, ',') != nullptr) {
            return false;
        }

        *comma = '\0';
        return TryParseFloat(buffer, pitch) && TryParseFloat(comma + 1, yaw);
    }

    bool HasAnyManualOverride() const
    {
        return rcReceiver.HasAnyManualOverride();
    }

    void PublishOverrideStateIfNeeded()
    {
        const bool overrideActive = HasAnyManualOverride();
        if (overrideActive == m_lastReportedOverrideState) {
            return;
        }

        m_lastReportedOverrideState = overrideActive;
        serialComm.SendCommand("override", overrideActive ? "1" : "0");
    }

    void ApplySelectedControl()
    {
        if (rcReceiver.HasThrottleManualActive()) {
            motor.SetSpeed(rcReceiver.GetMotorCommand());
            ResetWatchdogTimers();
        } else if (m_programMotionCommandActive) {
            motor.SetSpeed(m_programSpeedCommand);
        } else {
            motor.StopMotor();
        }

        if (rcReceiver.HasSteeringAssistActive()) {
            steering.SetSteering(rcReceiver.GetSteeringCommand());
        } else if (m_programMotionCommandActive) {
            steering.SetSteering(m_programSteeringCommand);
        } else {
            steering.SetNeutralSwing();
        }
    }

    void processCommand(const SimpleSerialCommunicator::Message& input) {
        if (strcmp(input.command, "speed") == 0) {
            float speed = 0.0f;
            if (!TryParseFloat(input.data, speed)) {
                serialComm.ReportError("bad_number");
                return;
            }
            m_programSpeedCommand = speed;
            m_programMotionCommandActive = true;
            ResetWatchdogTimers();
        } else if (strcmp(input.command, "steering") == 0) {
            float steeringValue = 0.0f;
            if (!TryParseFloat(input.data, steeringValue)) {
                serialComm.ReportError("bad_number");
                return;
            }
            m_programSteeringCommand = steeringValue;
            m_programMotionCommandActive = true;
            ResetWatchdogTimers();
        } else if (strcmp(input.command, "servos") == 0) {
            float pitch = 0.0f;
            float yaw = 0.0f;
            if (!TryParseServoPayload(input.data, pitch, yaw)) {
                serialComm.ReportError("bad_value_count");
                return;
            }
            this->SetCameraServos(pitch, yaw);
        } else if (strcmp(input.command, "stop") == 0) {
            if (input.dataLength != 0) {
                serialComm.ReportError("bad_length");
                return;
            }
            m_programSpeedCommand = 0.0f;
            m_programSteeringCommand = 0.0f;
            m_programMotionCommandActive = false;
            StopMotion();
        } else if (strcmp(input.command, "set_speed") == 0) {
            serialComm.ReportError("unsupported");
        } else {
            serialComm.ReportError("unknown_cmd");
        }
}


    void InitializeWatchdogTimers()
    {   
        m_CommunicationWatchdogTimer = TeensyTimerTool::OneShotTimer(TeensyTimerTool::TCK32); // Software timer (TCK), read manual
        m_CommunicationWatchdogTimer.begin([this]{this->CommunicationTimerCallback();});

    }

    void CommunicationTimerCallback()
    {
        motor.StopMotor();
        steering.SetNeutralSwing(); 
        if (m_cameraServosDecoupledFromWatchdog == false){
            servosCamera.SetNeutralSwing();
        }
        m_programMotionCommandActive = false;
        m_programCommandExpired = true;
        m_watchdogEventPending = true;
    }


    void ResetWatchdogTimers()
    {
        // m_CommunicationWatchdogTimer.stop();
        m_CommunicationWatchdogTimer.trigger(this->m_maxTimeBetweenCommands);
    }
};