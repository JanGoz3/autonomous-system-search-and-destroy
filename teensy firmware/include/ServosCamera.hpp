#pragma once
#include <Servo.h>
#include <Arduino.h>
#include <chassis_defines.h>
// #include <math.h>



class ServosCamera
{
private:
    uint8_t m_PitchControlPin = -1;
    Servo m_pitchServo;
    uint8_t m_YawControlPin = -1;
    Servo m_yawServo;
    bool m_isInitialized = false;
    float m_currentSetSwingPitch = 0.0f;
    float m_currentSetSwingYaw = 0.0f;


    float m_SwingConstraintPitchUp = chassis_defines::SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_UP;
    float m_SwingConstraintPitchDown = chassis_defines::SERVO_PITCH_SWING_DEFAULT_CONSTRAINT_DOWN;
    float m_SwingPitchNeutral = chassis_defines::SERVO_PITCH_DEFAULT_POSITION_MICROSECONDS;

    float m_SwingConstraintYawLeft = chassis_defines::SERVO_YAW_SWING_DEFAULT_CONSTRAINT_LEFT;
    float m_SwingConstraintYawRight = chassis_defines::SERVO_YAW_SWING_DEFAULT_CONSTRAINT_RIGHT;
    float m_SwingYawNeutral = chassis_defines::SERVO_YAW_DEFAULT_POSITION_MICROSECONDS;

    void attachServos()
    {
        m_pitchServo.attach(m_PitchControlPin);
        m_yawServo.attach(m_YawControlPin);
        
        m_pitchServo.writeMicroseconds(m_SwingPitchNeutral);
        m_yawServo.writeMicroseconds(m_SwingYawNeutral);

    }


    // converts from -1 to 1 range into 0-180 range
    int convertSwingToDegrees(float swing)
    {
        // OldRange = (OldMax - OldMin)  
        // NewRange = (NewMax - NewMin)  
        // NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        int deg = int(((swing - (-1.0f)) * (180 - 0)) / (1.0f - (-1.0f)) );
        return deg;
    }



public:


    ServosCamera()
    {
        
    }

    ServosCamera(uint8_t pitchPin, uint8_t yawPin,
                float pitchLimitUp = 1.0f, float pitchLimitDown = -1.0f, int pitchDefaultPosition = 1500,
                float yawLimitLeft = -1.0f, float yawLimitRight = 1.0f, int yawDefaultPosition = 1500)
    {
        Initialize(pitchPin, yawPin, pitchLimitUp, pitchLimitDown, pitchDefaultPosition, yawLimitLeft, yawLimitRight, yawDefaultPosition );
    }

    void Initialize(uint8_t pitchPin, uint8_t yawPin,
                float pitchLimitUp = 1.0f, float pitchLimitDown = -1.0f, int pitchDefaultPosition = 1500,
                float yawLimitLeft = -1.0f, float yawLimitRight = 1.0f, int yawDefaultPosition = 1500)
    {
        m_PitchControlPin = pitchPin;
        m_YawControlPin = yawPin;

        m_SwingConstraintPitchUp = pitchLimitUp;
        m_SwingConstraintPitchDown = pitchLimitDown;
        m_SwingPitchNeutral = pitchDefaultPosition;
        
        m_SwingConstraintYawLeft = yawLimitLeft;
        m_SwingConstraintYawRight = yawLimitRight;
        m_SwingYawNeutral = yawDefaultPosition;

        attachServos();

        m_isInitialized = true;
        

    }

    void SetPitchYaw(float swingPitch, float swingYaw)
    {
        SetPitch(swingPitch);
        SetYaw(swingYaw);

    }

    void SetPitch(float swingPitch)
    {
        if (m_isInitialized)
        {
            float swingConstrainedPitch = constrain(-1*swingPitch, m_SwingConstraintPitchDown, m_SwingConstraintPitchUp );
            m_pitchServo.write(convertSwingToDegrees(swingConstrainedPitch));
        }

    }

    void SetYaw(float swingYaw)
    {
        if (m_isInitialized)
        {
            float swingConstrainedYaw = constrain(-1*swingYaw, m_SwingConstraintYawLeft, m_SwingConstraintYawRight);
            m_yawServo.write(convertSwingToDegrees(swingConstrainedYaw));
        }

    }

    
    // Send the "zero" speed signal to servo.
    void SetNeutralSwing()
    {
        if (m_isInitialized)
        {
            m_pitchServo.writeMicroseconds(m_SwingPitchNeutral);
            m_yawServo.writeMicroseconds(m_SwingYawNeutral);
            m_currentSetSwingPitch = 0.0f;
            m_currentSetSwingYaw = 0.0f;
        }
    }

    std::pair<float, float> GetCurrentPitchYaw()
    {
        return std::pair{m_currentSetSwingPitch, m_currentSetSwingYaw};
    }
};