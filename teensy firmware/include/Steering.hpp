#pragma once
#include <Servo.h>
#include <Arduino.h>
#include <chassis_defines.h>
// #include <math.h>

// TODO: 
// 1. Test steering offset (trim)
// 1. Test all conversion functions
// 2. Test all Set functions
// 

class Steering
{
private:

    // Traxxas servo - 1ms (min) to 2ms (max), where 1,5ms is zero
    int m_minimumPWMSignalLengthInMicroseconds = chassis_defines::TRAXXAX_PWM_MICROSECONDS_MIN;
    int m_maximumPWMSignalLengtoInMicroseconds = chassis_defines::TRAXXAS_PWM_MICROSECONDS_MAX;
    int m_zeroPWMSignalInMicroseconds = min(max(chassis_defines::TRAXXAS_PWM_MICROSECONDS_ZERO + chassis_defines::STEERING_SERVO_TRIM_IN_MILISECONDS, chassis_defines::TRAXXAX_PWM_MICROSECONDS_MIN), chassis_defines::TRAXXAS_PWM_MICROSECONDS_MAX);
    
    uint8_t m_PWMControlPin = -1;
    Servo m_steeringPWM;
    float m_SwingConstraintLeft = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_LEFT;
    float m_SwingConstraintRight = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_RIGHT;
    bool m_isInitialized = false;


    float m_currentSetSwing = 0.0f;

    void attachServo()
    {
        m_steeringPWM.attach(m_PWMControlPin, m_minimumPWMSignalLengthInMicroseconds, m_maximumPWMSignalLengtoInMicroseconds);
        m_steeringPWM.writeMicroseconds(m_zeroPWMSignalInMicroseconds);
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

    Steering()
    {

    }

    Steering(uint8_t pin, 
            float steeringLimitLeft = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_LEFT, 
            int steeringLimitRight = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_RIGHT)
    {
        Initialize(pin, steeringLimitLeft, steeringLimitRight);
    }

    void Initialize(uint8_t pin, 
                    float steeringLimitLeft = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_LEFT,  
                    float steeringLimitRight = chassis_defines::STEERING_SWING_DEFAULT_CONSTRAINT_RIGHT)
    {
        m_PWMControlPin = pin;
        m_SwingConstraintLeft = steeringLimitLeft;
        m_SwingConstraintRight  = steeringLimitRight;
        attachServo();
        m_isInitialized = true;

    }

    void SetSteering(float swing)
    {
        if (m_isInitialized)
        {
            // float swingConstrained = (swing < m_SwingConstraintLeft) ? m_SwingConstraintLeft : ((swing > m_SwingConstraintRight) ? m_SwingConstraintRight : swing);
            float swingConstrained = constrain(swing, m_SwingConstraintLeft, m_SwingConstraintRight );

            m_steeringPWM.write(convertSwingToDegrees(swingConstrained));
            m_currentSetSwing = swingConstrained;
        }

    }

    void SetSteering(int swing)
    {
        if (m_isInitialized)
        {
            int swingConstrained = constrain(swing, -100, 100);
            float swingConverted = ((float)swingConstrained - (-100.0f)) * (1.0f - (-1.0f)) / (100.0f - (-100.0f))  + (-1.0f);
            SetSteering(swingConverted);
            
        }

    }

    // Send the "zero" speed signal to serwo.
    void SetNeutralSwing()
    {
        if (m_isInitialized)
        {
            m_steeringPWM.writeMicroseconds(m_zeroPWMSignalInMicroseconds);
            m_currentSetSwing = 0.0f;
        }
    }


    float GetCurrentSetSwing()
    {
        return m_currentSetSwing;
    }


};