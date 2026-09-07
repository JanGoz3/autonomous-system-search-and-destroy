#pragma once
#include <Servo.h>
#include <Arduino.h>
#include <chassis_defines.h>

// TODO: 
// 1. Test braking (how to brake without going into reverse)
// 2. Test all conversion functions
// 3. Test all Set functions

class Motor
{
private:
    // Traxxas XL5 ESC requires signals to be in 1ms (min) to 2ms (max), where 1,5ms is zero
    int m_minimumPWMSignalLengthInMicroseconds = chassis_defines::TRAXXAX_PWM_MICROSECONDS_MIN;
    int m_maximumPWMSignalLengtoInMicroseconds = chassis_defines::TRAXXAS_PWM_MICROSECONDS_MAX;
    int m_zeroPWMSignalInMicroseconds = chassis_defines::TRAXXAS_PWM_MICROSECONDS_ZERO;
    
    uint8_t m_PWMControlPin = -1;
    Servo m_motorPWM;
    float m_speedConstraintForward = 1.0f;
    float m_speedConstraintReverse = -1.0f;
    bool m_isInitialized = false;

    float m_currentSetSpeed = 0.0f;

    const float m_physicalMinForward = 0.19f;
    const float m_physicalMaxForward = 0.30f;
    const float m_physicalMinReverse = 0.22f;
    const float m_physicalMaxReverse = 0.30f;

    void attachServo()
    {
        m_motorPWM.attach(m_PWMControlPin, m_minimumPWMSignalLengthInMicroseconds, m_maximumPWMSignalLengtoInMicroseconds);
        m_motorPWM.writeMicroseconds(m_zeroPWMSignalInMicroseconds);
    }

public:
    Motor()
    {
    }

    Motor(uint8_t pin, float speedConstraintForward = 1.0f, float speedConstraintReverse = -1.0f)
    {
        Initialize(pin, speedConstraintForward, speedConstraintReverse);
    }

    void Initialize(uint8_t pin, float speedConstraintForward = 1.0f, float speedConstraintReverse = -1.0f)
    {
        m_PWMControlPin = pin;
        m_speedConstraintForward = speedConstraintForward;
        m_speedConstraintReverse = speedConstraintReverse;
        attachServo();
        m_isInitialized = true;
    }

    void SetSpeed(float speed)
    {
        if (m_isInitialized)
        {
            float speedConstrained = constrain(speed, m_speedConstraintReverse, m_speedConstraintForward);
            int pwmSignal = m_zeroPWMSignalInMicroseconds;

            if (speedConstrained > 0.0f)
            {
                float physicalSpeed = m_physicalMinForward + (speedConstrained * (m_physicalMaxForward - m_physicalMinForward));
                pwmSignal = m_zeroPWMSignalInMicroseconds + (int)(physicalSpeed * (m_maximumPWMSignalLengtoInMicroseconds - m_zeroPWMSignalInMicroseconds));
            }
            else if (speedConstrained < 0.0f)
            {
                float absSpeed = -speedConstrained;
                float physicalSpeed = m_physicalMinReverse + (absSpeed * (m_physicalMaxReverse - m_physicalMinReverse));
                pwmSignal = m_zeroPWMSignalInMicroseconds - (int)(physicalSpeed * (m_zeroPWMSignalInMicroseconds - m_minimumPWMSignalLengthInMicroseconds));
            }

            m_motorPWM.writeMicroseconds(pwmSignal);
            m_currentSetSpeed = speedConstrained;
        }
    }

    void SetSpeed(int speed)
    {
        if (m_isInitialized)
        {
            int speedConstrained = constrain(speed, -100, 100);
            float speedConverted = ((float)speedConstrained - (-100.0f)) * (1.0f - (-1.0f)) / (100.0f - (-100.0f))  + (-1.0f);
            SetSpeed(speedConverted);
        }
    }

    // Send the "zero" speed signal to motor. Not the same as braking!
    void StopMotor()
    {
        if (m_isInitialized)
        {
            m_motorPWM.writeMicroseconds(m_zeroPWMSignalInMicroseconds);
            m_currentSetSpeed = 0.0f;
        }
    }

    //TODO: figure out how to do it
    // void Brake()
    // {   

    // }

    float GetCurrentSetSpeed()
    {
        return m_currentSetSpeed;
    }
};