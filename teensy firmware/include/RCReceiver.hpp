#pragma once

#include <Arduino.h>
#include <chassis_defines.h>

class RCReceiver
{
private:
    struct ChannelState
    {
        uint8_t pin = 255;
        volatile uint32_t riseTimeMicros = 0;
        volatile uint32_t lastPulseTimeMicros = 0;
        volatile uint16_t pulseWidthMicros = chassis_defines::TRAXXAS_PWM_MICROSECONDS_ZERO;
    };

    static RCReceiver* s_instance;

    ChannelState m_motorChannel = {};
    ChannelState m_steeringChannel = {};

    bool m_isInitialized = false;
    bool m_throttleManualActive = false;
    bool m_steeringAssistActive = false;

    float m_motorCommand = 0.0f;
    float m_steeringCommand = 0.0f;

    uint32_t m_lastSteeringMotionTimeMillis = 0;
    uint16_t m_lastSteeringPulseWidthMicros = chassis_defines::RC_INPUT_NEUTRAL_MICROSECONDS;
    bool m_hasPreviousSteeringPulse = false;

    static void MotorChannelInterruptHandler()
    {
        if (s_instance != nullptr) {
            s_instance->HandleInterrupt(s_instance->m_motorChannel);
        }
    }

    static void SteeringChannelInterruptHandler()
    {
        if (s_instance != nullptr) {
            s_instance->HandleInterrupt(s_instance->m_steeringChannel);
        }
    }

    void HandleInterrupt(ChannelState& channel)
    {
        const uint32_t now = micros();
        const bool isHighLevel = digitalRead(channel.pin) == HIGH;

        if (isHighLevel) {
            channel.riseTimeMicros = now;
            return;
        }

        if (channel.riseTimeMicros == 0) {
            return;
        }

        const uint32_t pulseWidth = now - channel.riseTimeMicros;
        channel.riseTimeMicros = 0;

        if (pulseWidth < chassis_defines::RC_INPUT_VALID_MIN_MICROSECONDS ||
            pulseWidth > chassis_defines::RC_INPUT_VALID_MAX_MICROSECONDS) {
            return;
        }

        channel.pulseWidthMicros = static_cast<uint16_t>(pulseWidth);
        channel.lastPulseTimeMicros = now;
    }

    static float ConvertPulseToNormalizedCommand(uint16_t pulseWidthMicros, bool invert)
    {
        const int minPulse = chassis_defines::TRAXXAX_PWM_MICROSECONDS_MIN;
        const int maxPulse = chassis_defines::TRAXXAS_PWM_MICROSECONDS_MAX;
        const int neutralPulse = chassis_defines::RC_INPUT_NEUTRAL_MICROSECONDS;

        float normalizedValue = 0.0f;
        if (pulseWidthMicros >= neutralPulse) {
            normalizedValue = static_cast<float>(pulseWidthMicros - neutralPulse) /
                              static_cast<float>(maxPulse - neutralPulse);
        } else {
            normalizedValue = static_cast<float>(pulseWidthMicros - neutralPulse) /
                              static_cast<float>(neutralPulse - minPulse);
        }

        normalizedValue = constrain(normalizedValue, -1.0f, 1.0f);
        return invert ? (-1.0f * normalizedValue) : normalizedValue;
    }

    static bool IsThrottleManual(uint16_t pulseWidthMicros)
    {
        const int deviation = static_cast<int>(pulseWidthMicros) - chassis_defines::RC_INPUT_NEUTRAL_MICROSECONDS;
        return abs(deviation) >= chassis_defines::RC_THROTTLE_MANUAL_DEADBAND_MICROSECONDS;
    }

    bool ReadChannelSnapshot(const ChannelState& channel, uint16_t& pulseWidthMicros) const
    {
        uint32_t lastPulseTimeMicros = 0;
        noInterrupts();
        pulseWidthMicros = channel.pulseWidthMicros;
        lastPulseTimeMicros = channel.lastPulseTimeMicros;
        interrupts();

        if (lastPulseTimeMicros == 0) {
            return false;
        }

        const uint32_t pulseAgeMillis = (micros() - lastPulseTimeMicros) / 1000U;
        return pulseAgeMillis <= chassis_defines::RC_INPUT_SIGNAL_TIMEOUT_MILLISECONDS;
    }

public:
    bool Initialize(uint8_t motorPin, uint8_t steeringPin)
    {
        const int motorInterrupt = digitalPinToInterrupt(motorPin);
        const int steeringInterrupt = digitalPinToInterrupt(steeringPin);
        if (motorInterrupt == NOT_AN_INTERRUPT || steeringInterrupt == NOT_AN_INTERRUPT) {
            return false;
        }

        m_motorChannel.pin = motorPin;
        m_steeringChannel.pin = steeringPin;

        pinMode(motorPin, INPUT);
        pinMode(steeringPin, INPUT);

        s_instance = this;
        attachInterrupt(motorInterrupt, MotorChannelInterruptHandler, CHANGE);
        attachInterrupt(steeringInterrupt, SteeringChannelInterruptHandler, CHANGE);

        m_isInitialized = true;
        m_throttleManualActive = false;
        m_steeringAssistActive = false;
        m_motorCommand = 0.0f;
        m_steeringCommand = 0.0f;
        m_lastSteeringMotionTimeMillis = 0;
        m_lastSteeringPulseWidthMicros = chassis_defines::RC_INPUT_NEUTRAL_MICROSECONDS;
        m_hasPreviousSteeringPulse = false;
        return true;
    }

    void Update()
    {
        if (!m_isInitialized) {
            m_throttleManualActive = false;
            m_steeringAssistActive = false;
            m_motorCommand = 0.0f;
            m_steeringCommand = 0.0f;
            return;
        }

        uint16_t motorPulseWidthMicros = chassis_defines::TRAXXAS_PWM_MICROSECONDS_ZERO;
        uint16_t steeringPulseWidthMicros = chassis_defines::TRAXXAS_PWM_MICROSECONDS_ZERO;

        const bool hasMotorSignal = ReadChannelSnapshot(m_motorChannel, motorPulseWidthMicros);
        const bool hasSteeringSignal = ReadChannelSnapshot(m_steeringChannel, steeringPulseWidthMicros);
        const bool hasValidSignal = hasMotorSignal && hasSteeringSignal;

        if (!hasValidSignal) {
            m_throttleManualActive = false;
            m_steeringAssistActive = false;
            m_motorCommand = 0.0f;
            m_steeringCommand = 0.0f;
            m_hasPreviousSteeringPulse = false;
            return;
        }

        m_motorCommand = ConvertPulseToNormalizedCommand(motorPulseWidthMicros, chassis_defines::RC_INPUT_INVERT_MOTOR);
        m_steeringCommand = ConvertPulseToNormalizedCommand(steeringPulseWidthMicros, chassis_defines::RC_INPUT_INVERT_STEERING);

        m_throttleManualActive = IsThrottleManual(motorPulseWidthMicros);

        bool steeringMovedRecently = false;
        if (m_hasPreviousSteeringPulse) {
            const int steeringDelta = abs(static_cast<int>(steeringPulseWidthMicros) - static_cast<int>(m_lastSteeringPulseWidthMicros));
            if (steeringDelta >= chassis_defines::RC_STEERING_MOVEMENT_THRESHOLD_MICROSECONDS) {
                m_lastSteeringMotionTimeMillis = millis();
                steeringMovedRecently = true;
            }
        }
        m_lastSteeringPulseWidthMicros = steeringPulseWidthMicros;
        m_hasPreviousSteeringPulse = true;

        const uint32_t now = millis();
        const bool steeringAssistWindowActive = (now - m_lastSteeringMotionTimeMillis) <= chassis_defines::RC_STEERING_ASSIST_HOLD_MILLISECONDS;
        m_steeringAssistActive = m_throttleManualActive || steeringMovedRecently || steeringAssistWindowActive;
    }

    bool IsInitialized() const
    {
        return m_isInitialized;
    }

    bool HasThrottleManualActive() const
    {
        return m_throttleManualActive;
    }

    bool HasSteeringAssistActive() const
    {
        return m_steeringAssistActive;
    }

    bool HasAnyManualOverride() const
    {
        return m_throttleManualActive || m_steeringAssistActive;
    }

    float GetMotorCommand() const
    {
        return m_motorCommand;
    }

    float GetSteeringCommand() const
    {
        return m_steeringCommand;
    }
};

inline RCReceiver* RCReceiver::s_instance = nullptr;