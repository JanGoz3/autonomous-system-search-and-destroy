#pragma once
#include <Arduino.h>
#include <cstdio>
#include <Wire.h>
#include "MPU9250.h"




class WaveshareIMU
{
    private:

        MPU9250Setting mpuSettings;
        MPU9250 mpu;
        bool m_isInitialized = false;
        bool m_hasValidData = false;


        unsigned long int m_timeOfPreviousUpdate = 0;


        float yaw = 0.0f;
        float pitch = 0.0f;
        float roll = 0.0f;

        float accel_x = 0.0f;
        float accel_y = 0.0f;
        float accel_z = 0.0f;

        float gyro_x = 0.0f;
        float gyro_y = 0.0f;
        float gyro_z = 0.0f;

    

    void UpdateValues()
    {
        mpu.update();
        yaw = mpu.getYaw();
        pitch = mpu.getPitch();
        roll = mpu.getRoll();
        
        accel_x = mpu.getAccX();
        accel_y = mpu.getAccY();
        accel_z = mpu.getAccZ();

        gyro_x = mpu.getGyroX();
        gyro_y = mpu.getGyroY();
        gyro_z = mpu.getGyroZ();

    }

    public:
        WaveshareIMU()
        {

        }

        bool Initialize(int address = 0x68)
        {   
            Wire1.begin();
            
            if (mpu.setup(address, mpuSettings , Wire1)) // Wire1 - Pins 16 and 17 on Teensy 4.1
            {
                m_isInitialized = true;
                return true;
            }

            return false;
        }


        void Update()
        {
            int timeSinceLastUpdate = millis() - m_timeOfPreviousUpdate;
            if (m_isInitialized && timeSinceLastUpdate > 25)
            {
                UpdateValues();
                m_hasValidData = true;
                m_timeOfPreviousUpdate = millis();
            }
        }

        bool IsInitialized() const
        {
            return m_isInitialized;
        }

        bool HasValidData() const
        {
            return m_isInitialized && m_hasValidData;
        }

        bool FormatCsv(char* destination, size_t destinationSize) const
        {
            if (!HasValidData() || destination == nullptr || destinationSize == 0) {
                return false;
            }

            const int written = snprintf(destination,
                                         destinationSize,
                                         "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                                         yaw,
                                         pitch,
                                         roll,
                                         accel_x,
                                         accel_y,
                                         accel_z,
                                         gyro_x,
                                         gyro_y,
                                         gyro_z);
            return written > 0 && static_cast<size_t>(written) < destinationSize;
        }


        String GetValues()
        {
            if (HasValidData())
            {
                char output[160] = {0};
                if (FormatCsv(output, sizeof(output))) {
                    return String(output);
                }
            }
            return "no data";
        }


};