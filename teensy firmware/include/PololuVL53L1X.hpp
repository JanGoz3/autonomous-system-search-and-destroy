#pragma once
#include <Arduino.h>
#include <cstdio>
#include <Wire.h>
#include <VL53L1X.h>



class PololuVL53L1X
{
    private:

        VL53L1X sensor;

        bool m_isInitialized = false;
        bool m_hasValidData = false;


        unsigned long int m_timeOfPreviousUpdate = 0;

        const int timingBudgetUs = 50000;
        const int timingPeriodMs = timingBudgetUs / 1000;

        uint16_t distance = 0;



    

    void UpdateValues()
    {
        if (!sensor.dataReady()) {
            return;
        }

        distance = sensor.read(false);
        m_hasValidData = true;

    }

    public:
        PololuVL53L1X()
        {

        }

        bool Initialize(int address = 0x29)
        {   
            (void)address;
            Wire.begin();
            Wire.setClock(400000); // use 400 kHz I2C
            sensor.setTimeout(500);

            if (!sensor.init())
            {
                return false;
            }

            m_isInitialized = true;
            sensor.setDistanceMode(VL53L1X::Long);
            sensor.setMeasurementTimingBudget(timingBudgetUs);

            // Start continuous readings at a rate of one measurement every 50 ms. This period
            // should be at least as long as the timing budget.
            sensor.startContinuous(timingPeriodMs);
            return true;
        }

        void Update()
        {
            int timeSinceLastUpdate = millis() - m_timeOfPreviousUpdate;
            if (m_isInitialized && timeSinceLastUpdate > timingPeriodMs)
            {
                UpdateValues();
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
                                         "%u",
                                         static_cast<unsigned int>(distance));
            return written > 0 && static_cast<size_t>(written) < destinationSize;
        }


        String GetValues()
        {
            if (HasValidData())
            {
                char output[16] = {0};
                if (FormatCsv(output, sizeof(output))) {
                    return String(output);
                }
            }
            return "no TOF data";
        }


};