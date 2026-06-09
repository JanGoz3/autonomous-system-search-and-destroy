#pragma once
#include <Arduino.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>


// Fixed-buffer protocol handler for packets shaped as:
// <command;length;data;checksum>

class SimpleSerialCommunicator
{
public:
    static constexpr uint8_t MAX_COMMAND_LENGTH = 16;
    static constexpr uint16_t MAX_DATA_LENGTH = 128;
    static constexpr uint16_t MAX_FRAME_LENGTH = MAX_COMMAND_LENGTH + MAX_DATA_LENGTH + 16;
    static constexpr uint8_t MAX_QUEUED_MESSAGES = 8;

    struct Message
    {
        char command[MAX_COMMAND_LENGTH + 1] = {0};
        char data[MAX_DATA_LENGTH + 1] = {0};
        uint16_t dataLength = 0;
    };

private:
    bool m_isInitialized = false;
    uint32_t m_baudRate = 115200;
    uint32_t m_serialTimeoutInMilliseconds = 1000;
    uint32_t m_receiveTimeoutInMilliseconds = 100;

    char m_messageStartMarker = '<';
    char m_messageEndMarker = '>';
    char m_messageDelimiter = ';';

    bool m_inPacket = false;
    uint32_t m_packetStartTime = 0;
    char m_frameBuffer[MAX_FRAME_LENGTH + 1] = {0};
    uint16_t m_frameLength = 0;

    Message m_messageQueue[MAX_QUEUED_MESSAGES] = {};
    uint8_t m_queueHead = 0;
    uint8_t m_queueTail = 0;
    uint8_t m_queueCount = 0;

    uint32_t m_lastErrorReportTime = 0;
    char m_lastErrorCode[24] = {0};


    void ResetCurrentPacketState()
    {
        m_inPacket = false;
        m_packetStartTime = 0;
        m_frameLength = 0;
        m_frameBuffer[0] = '\0';
    }

    bool IsValidCommandCharacter(char c) const
    {
        return ((c >= 'a') && (c <= 'z')) || (c == '_');
    }

    bool IsHexCharacter(char c) const
    {
        return ((c >= '0') && (c <= '9')) || ((c >= 'A') && (c <= 'F'));
    }

    uint8_t HexCharacterToValue(char c) const
    {
        if ((c >= '0') && (c <= '9')) {
            return static_cast<uint8_t>(c - '0');
        }
        return static_cast<uint8_t>(10 + (c - 'A'));
    }

    uint8_t CalculateChecksum(const char* command, const char* dataLength, const char* data) const
    {
        uint16_t sum = 0;

        for (const char* cursor = command; *cursor != '\0'; ++cursor) {
            sum += static_cast<uint8_t>(*cursor);
        }
        for (const char* cursor = dataLength; *cursor != '\0'; ++cursor) {
            sum += static_cast<uint8_t>(*cursor);
        }
        for (const char* cursor = data; *cursor != '\0'; ++cursor) {
            sum += static_cast<uint8_t>(*cursor);
        }

        return static_cast<uint8_t>(sum % 256);
    }

    void CheckForReceiveTimeout()
    {
        if (!m_inPacket) {
            return;
        }

        const uint32_t now = millis();
        if ((now - m_packetStartTime) > m_receiveTimeoutInMilliseconds) {
            ResetCurrentPacketState();
            ReportError("timeout");
        }
    }

    void PushMessage(const Message& message)
    {
        if (m_queueCount == MAX_QUEUED_MESSAGES) {
            m_queueHead = static_cast<uint8_t>((m_queueHead + 1) % MAX_QUEUED_MESSAGES);
            m_queueCount--;
            ReportError("overflow");
        }

        m_messageQueue[m_queueTail] = message;
        m_queueTail = static_cast<uint8_t>((m_queueTail + 1) % MAX_QUEUED_MESSAGES);
        m_queueCount++;
    }

    bool TryParseFrame(Message& message)
    {
        int separatorIndices[3] = {-1, -1, -1};
        uint8_t separatorCount = 0;

        for (uint16_t i = 0; i < m_frameLength; ++i) {
            if (m_frameBuffer[i] == m_messageDelimiter) {
                if (separatorCount < 3) {
                    separatorIndices[separatorCount] = i;
                }
                separatorCount++;
            }
        }

        if (separatorCount != 3) {
            ReportError("format");
            return false;
        }

        const uint16_t commandLength = static_cast<uint16_t>(separatorIndices[0]);
        const uint16_t lengthFieldLength = static_cast<uint16_t>(separatorIndices[1] - separatorIndices[0] - 1);
        const uint16_t dataLength = static_cast<uint16_t>(separatorIndices[2] - separatorIndices[1] - 1);
        const uint16_t checksumLength = static_cast<uint16_t>(m_frameLength - separatorIndices[2] - 1);

        if ((commandLength == 0) || (commandLength > MAX_COMMAND_LENGTH) || (lengthFieldLength == 0) || (lengthFieldLength > 3) || (checksumLength != 2)) {
            ReportError("format");
            return false;
        }

        char commandBuffer[MAX_COMMAND_LENGTH + 1] = {0};
        char lengthBuffer[4] = {0};
        char dataBuffer[MAX_DATA_LENGTH + 1] = {0};
        char checksumBuffer[3] = {0};

        memcpy(commandBuffer, m_frameBuffer, commandLength);
        memcpy(lengthBuffer, &m_frameBuffer[separatorIndices[0] + 1], lengthFieldLength);

        if (dataLength > MAX_DATA_LENGTH) {
            ReportError("bad_length");
            return false;
        }
        memcpy(dataBuffer, &m_frameBuffer[separatorIndices[1] + 1], dataLength);
        memcpy(checksumBuffer, &m_frameBuffer[separatorIndices[2] + 1], checksumLength);

        for (uint16_t i = 0; i < commandLength; ++i) {
            if (!IsValidCommandCharacter(commandBuffer[i])) {
                ReportError("format");
                return false;
            }
        }

        for (uint16_t i = 0; i < lengthFieldLength; ++i) {
            if ((lengthBuffer[i] < '0') || (lengthBuffer[i] > '9')) {
                ReportError("bad_length");
                return false;
            }
        }

        for (uint16_t i = 0; i < checksumLength; ++i) {
            if (!IsHexCharacter(checksumBuffer[i])) {
                ReportError("bad_checksum_field");
                return false;
            }
        }

        const uint16_t declaredLength = static_cast<uint16_t>(atoi(lengthBuffer));
        if (declaredLength > MAX_DATA_LENGTH) {
            ReportError("bad_length");
            return false;
        }

        if (declaredLength != dataLength) {
            ReportError("bad_length");
            return false;
        }

        const uint8_t calculatedChecksum = CalculateChecksum(commandBuffer, lengthBuffer, dataBuffer);
        const uint8_t receivedChecksum = static_cast<uint8_t>((HexCharacterToValue(checksumBuffer[0]) << 4) | HexCharacterToValue(checksumBuffer[1]));
        if (calculatedChecksum != receivedChecksum) {
            ReportError("checksum");
            return false;
        }

        memcpy(message.command, commandBuffer, commandLength + 1);
        memcpy(message.data, dataBuffer, dataLength + 1);
        message.dataLength = dataLength;
        return true;
    }

    void ProcessIncomingByte(char c)
    {
        if (c == m_messageStartMarker) {
            m_inPacket = true;
            m_packetStartTime = millis();
            m_frameLength = 0;
            m_frameBuffer[0] = '\0';
            return;
        }

        if (!m_inPacket) {
            return;
        }

        if (c == m_messageEndMarker) {
            Message message;
            const bool parsed = TryParseFrame(message);
            ResetCurrentPacketState();
            if (parsed) {
                PushMessage(message);
            }
            return;
        }

        if (m_frameLength >= MAX_FRAME_LENGTH) {
            ResetCurrentPacketState();
            ReportError("overflow");
            return;
        }

        m_frameBuffer[m_frameLength++] = c;
        m_frameBuffer[m_frameLength] = '\0';
    }

    void FormatLengthField(uint16_t dataLength, char* destination, size_t destinationSize) const
    {
        if (dataLength < 100) {
            snprintf(destination, destinationSize, "%02u", static_cast<unsigned>(dataLength));
            return;
        }

        snprintf(destination, destinationSize, "%u", static_cast<unsigned>(dataLength));
    }

    bool ShouldReportError(const char* errorCode)
    {
        const uint32_t now = millis();
        if ((strncmp(m_lastErrorCode, errorCode, sizeof(m_lastErrorCode) - 1) == 0) && ((now - m_lastErrorReportTime) < 250U)) {
            return false;
        }

        strncpy(m_lastErrorCode, errorCode, sizeof(m_lastErrorCode) - 1);
        m_lastErrorCode[sizeof(m_lastErrorCode) - 1] = '\0';
        m_lastErrorReportTime = now;
        return true;
    }

public:
    void Initialize(int baudRate)
    {
        uint32_t elapsedTime = 0;
        uint32_t startTime = millis();
        Serial.begin(baudRate);
        while (!Serial && elapsedTime < m_serialTimeoutInMilliseconds) {
            elapsedTime = millis() - startTime;
        }

        m_baudRate = baudRate;
        m_isInitialized = true;
        ResetCurrentPacketState();
    }

    SimpleSerialCommunicator()
    {
    }

    SimpleSerialCommunicator(int baudRate, char startMarker = '<', char endMarker = '>', char delimiter = ';')
        : m_baudRate(baudRate), m_messageStartMarker(startMarker), m_messageEndMarker(endMarker), m_messageDelimiter(delimiter)
    {
        Initialize(baudRate);
    }

    void Update()
    {
        CheckForReceiveTimeout();

        while (Serial.available()) {
            char c = Serial.read();
            ProcessIncomingByte(c);
        }

        CheckForReceiveTimeout();
    }

    bool HasNewData()
    {
        return m_queueCount > 0;
    }

    bool GetData(Message& message)
    {
        if (m_queueCount == 0) {
            return false;
        }

        message = m_messageQueue[m_queueHead];
        m_queueHead = static_cast<uint8_t>((m_queueHead + 1) % MAX_QUEUED_MESSAGES);
        m_queueCount--;
        return true;
    }

    bool SendCommand(const char* command, const char* data = "")
    {
        if (!m_isInitialized || (command == nullptr) || (data == nullptr)) {
            return false;
        }

        const size_t commandLength = strlen(command);
        const size_t dataLength = strlen(data);
        if ((commandLength == 0) || (commandLength > MAX_COMMAND_LENGTH) || (dataLength > MAX_DATA_LENGTH)) {
            return false;
        }

        for (size_t index = 0; index < commandLength; ++index) {
            if (!IsValidCommandCharacter(command[index])) {
                return false;
            }
        }

        char lengthBuffer[4] = {0};
        FormatLengthField(static_cast<uint16_t>(dataLength), lengthBuffer, sizeof(lengthBuffer));

        const uint8_t checksum = CalculateChecksum(command, lengthBuffer, data);
        char packetBuffer[MAX_FRAME_LENGTH + 1] = {0};
        const int written = snprintf(packetBuffer,
                                     sizeof(packetBuffer),
                                     "%c%s%c%s%c%s%c%02X%c",
                                     m_messageStartMarker,
                                     command,
                                     m_messageDelimiter,
                                     lengthBuffer,
                                     m_messageDelimiter,
                                     data,
                                     m_messageDelimiter,
                                     checksum,
                                     m_messageEndMarker);

        if ((written <= 0) || (written >= static_cast<int>(sizeof(packetBuffer)))) {
            return false;
        }

        Serial.print(packetBuffer);
        return true;
    }

    bool ReportError(const char* errorCode)
    {
        if (!ShouldReportError(errorCode)) {
            return false;
        }

        return SendCommand("error", errorCode);
    }

};

