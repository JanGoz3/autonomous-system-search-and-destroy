#pragma once
#include "helper_variables.h"



void _softRestart() 
{
  Serial.end();  //clears the serial monitor  if used
  SCB_AIRCR = 0x05FA0004;  //write value for restart
}

void resetTeensy()
{
	Serial.println("Ok.");
	_softRestart();
	while(1);
}

void initLED()
{
    pinMode(LED_BUILTIN_TEENSY, OUTPUT); 
}

void initSerial(int baudRate = 9600)
{
    Serial.begin(baudRate);
    delay(1000);
}




// Motor Control Functions

// void MotorL_Brake()      
// {
//   digitalWrite(IN1,LOW); 
//   digitalWrite(IN2,LOW); 
// }  

// void MotorR_Brake()
// {
//   digitalWrite(IN3,LOW); 
//   digitalWrite(IN4,LOW); 
// }

// void Brake()
// {
//   MotorL_Brake();
//   MotorR_Brake();
// }


// void MotorL_Move(int level) 
// {
//   if(level < 0)
//   {
//     digitalWrite(IN1,HIGH); 
//     digitalWrite(IN2,LOW);  
//     analogWrite(ENA,-level);
//   }
//   else if (level > 0)
//   {
//     digitalWrite(IN1,LOW);
//     digitalWrite(IN2,HIGH);  
//     analogWrite(ENA,level);
//   }
//   else
//   {
//     MotorL_Brake();
//   }

// }

// void MotorR_Move(int level) 
// {
//   if(level < 0)
//   {
//     digitalWrite(IN3,HIGH); 
//     digitalWrite(IN4,LOW);  
//     analogWrite(ENB,-level);
//   }
//   else if (level > 0)
//   {
//     digitalWrite(IN3,LOW); 
//     digitalWrite(IN4,HIGH);  
//     analogWrite(ENB,level);
//   }
//   else
//   {
//     MotorR_Brake();
//   }

// }


// void SetPowerLevel(EngineSelector side, int level, int levelConstaint = 150)
// {
// 	level = constrain(level, -levelConstaint, levelConstaint);

// 	if (side == EngineSelector::Left) 
//   {
// 		MotorL_Move(level);
// 	}
// 	if (side == EngineSelector::Right) 
//   {
// 		MotorR_Move(level);
// 	}	
// }

// Servo Control Functions

// void calibrateServo(ServoSelector servo, int minPosition, int maxPosition, int centerPosition)
// {
//   if (servo == ServoSelector::Yaw)
//   {
//     yawMin = minPosition;
//     yawMax = maxPosition;
//     yawCenter = centerPosition;
//   }
//   if (servo == ServoSelector::Pitch)
//   {
//     pitchMin = minPosition;
//     pitchMax = maxPosition;
//     pitchCenter = centerPosition;
//   }

// }

// void centerServos()
// {

//   int m_yawCenter = constrain(yawCenter, yawMin, yawMax);
//   int m_pitchCenter = constrain(pitchCenter, pitchMin, pitchMax);

//   servoYaw.write(m_yawCenter);
//   servoPitch.write(m_pitchCenter);

// }

// void moveServo(ServoSelector servo, int position)
// {

//   int yawConstraint = constrain(position, yawMin, yawMax);
//   int pitchConstrain = constrain(position, pitchMin, pitchMax);

//   if (servo == ServoSelector::Yaw)
//   {
//     servoYaw.write(yawConstraint);
//   }
//   if (servo == ServoSelector::Pitch)
//   {
//     servoPitch.write(pitchConstrain);
//   }

// }


// //Initialization functions
// void initServos()
// {
//     servoYaw.attach(20);
//     servoPitch.attach(21);
//     centerServos();
// }


// void initESP826()
// {
//     //Activate ESP8266
//     pinMode(ESP8266_RST, OUTPUT);
//     pinMode(ESP8266_CH_PD, OUTPUT);

//     digitalWrite(ESP8266_RST, HIGH);
//     digitalWrite(ESP8266_CH_PD, HIGH);
// }

// void initMotors()
// {
//     pinMode(IN1, OUTPUT);
//     pinMode(IN2, OUTPUT);
//     pinMode(ENA, OUTPUT);

//     pinMode(IN4, OUTPUT);
//     pinMode(IN3, OUTPUT);
//     pinMode(ENB, OUTPUT);
// }



