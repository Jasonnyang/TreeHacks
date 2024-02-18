#include <Servo.h>
#define TRIG_PIN 8
#define ECHO_PIN 11

Servo servoLeft;
Servo servoRight;
int pinServoLeft = 9;
int pinServoRight = 10;
Servo servobackLeft;
Servo servobackRight;
int pinServobackLeft = 12;
int pinServobackRight = 13;

void setup() {
  servoLeft.attach(pinServoLeft);
  servoRight.attach(pinServoRight);
  servobackLeft.attach(pinServobackLeft);
  servobackRight.attach(pinServobackRight);
  pinMode(TRIG_PIN, OUTPUT); 
  pinMode(ECHO_PIN, INPUT); 
  Serial.begin(9600); // Start serial communication at 9600 baud rate
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the incoming command
    long duration, distance;
    digitalWrite(TRIG_PIN, LOW);  
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    duration = pulseIn(ECHO_PIN, HIGH);
    distance = duration * 0.034 / 2;
    Serial.print("Distance: ");
    Serial.println(distance);
    controlRobot(command, distance); // Control the robot based on the command
    
    delay(1000);
  }
}

void controlRobot(String command, long distance) {
  if (command == "up" && distance > 5) {
    // Forward
    servoLeft.writeMicroseconds(1400); // Adjust as needed for your servo
    servoRight.writeMicroseconds(1600); // Reverse direction for the other servo
    servobackLeft.writeMicroseconds(1400); // Adjust as needed for your servo
    servobackRight.writeMicroseconds(1600); // Reverse direction for the other servo
  } else if (command == "down") {
    // Backward
    servoLeft.writeMicroseconds(1600);
    servoRight.writeMicroseconds(1400);
    servobackLeft.writeMicroseconds(1600);
    servobackRight.writeMicroseconds(1400);
  } else if (command == "left" && distance > 1) {
    // Turn left
    servoLeft.writeMicroseconds(1600); // Both servos in the same direction
    servoRight.writeMicroseconds(1600);
    servobackLeft.writeMicroseconds(1600); // Both servos in the same direction
    servobackRight.writeMicroseconds(1600);
  } else if (command == "right" && distance > 1) {
    // Turn right
    servoLeft.writeMicroseconds(1400); // Both servos in the same direction
    servoRight.writeMicroseconds(1400);
    servobackLeft.writeMicroseconds(1400); // Both servos in the same direction
    servobackRight.writeMicroseconds(1400);
  } else {
    // Stop
    servoLeft.writeMicroseconds(1500); // Neutral position - servo stops
    servoRight.writeMicroseconds(1500); // Neutral position - servo stops
    servobackLeft.writeMicroseconds(1500); // Neutral position - servo stops
    servobackRight.writeMicroseconds(1500); // Neutral position - servo stops
  }
}