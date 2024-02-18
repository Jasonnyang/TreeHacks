#include <Servo.h>

Servo servoLeft;  // Create servo object for the left servo
Servo servoRight; // Create servo object for the right servo

void setup() {
  servoLeft.attach(9);  // Attaches the left servo on pin 9
  servoRight.attach(10); // Attaches the right servo on pin 10
  Serial.begin(9600); // Start serial communication at 9600 baud rate
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the incoming command
    controlRobot(command); // Control the robot based on the command
  }
}

void controlRobot(String command) {
  if (command == "up") {
    // Forward
    servoLeft.writeMicroseconds(1400); // Adjust as needed for your servo
    servoRight.writeMicroseconds(1600); // Reverse direction for the other servo
  } else if (command == "down") {
    // Backward
    servoLeft.writeMicroseconds(1600);
    servoRight.writeMicroseconds(1400);
  } else if (command == "left") {
    // Turn left
    servoLeft.writeMicroseconds(1600); // Both servos in the same direction
    servoRight.writeMicroseconds(1600);
  } else if (command == "right") {
    // Turn right
    servoLeft.writeMicroseconds(1400); // Both servos in the same direction
    servoRight.writeMicroseconds(1400);
  } else {
    // Stop
    servoLeft.writeMicroseconds(1500); // Neutral position - servo stops
    servoRight.writeMicroseconds(1500); // Neutral position - servo stops
  }
}
