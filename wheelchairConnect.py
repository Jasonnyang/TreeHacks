import serial
import time

# Establish a serial connection (Adjust 'COM3' and baud rate as needed)
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish

def send_command(command):
    ser.write((command + '\n').encode())  # Send the command to the Arduino, add newline as a delimiter
    time.sleep(1)  # Wait for the Arduino to process the command

# Example usage based on user input
while True:
    cmd = input("Enter command (up, down, left, right): ")
    send_command(cmd)
    if cmd == "quit":
        break

ser.close()  # Close the serial connection when done
