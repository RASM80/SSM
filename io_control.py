import RPi.GPIO as GPIO
import time
import sys

def main():
    """Control the relay board CH1."""
    # Set GPIO mode to BCM
    GPIO.setmode(GPIO.BCM)
    channel = 26  # BCM 26 for CH1
    
    # Set up the channel as output
    GPIO.setup(channel, GPIO.OUT)
    
    try:
        # Turn ON (assuming active HIGH)
        GPIO.output(channel, GPIO.LOW)
        print("Relay CH1 ON")
        
        # Wait for 10 seconds
        time.sleep(10)
        
        # Turn OFF
        GPIO.output(channel, GPIO.HIGH)
        print("Relay CH1 OFF")
    finally:
        # Clean up GPIO settings
        GPIO.cleanup()

if __name__ == "__main__":
    main()
