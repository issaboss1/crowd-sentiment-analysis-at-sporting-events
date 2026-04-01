//code from https://www.youtube.com/watch?v=TYzY9gv98IU

const int ledPin = 26;       // Pin to which LED is connected (you can change this pin)
const int buttonPin = 27;    // Pin to which button is connected (you can change this pin)

bool ledOn = false;          // Flag to indicate LED status

unsigned long lastDebounceTime = 0;  // Last time the button pin was toggled
unsigned long debounceDelay = 50;    // Debounce time in milliseconds

void setup() {
  pinMode(ledPin, OUTPUT);     // Set LED pin as output
  pinMode(buttonPin, INPUT_PULLUP);   // Set button pin as input with pull-up resistor
}

void loop() {
  int buttonState = digitalRead(buttonPin);

  // Check if the button is pressed and debounce the input
  if (buttonState == LOW && (millis() - lastDebounceTime) > debounceDelay) {
    lastDebounceTime = millis();

    // Button is pressed
    ledOn = !ledOn;  // Toggle LED status

    // Update LED state
    digitalWrite(ledPin, ledOn ? HIGH : LOW);
  }
}