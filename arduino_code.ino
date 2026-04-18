const int pwmPin = 9;
const int dirPin = 7;
const int enPin  = 8;
const int analogPin = A0;

const int MOTOR_SPEED = 255;   // full speed, reduce if too aggressive

void stopMotor() {
  analogWrite(pwmPin, 0);
}

void moveLeft() {
  digitalWrite(enPin, HIGH);
  digitalWrite(dirPin, LOW);   // swap LOW/HIGH if direction is reversed
  analogWrite(pwmPin, MOTOR_SPEED);
}

void moveRight() {
  digitalWrite(enPin, HIGH);
  digitalWrite(dirPin, HIGH);  // swap if direction is reversed
  analogWrite(pwmPin, MOTOR_SPEED);
}

void setup() {
  pinMode(pwmPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  pinMode(analogPin, INPUT);

  digitalWrite(enPin, HIGH);
  stopMotor();

  Serial.begin(9600);
}

void loop() {
  int a0Value = analogRead(analogPin);

  Serial.print("A0 raw: ");
  Serial.println(a0Value);

  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'R') {
      moveLeft();
    }
    else if (c == 'L') {
      moveRight();
    }
    else if (c == 'C' || c == 'S') {
      stopMotor();
    }
  }

  delay(20);
}