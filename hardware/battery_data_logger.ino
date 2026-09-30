/*
  EV Battery SoH Project - Data Logger
  Board: ESP32
  Sensors: INA219 (voltage/current), DS18B20 (temperature)
  Logs data to Serial (CSV format) -> copy to Excel/CSV file
  Optional: add SD card module to log directly to file

  Wiring:
  INA219   -> ESP32
  VCC      -> 3.3V
  GND      -> GND
  SDA      -> GPIO 21
  SCL      -> GPIO 22

  DS18B20  -> ESP32
  Data pin -> GPIO 4 (with 4.7k ohm pull-up resistor to 3.3V)
  VCC      -> 3.3V
  GND      -> GND

  MOSFET Gate (discharge control) -> GPIO 5
*/

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------------- Pin Setup ----------------
#define ONE_WIRE_BUS 4
#define MOSFET_PIN 5

Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

// ---------------- Settings ----------------
unsigned long logInterval = 1000;   // log every 1 second
unsigned long startTime;
float initialCapacity_mAh = 2500.0; // rated capacity of your cell (check datasheet)
float consumedCapacity_mAh = 0.0;
unsigned long lastMillis = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    while (1) { delay(10); }
  }

  tempSensor.begin();
  pinMode(MOSFET_PIN, OUTPUT);
  digitalWrite(MOSFET_PIN, LOW); // discharge OFF initially

  startTime = millis();
  lastMillis = millis();

  // CSV header
  Serial.println("time_s,voltage_V,current_mA,power_mW,temperature_C,SoH_percent");
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastMillis >= logInterval) {
    float dt_hours = (currentMillis - lastMillis) / 3600000.0; // ms to hours
    lastMillis = currentMillis;

    // Read sensors
    float busVoltage = ina219.getBusVoltage_V();
    float current_mA = ina219.getCurrent_mA();
    float power_mW = ina219.getPower_mW();

    tempSensor.requestTemperatures();
    float tempC = tempSensor.getTempCByIndex(0);

    // Coulomb counting for capacity used
    if (current_mA > 0) { // discharging
      consumedCapacity_mAh += current_mA * dt_hours;
    }

    float SoH = 100.0 - (consumedCapacity_mAh / initialCapacity_mAh * 100.0);
    if (SoH < 0) SoH = 0;

    float elapsed_s = (currentMillis - startTime) / 1000.0;

    // Print CSV row
    Serial.print(elapsed_s, 2); Serial.print(",");
    Serial.print(busVoltage, 3); Serial.print(",");
    Serial.print(current_mA, 3); Serial.print(",");
    Serial.print(power_mW, 3); Serial.print(",");
    Serial.print(tempC, 2); Serial.print(",");
    Serial.println(SoH, 3);

    // Safety cutoff (adjust for your cell chemistry)
    if (busVoltage < 3.0) {
      digitalWrite(MOSFET_PIN, LOW); // stop discharge
      Serial.println("Low voltage cutoff reached. Discharge stopped.");
    }
  }
}

/*
  HOW TO USE:
  1. Install libraries in Arduino IDE:
     - Adafruit INA219
     - OneWire
     - DallasTemperature
  2. Select Board: ESP32 Dev Module, correct COM port
  3. Upload this code
  4. Open Serial Monitor (115200 baud) OR Serial Plotter
  5. Copy printed CSV data -> paste into a .csv file -> use in Python (pandas.read_csv)
  6. Repeat for multiple charge-discharge cycles to build your dataset
  7. Update initialCapacity_mAh to match your actual cell's rated capacity
*/
