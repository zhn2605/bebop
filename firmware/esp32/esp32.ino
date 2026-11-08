// Byte to Bite ESP32 hub: e-stop button, ultrasonic distance, contact inputs, OLED + 3 status LEDs.
//
// USB serial at 115200, one JSON object per line (.config/docs/CORE.md lines 202-207):
//   ESP32 -> laptop every 50 ms:  {"t": ms, "estop": 0|1, "dist_cm": 23.4|null, "contact": 0|1}
//   ESP32 -> laptop on a press:   {"event": "ESTOP"}  (once per press)
//   laptop -> ESP32:              STATE <NAME>  (hw/esp32.py re-sends the state every 250 ms)
// Names: READY LISTENING THINKING MOVING PRESENTING STOPPED HELP (hw/esp32.py) and the older
// SCOOPING DELIVERING OPEN_WIDE (CORE.md line 206). Lines that aren't STATE lines are ignored.
//
// E-stop reset rule: a press latches "estop": 1. The latch clears once the laptop has answered with
// STATE STOPPED (any time after the press) and the button is released. Held = still 1. The laptop's
// own stop flag stays latched until the operator resumes, so it doesn't need this one too.
// Clearing on STATE READY instead would deadlock with hw/esp32.py: it trips stop on every
// "estop": 1 line, so after "continue" the robot is re-stopped before it can get home to READY.
//
// Watchdog: no STATE line for 2 s -> "NO LINK" on the OLED and a slow red blink. Telemetry keeps
// going; the laptop runs its own 500 ms watchdog on it.
//
// Libraries (Arduino Library Manager): Adafruit SSD1306, Adafruit GFX Library.
// Board: "ESP32 Dev Module" (esp32:esp32:esp32). See firmware/README.md.

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Pins, from CORE.md "ESP32 hub pin map" (lines 167-201).
const int PIN_ESTOP = 27;       // line 172: button to GND, INPUT_PULLUP, pressed = LOW
const int PIN_TRIG = 5;         // line 175: direct
const int PIN_ECHO = 18;        // line 178-179: through the 1k/2k divider (5 V -> 3.3 V)
const int PIN_SDA = 21;         // line 184: OLED SDA
const int PIN_SCL = 22;         // line 184: OLED SCL
const int PIN_LED_GREEN = 25;   // line 186-187: green (ready)
const int PIN_LED_YELLOW = 26;  // line 189-190: yellow (thinking); there is no blue LED
const int PIN_LED_RED = 33;     // line 192-193: red (stopped)
const int PIN_CONTACT = 32;     // line 196-197: mount button to GND, INPUT_PULLUP; optional
const int PIN_PIEZO = 34;       // line 199-200: piezo, 1 MΩ to GND; input-only, ADC1; optional

// 0 when the piezo (and its 1 MΩ) isn't wired: a floating GPIO34 reads noise -> random contact.
#define PIEZO_ENABLED 1

const uint32_t TELEMETRY_MS = 50;
const uint32_t PING_MS = 60;
const uint32_t ECHO_TIMEOUT_US = 25000;  // ~430 cm round trip; longer = no echo = null
const float CM_PER_US = 0.01715;         // 343 m/s, halved for the round trip
const uint32_t DEBOUNCE_MS = 30;
const uint32_t CONTACT_HOLD_MS = 200;    // a tap or switch closure reads as contact for this long
const int PIEZO_THRESHOLD = 400;         // raw 12-bit ADC (0-4095); a tap spikes well above this
const uint32_t LINK_TIMEOUT_MS = 2000;
const uint32_t OLED_MS = 200;            // a full redraw blocks ~25 ms on I2C, so not every loop

const int OLED_W = 128;
const int OLED_H = 64;
Adafruit_SSD1306 oled(OLED_W, OLED_H, &Wire, -1);
uint8_t oledAddr = 0;  // 0 = no display found; everything else keeps running

// LED looks per state.
enum Blink : uint8_t { DARK, LIT, SLOW, BLINK, FAST };

struct Look {
  const char *name;
  Blink green, yellow, red;
};

const Look LOOKS[] = {
    {"READY", LIT, DARK, DARK},
    {"LISTENING", SLOW, DARK, DARK},
    {"THINKING", DARK, LIT, DARK},
    {"MOVING", DARK, BLINK, DARK},
    {"SCOOPING", DARK, BLINK, DARK},
    {"DELIVERING", DARK, BLINK, DARK},
    {"PRESENTING", LIT, LIT, DARK},
    {"OPEN_WIDE", LIT, LIT, DARK},
    {"STOPPED", DARK, DARK, LIT},
    {"HELP", DARK, DARK, FAST},
};
const Look LOOK_BOOT = {"BOOT", LIT, LIT, LIT};  // lamp test until the first STATE line
const Look LOOK_NO_LINK = {"NO LINK", DARK, DARK, SLOW};
const Look LOOK_ESTOP = {"E-STOP", DARK, DARK, LIT};
const Look LOOK_UNKNOWN = {"", DARK, DARK, DARK};

// State from the laptop.
char stateName[16] = "BOOT";
const Look *stateLook = &LOOK_BOOT;
uint32_t lastStateMs = 0;

// E-stop: debounced, latched.
bool btnRaw = HIGH, btnStable = HIGH;  // starts "released", so a button held at boot still latches
uint32_t btnChangedMs = 0;
bool estopLatched = false, estopAcked = false;

// Ultrasonic: last 3 readings, NAN = no echo.
float pings[3] = {NAN, NAN, NAN};
uint8_t pingIdx = 0;
uint32_t lastPingMs = 0;

bool contactSeen = false;
uint32_t lastContactMs = 0;

uint32_t lastTelemetryMs = 0, lastOledMs = 0;

char rxBuf[48];
uint8_t rxLen = 0;
bool rxOverflow = false;

bool lit(Blink b, uint32_t now) {
  switch (b) {
    case LIT: return true;
    case SLOW: return now % 1000 < 500;
    case BLINK: return now % 500 < 250;
    case FAST: return now % 200 < 100;
    default: return false;
  }
}

// What the OLED and LEDs show right now: e-stop > lost link > the laptop's state.
const Look *currentLook(uint32_t now, const char **name) {
  if (estopLatched) {
    *name = LOOK_ESTOP.name;
    return &LOOK_ESTOP;
  }
  if (now - lastStateMs >= LINK_TIMEOUT_MS) {
    *name = LOOK_NO_LINK.name;
    return &LOOK_NO_LINK;
  }
  *name = stateName;
  return stateLook;
}

void setState(const char *name) {
  lastStateMs = millis();
  strncpy(stateName, name, sizeof(stateName) - 1);
  stateName[sizeof(stateName) - 1] = '\0';
  stateLook = &LOOK_UNKNOWN;  // shown as text, LEDs off
  for (const Look &l : LOOKS) {
    if (strcmp(l.name, stateName) == 0) {
      stateLook = &l;
      break;
    }
  }
  if (estopLatched && strcmp(stateName, "STOPPED") == 0) estopAcked = true;
}

void handleLine(char *line) {
  // Trim, upper-case, then accept only "STATE <NAME>".
  while (*line == ' ' || *line == '\t') line++;
  int n = strlen(line);
  while (n > 0 && (line[n - 1] == ' ' || line[n - 1] == '\t')) line[--n] = '\0';
  for (int i = 0; i < n; i++) line[i] = toupper((unsigned char)line[i]);
  if (strncmp(line, "STATE ", 6) != 0) return;
  char *name = line + 6;
  while (*name == ' ') name++;
  if (*name) setState(name);
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (!rxOverflow) {
        rxBuf[rxLen] = '\0';
        handleLine(rxBuf);
      }
      rxLen = 0;
      rxOverflow = false;
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxOverflow = true;  // too long to be a STATE line: drop it
    }
  }
}

void readButton(uint32_t now) {
  bool raw = digitalRead(PIN_ESTOP);
  if (raw != btnRaw) {
    btnRaw = raw;
    btnChangedMs = now;
  }
  if (raw != btnStable && now - btnChangedMs >= DEBOUNCE_MS) {
    btnStable = raw;
    if (btnStable == LOW && !estopLatched) {
      estopLatched = true;
      estopAcked = false;
      Serial.print("{\"event\": \"ESTOP\"}\n");
    }
  }
  if (estopLatched && estopAcked && btnStable == HIGH) estopLatched = false;
}

void readContact(uint32_t now) {
  bool hit = digitalRead(PIN_CONTACT) == LOW;
#if PIEZO_ENABLED
  hit = hit || analogRead(PIN_PIEZO) > PIEZO_THRESHOLD;
#endif
  if (hit) {
    contactSeen = true;
    lastContactMs = now;
  }
}

bool contactNow(uint32_t now) {
  return contactSeen && now - lastContactMs < CONTACT_HOLD_MS;
}

// Blocks for up to ECHO_TIMEOUT_US.
void ping() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, ECHO_TIMEOUT_US);
  pings[pingIdx] = us ? us * CM_PER_US : NAN;
  pingIdx = (pingIdx + 1) % 3;
}

// Median of the last 3 pings, no echo counting as "far": NAN once 2 of the 3 had no echo.
float distCm() {
  float a = isnan(pings[0]) ? INFINITY : pings[0];
  float b = isnan(pings[1]) ? INFINITY : pings[1];
  float c = isnan(pings[2]) ? INFINITY : pings[2];
  float m = max(min(a, b), min(max(a, b), c));
  return isinf(m) ? NAN : m;
}

void sendTelemetry(uint32_t now) {
  float d = distCm();
  if (isnan(d)) {
    Serial.printf("{\"t\": %lu, \"estop\": %d, \"dist_cm\": null, \"contact\": %d}\n",
                  (unsigned long)now, estopLatched ? 1 : 0, contactNow(now) ? 1 : 0);
  } else {
    Serial.printf("{\"t\": %lu, \"estop\": %d, \"dist_cm\": %.1f, \"contact\": %d}\n",
                  (unsigned long)now, estopLatched ? 1 : 0, d, contactNow(now) ? 1 : 0);
  }
}

void showLeds(const Look *look, uint32_t now) {
  digitalWrite(PIN_LED_GREEN, lit(look->green, now));
  digitalWrite(PIN_LED_YELLOW, lit(look->yellow, now));
  digitalWrite(PIN_LED_RED, lit(look->red, now));
}

bool oledPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void initOled() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setTimeOut(20);  // ms; a wedged bus must not stall telemetry
  for (uint8_t addr : {0x3C, 0x3D}) {
    if (oledPresent(addr) && oled.begin(SSD1306_SWITCHCAPVCC, addr, true, false)) {
      oledAddr = addr;
      break;
    }
  }
}

// State name large (centred), dist and estop small. Redraws only when the text changed.
void drawOled(const char *name) {
  if (!oledAddr) return;
  static char lastBig[16] = "", lastSmall[24] = "";
  char small[24];
  float d = distCm();
  if (isnan(d)) {
    snprintf(small, sizeof(small), "dist --  estop %d", estopLatched ? 1 : 0);
  } else {
    snprintf(small, sizeof(small), "dist %.1fcm  estop %d", d, estopLatched ? 1 : 0);
  }
  if (strcmp(name, lastBig) == 0 && strcmp(small, lastSmall) == 0) return;
  if (!oledPresent(oledAddr)) return;  // unplugged mid-run: skip the ~1 KB write
  strncpy(lastBig, name, sizeof(lastBig) - 1);
  strncpy(lastSmall, small, sizeof(lastSmall) - 1);

  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  int len = strlen(name);
  int size = len <= 7 ? 3 : 2;  // 18 px/char fits 7 across, 12 px/char fits 10
  int x = (OLED_W - len * 6 * size) / 2;
  oled.setTextSize(size);
  oled.setCursor(x > 0 ? x : 0, size == 3 ? 10 : 14);
  oled.print(name);
  oled.setTextSize(1);
  oled.setCursor(0, OLED_H - 10);
  oled.print(small);
  oled.display();
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  pinMode(PIN_CONTACT, INPUT_PULLUP);
  pinMode(PIN_PIEZO, INPUT);  // no internal pulls on GPIO34-39; the 1 MΩ is the pull-down
  pinMode(PIN_TRIG, OUTPUT);
  digitalWrite(PIN_TRIG, LOW);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  initOled();
}

void loop() {
  uint32_t now = millis();
  readSerial();
  readButton(now);
  readContact(now);

  if (now - lastTelemetryMs >= TELEMETRY_MS) {
    lastTelemetryMs = now;
    sendTelemetry(now);
  }

  const char *name;
  const Look *look = currentLook(now, &name);
  showLeds(look, now);

  // At most one blocking job (ping <= 25 ms, OLED ~25 ms) per pass keeps telemetry on time.
  if (now - lastPingMs >= PING_MS) {
    lastPingMs = now;
    ping();
  } else if (now - lastOledMs >= OLED_MS) {
    lastOledMs = now;
    drawOled(name);
  }
}
