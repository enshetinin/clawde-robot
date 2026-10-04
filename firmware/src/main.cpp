// CLAWDE diagnostic firmware: protocol v1 WITHOUT servo output.
// - Non-blocking serial reader with a bounded buffer (no readString, no long delay()).
// - Motion commands (HOME, GRIP, MOVE) answer ERR NOT_CONFIGURED.
// - ARM/DISARM only change a logical flag; a communication watchdog drops it.
// - STOP is always accepted and would interrupt interpolation once motion exists.
#include <Arduino.h>
#include <string.h>
#include <stdlib.h>

#include "protocol.h"

#if defined(CLAWDE_ENABLE_SERVOS)
#error "Servo control is not implemented: board, pins, limits and calibration are pending."
#endif

namespace {

enum class State : uint8_t { DISARMED, ARMED, STOPPED };

char lineBuf[CLAWDE_MAX_LINE + 1];
uint8_t lineLen = 0;
bool discarding = false;  // current line overflowed: drop until newline
State state = State::DISARMED;
unsigned long lastHostMs = 0;
volatile bool stopRequested = false;  // checked by any future interpolation loop

const char* stateName() {
  switch (state) {
    case State::ARMED: return "ARMED";
    case State::STOPPED: return "STOPPED";
    default: return "DISARMED";
  }
}

void reply(const char* id, const char* kind, const char* a = nullptr, const char* b = nullptr) {
  Serial.print(id);
  Serial.print(' ');
  Serial.print(kind);
  if (a) { Serial.print(' '); Serial.print(a); }
  if (b) { Serial.print(' '); Serial.print(b); }
  Serial.print('\n');
}

bool validId(const char* token) {
  size_t n = strlen(token);
  if (n == 0 || n > 5) return false;
  for (size_t i = 0; i < n; ++i) {
    if (token[i] < '0' || token[i] > '9') return false;
  }
  long value = atol(token);
  return value >= 1 && value <= 65535;
}

void handleLine(char* line) {
  char* tokens[CLAWDE_MAX_TOKENS];
  uint8_t count = 0;
  for (char* tok = strtok(line, " "); tok && count < CLAWDE_MAX_TOKENS; tok = strtok(nullptr, " ")) {
    tokens[count++] = tok;
  }
  if (count < 2 || !validId(tokens[0])) {
    reply("0", "EVT", ERR_MALFORMED);
    return;
  }
  const char* id = tokens[0];
  const char* cmd = tokens[1];
  lastHostMs = millis();

  if (strcmp(cmd, "STOP") == 0) {
    stopRequested = true;
    state = State::STOPPED;
    reply(id, "ACK");
    reply(id, "DONE", "STOPPED");
  } else if (strcmp(cmd, "PING") == 0) {
    reply(id, "ACK");
    reply(id, "DONE", "PONG", CLAWDE_PROTOCOL_NAME);
  } else if (strcmp(cmd, "STATUS") == 0) {
    reply(id, "ACK");
    reply(id, "DONE", stateName(), "SERVOS=NONE");
  } else if (strcmp(cmd, "ARM") == 0) {
    stopRequested = false;
    state = State::ARMED;  // logical only: no servo is attached in this firmware
    reply(id, "ACK");
    reply(id, "DONE", "ARMED", "DIAGNOSTIC");
  } else if (strcmp(cmd, "DISARM") == 0) {
    state = State::DISARMED;
    reply(id, "ACK");
    reply(id, "DONE", "DISARMED");
  } else if (strcmp(cmd, "HOME") == 0 || strcmp(cmd, "GRIP") == 0 || strcmp(cmd, "MOVE") == 0) {
    reply(id, "ERR", ERR_NOT_CONFIGURED);
  } else {
    reply(id, "ERR", ERR_UNKNOWN);
  }
}

void pollSerial() {
  // Bounded work per loop: never block waiting for bytes.
  int budget = CLAWDE_MAX_LINE;
  while (Serial.available() > 0 && budget-- > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      if (discarding) {
        discarding = false;
        reply("0", "EVT", ERR_TOO_LONG);
      } else if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      continue;
    }
    if (discarding) continue;
    if (c < 0x20 || c > 0x7e || lineLen >= CLAWDE_MAX_LINE - 1) {
      discarding = true;  // non-ASCII or overlong: reject whole line
      lineLen = 0;
      continue;
    }
    lineBuf[lineLen++] = c;
  }
}

void checkWatchdog() {
  if (state == State::ARMED && millis() - lastHostMs > CLAWDE_WATCHDOG_MS) {
    state = State::DISARMED;  // invalidate arming; never starts a trajectory
    reply("0", "EVT", "WATCHDOG", "DISARMED");
  }
}

}  // namespace

void setup() {
  Serial.begin(CLAWDE_BAUDRATE);
  lastHostMs = millis();
  // No servo attach, no motion at boot.
  reply("0", "EVT", "BOOT", CLAWDE_PROTOCOL_NAME);
}

void loop() {
  pollSerial();
  checkWatchdog();
}
