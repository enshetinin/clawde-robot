// CLAWDE serial protocol v1 (see docs/serial-protocol.md).
// Host -> device: "<id> <CMD>[ <arg>...]\n"   id = 1..65535
// Device -> host: "<id> ACK" | "<id> DONE[ ...]" | "<id> ERR <CODE>[ ...]" | "0 EVT <NAME>[ ...]"
#pragma once

#define CLAWDE_PROTOCOL_NAME "CLAWDE/1"
#define CLAWDE_MAX_LINE 64          // bytes, including the newline
#define CLAWDE_MAX_TOKENS 10
#define CLAWDE_BAUDRATE 115200UL
#define CLAWDE_WATCHDOG_MS 2000UL   // armed state is dropped without host traffic

// Error codes
#define ERR_MALFORMED "MALFORMED"
#define ERR_TOO_LONG "TOO_LONG"
#define ERR_UNKNOWN "UNKNOWN_CMD"
#define ERR_NOT_CONFIGURED "NOT_CONFIGURED"
#define ERR_NOT_ARMED "NOT_ARMED"
