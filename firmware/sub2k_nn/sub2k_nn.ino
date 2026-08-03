#include <avr/pgmspace.h>
#include "nn_table.h"

#define SYNC_BYTE 0xAA
#define SERIAL_TIMEOUT_MS 2000

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  Serial.println(F("sub2k-nn (INT8 Nativo) pronto. Envie [0xAA][64 pixels][checksum]."));
  Serial.print(F("Entrada: ")); Serial.print(INPUT_DIM);
  Serial.print(F(" Ocultos: ")); Serial.println(HIDDEN_DIM);
}

int8_t infer(const uint8_t* raw_pixels, int32_t* outScores) {
  int32_t hidden[HIDDEN_DIM];

  for (uint8_t j = 0; j < HIDDEN_DIM; j++) {
    int32_t acc = (int32_t)pgm_read_dword(&B1[j]);
    for (uint8_t i = 0; i < INPUT_DIM; i++) {
      uint8_t real_idx = pgm_read_byte(&active_pixels[i]);
      acc += (int32_t)raw_pixels[real_idx] * (int8_t)pgm_read_byte(&W1[i][j]);
    }
    hidden[j] = acc > 0 ? acc : 0; 
  }

  int32_t bestScore = -2147483647L;
  int8_t bestClass = 0;

  for (uint8_t k = 0; k < OUTPUT_DIM; k++) {
    int32_t acc = (int32_t)pgm_read_dword(&B2[k]);
    for (uint8_t j = 0; j < HIDDEN_DIM; j++) {
      acc += hidden[j] * (int8_t)pgm_read_byte(&W2[j][k]);
    }
    outScores[k] = acc;
    if (acc > bestScore) {
      bestScore = acc;
      bestClass = k;
    }
  }

  return bestClass;
}

void loop() {
  if (Serial.available() <= 0) return;
  if (Serial.peek() != SYNC_BYTE) {
    Serial.read(); 
    return;
  }
  Serial.read(); 

  uint8_t pixels[64];
  size_t got = Serial.readBytes((char*)pixels, 64);
  if (got != 64) {
    Serial.println(F("ERR"));
    return;
  }

  uint8_t checksum;
  if (Serial.readBytes((char*)&checksum, 1) != 1) {
    Serial.println(F("ERR"));
    return;
  }

  uint16_t calc = 0;
  for (uint8_t i = 0; i < 64; i++) calc += pixels[i];
  if ((uint8_t)(calc & 0xFF) != checksum) {
    Serial.println(F("ERR"));
    return;
  }

  int32_t scores[OUTPUT_DIM];
  
  // Para medição de CPU isolada (como você faz):
  unsigned long t0 = micros();
  int8_t pred = infer(pixels, scores);
  unsigned long t1 = micros();

  Serial.print(F("PRED="));
  Serial.print(pred);
  Serial.print(F(" CPU_MICROS="));
  Serial.print(t1 - t0);
  Serial.print(F(" SCORES="));
  for (uint8_t k = 0; k < OUTPUT_DIM; k++) {
    Serial.print(scores[k]);
    if (k < OUTPUT_DIM - 1) Serial.print(' ');
  }
  Serial.println();
}
