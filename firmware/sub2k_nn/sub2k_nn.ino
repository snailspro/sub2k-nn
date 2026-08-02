#include <avr/pgmspace.h>
#include "nn_table.h"

#define SYNC_BYTE 0xAA
#define SERIAL_TIMEOUT_MS 2000

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  Serial.println(F("sub2k-nn (Ensemble) pronto. Envie [0xAA][64 pixels][checksum]."));
  Serial.print(F("Tabela: "));
  Serial.print(NUM_MODELS);
  Serial.print(F(" Redes de "));
  Serial.print(HIDDEN_DIM);
  Serial.println(F(" ocultos."));
}

void infer_single_network(const uint8_t* raw_pixels, int32_t* outScores, const uint8_t W1[][HIDDEN_DIM/2], const int32_t* B1, const uint8_t W2[][OUTPUT_DIM/2], const int32_t* B2) {
  int32_t hidden[HIDDEN_DIM];

  for (uint8_t j = 0; j < HIDDEN_DIM; j += 2) {
    int32_t acc0 = (int32_t)pgm_read_dword(&B1[j]);
    int32_t acc1 = (int32_t)pgm_read_dword(&B1[j+1]);
    for (uint8_t i = 0; i < INPUT_DIM; i++) {
      uint8_t real_idx = pgm_read_byte(&active_pixels[i]);
      uint8_t packed = pgm_read_byte(&W1[i][j/2]);
      int8_t w0 = (int8_t)(packed & 0xF0) >> 4;
      int8_t w1 = (int8_t)(packed << 4) >> 4;
      acc0 += (int32_t)raw_pixels[real_idx] * w0;
      acc1 += (int32_t)raw_pixels[real_idx] * w1;
    }
    hidden[j] = acc0 > 0 ? acc0 : 0; 
    hidden[j+1] = acc1 > 0 ? acc1 : 0; 
  }

  for (uint8_t k = 0; k < OUTPUT_DIM; k += 2) {
    int32_t acc0 = (int32_t)pgm_read_dword(&B2[k]);
    int32_t acc1 = (int32_t)pgm_read_dword(&B2[k+1]);
    for (uint8_t j = 0; j < HIDDEN_DIM; j++) {
      uint8_t packed = pgm_read_byte(&W2[j][k/2]);
      int8_t w0 = (int8_t)(packed & 0xF0) >> 4;
      int8_t w1 = (int8_t)(packed << 4) >> 4;
      acc0 += hidden[j] * w0;
      acc1 += hidden[j] * w1;
    }
    outScores[k] = acc0;
    outScores[k+1] = acc1;
  }
}

int8_t infer_ensemble(const uint8_t* raw_pixels, int32_t* outScoresTotal) {
  for(uint8_t k=0; k<OUTPUT_DIM; k++) outScoresTotal[k] = 0;
  
  for(uint8_t m=0; m<NUM_MODELS; m++) {
      int32_t scores_tmp[OUTPUT_DIM];
      
      const uint8_t (*w1_ptr)[HIDDEN_DIM/2] = (const uint8_t (*)[HIDDEN_DIM/2])pgm_read_word(&W1_PTRS[m]);
      const int32_t* b1_ptr = (const int32_t*)pgm_read_word(&B1_PTRS[m]);
      const uint8_t (*w2_ptr)[OUTPUT_DIM/2] = (const uint8_t (*)[OUTPUT_DIM/2])pgm_read_word(&W2_PTRS[m]);
      const int32_t* b2_ptr = (const int32_t*)pgm_read_word(&B2_PTRS[m]);
      
      infer_single_network(raw_pixels, scores_tmp, w1_ptr, b1_ptr, w2_ptr, b2_ptr);
      
      for(uint8_t k=0; k<OUTPUT_DIM; k++) {
          outScoresTotal[k] += scores_tmp[k];
      }
  }
  
  int8_t bestClass = 0;
  int32_t bestScore = -2147483647L;
  for(uint8_t k=0; k<OUTPUT_DIM; k++) {
      if (outScoresTotal[k] > bestScore) {
          bestScore = outScoresTotal[k];
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
  int8_t pred = infer_ensemble(pixels, scores);

  Serial.print(F("PRED="));
  Serial.print(pred);
  Serial.print(F(" SCORES="));
  for (uint8_t k = 0; k < OUTPUT_DIM; k++) {
    Serial.print(scores[k]);
    if (k < OUTPUT_DIM - 1) Serial.print(' ');
  }
  Serial.println();
}
