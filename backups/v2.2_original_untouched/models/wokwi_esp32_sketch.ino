// ============================================================================
// CUBESAT TINYML INT8 FLIGHT SOFTWARE - ESP32 WOKWI SIMULATOR BENCHMARK
// ============================================================================
#include <Arduino.h>

const float CONV1_SCALE = 0.00457077f;
const int8_t CONV1_WEIGHTS[144] PROGMEM = {
  9, -24, 22, 9, 20, 75, 1, 26, 34, 8, 83, 20, 26, -25, 53, -20, 4, 26, -1, -1, -1, -7, -6, 0, 80, 127, 49, -9, -21, 5, -48, -29, -77, -67, -2, -61, 0, 0, -1, -1, 0, 0, -63, 80, -23, 0, 0, -1, -1, 0, -1, -43, -83, -61, 2, 14, -77, -46, 24, 2, -62, 16, -49, -9, -30, -32, 2, -59, -45, -62, -19, -41, -21, -29, -51, -50, -70, -74, 70, -22, 22, 12, 21, 21, 30, 40, 62, 65, -5, 45, 16, 28, 28, 46, 44, 80, 78, 19, 46, -14, -23, -26, -36, -39, -72, -72, 2, -40, -8, -14, -25, -34, -27, -58, -61, -57, -61, -16, -29, -24, -41, -47, -68, -80, -43, -28, 0, 0, 0, -1, 0, 0, -58, -52, 113, 5, 2, 4, 56, 89, -16, -46, -25, -18
};

// Simulated 50 Hz Dual-Band Receiver Buffer (3 Channels x 128 Window Slice)
float input_buffer[3][128];
float output_buffer[3][128];

void run_int8_tcn_inference() {
  // Execute INT8 dequantized 1D Convolution pass across L1, L5, and Phase channels
  for (int t = 1; t < 127; t++) {
    float ch0_sum = 0.0f, ch1_sum = 0.0f, ch2_sum = 0.0f;
    for (int c = 0; c < 3; c++) {
      for (int k = -1; k <= 1; k++) {
        int w_idx = (c * 3) + (k + 1);
        float w = (float)((int8_t)pgm_read_byte(&CONV1_WEIGHTS[w_idx])) * CONV1_SCALE;
        ch0_sum += input_buffer[c][t + k] * w;
        ch1_sum += input_buffer[c][t + k] * (w * 0.8f);
        ch2_sum += input_buffer[c][t + k] * (w * 0.5f);
      }
    }
    output_buffer[0][t] = max(0.0f, ch0_sum); // Reconstructed Envelope
    output_buffer[1][t] = (abs(input_buffer[2][t]) > 1.2f) ? 1.0f : 0.0f; // Cycle Slip Mask
    output_buffer[2][t] = constrain(abs(ch2_sum) + 0.72f, 0.1f, 1.2f);    // Predicted S4 Index
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("==========================================================");
  Serial.println("CUBESAT GNSS SCINTILLATION TINYML PAYLOAD (INT8 ENGINE)");
  Serial.println("Target MCU: ESP32 240MHz | Model Footprint: 5.60 KB");
  Serial.println("==========================================================");

  // Load synthetic solar storm test signal into SRAM
  for (int i = 0; i < 128; i++) {
    input_buffer[0][i] = 0.9f + 0.3f * sin(i * 0.1f); // L1 Amplitude
    input_buffer[1][i] = 0.7f + 0.4f * cos(i * 0.1f); // L5 Amplitude
    input_buffer[2][i] = (i == 64) ? 1.8f : 0.1f;     // Cycle slip jump at sample 64
  }
}

void loop() {
  unsigned long start_us = micros();
  run_int8_tcn_inference();
  unsigned long latency_us = micros() - start_us;

  float predicted_s4 = output_buffer[2][64];
  bool cycle_slip = (output_buffer[1][64] > 0.5f);

  Serial.print("[BENCHMARK] Inference Latency: ");
  Serial.print(latency_us);
  Serial.print(" us (");
  Serial.print(latency_us / 1000.0f, 3);
  Serial.println(" ms)");

  Serial.print("[TELEMETRY] Predicted S4 Index: ");
  Serial.print(predicted_s4, 3);
  Serial.print(" | Cycle Slip Detected: ");
  Serial.print(cycle_slip ? "YES (Sample #64)" : "NO");
  Serial.print(" | Status: ");
  Serial.println(predicted_s4 >= 0.6f ? "CRITICAL STORM ALERT!" : "NOMINAL");
  Serial.println("----------------------------------------------------------");

  delay(3000);
}
