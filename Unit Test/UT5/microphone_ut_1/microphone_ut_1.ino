/*
  ESP32 WAV Recorder (Analog mic on GPIO36 + SD card + toggle button + LED)

  Your wiring
  - LED     -> GPIO26 (through 330Ω to GND, or LED+res to 3V3 depending on your wiring)
  - Button  -> GPIO27 to GND (uses INPUT_PULLUP)
  - Mic OUT -> GPIO36 (ADC1_CH0). MAX4466 powered from 3V3 + GND
  - SD SPI  -> MOSI=23, MISO=19, SCK=18, CS=5 (SD module powered from 3V3)

  Behavior
  - Press button once  -> start recording (LED ON)
  - Press again        -> stop recording (LED OFF), finalize WAV, close file
*/

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include "driver/i2s.h"
#include "driver/adc.h"

// ---------------- Pins ----------------
static const int PIN_LED     = 26;
static const int PIN_BUTTON  = 27;   // to GND, uses INPUT_PULLUP

static const int PIN_SD_MOSI = 23;
static const int PIN_SD_MISO = 19;
static const int PIN_SD_SCK  = 18;
static const int PIN_SD_CS   = 5;

// Analog mic on GPIO36 => ADC1_CHANNEL_0
static const adc1_channel_t MIC_ADC_CH = ADC1_CHANNEL_0;

// ---------------- Audio settings ----------------
static const uint32_t SAMPLE_RATE_HZ   = 16000;  // stable for crowd audio + smaller files
static const uint16_t BITS_PER_SAMPLE  = 16;
static const uint16_t CHANNELS         = 1;

// I2S/ADC settings
static const i2s_port_t I2S_PORT = I2S_NUM_0;

// Buffer sizes
static const size_t I2S_SAMPLES_PER_READ = 1024; // samples
static const size_t I2S_BYTES_PER_READ   = I2S_SAMPLES_PER_READ * sizeof(uint16_t);

// ---------------- State ----------------
static bool recording = false;
static File wavFile;
static uint32_t dataBytesWritten = 0;
static char currentName[24] = {0};

// ---------------- Helpers: button edge + debounce ----------------
static bool button_pressed_edge() {
  // INPUT_PULLUP: released=HIGH, pressed=LOW
  static uint8_t lastStable = HIGH;
  static uint8_t lastRead   = HIGH;
  static uint32_t lastChangeMs = 0;

  const uint8_t now = digitalRead(PIN_BUTTON);

  if (now != lastRead) {
    lastRead = now;
    lastChangeMs = millis();
  }

  // debounce window
  if (millis() - lastChangeMs >= 30) {
    if (lastStable != lastRead) {
      lastStable = lastRead;
      if (lastStable == LOW) return true; // falling edge = press
    }
  }
  return false;
}

// ---------------- WAV header ----------------
static void write_wav_header_placeholder(File &f) {
  // 44-byte header, sizes patched on stop
  uint8_t hdr[44] = {0};

  auto w32 = [&](int idx, uint32_t v) {
    hdr[idx + 0] = (uint8_t)(v & 0xFF);
    hdr[idx + 1] = (uint8_t)((v >> 8) & 0xFF);
    hdr[idx + 2] = (uint8_t)((v >> 16) & 0xFF);
    hdr[idx + 3] = (uint8_t)((v >> 24) & 0xFF);
  };
  auto w16 = [&](int idx, uint16_t v) {
    hdr[idx + 0] = (uint8_t)(v & 0xFF);
    hdr[idx + 1] = (uint8_t)((v >> 8) & 0xFF);
  };

  // RIFF
  hdr[0] = 'R'; hdr[1] = 'I'; hdr[2] = 'F'; hdr[3] = 'F';
  w32(4, 36); // placeholder (36 + dataBytes)
  hdr[8] = 'W'; hdr[9] = 'A'; hdr[10] = 'V'; hdr[11] = 'E';

  // fmt chunk
  hdr[12] = 'f'; hdr[13] = 'm'; hdr[14] = 't'; hdr[15] = ' ';
  w32(16, 16);              // PCM fmt chunk size
  w16(20, 1);               // audio format 1 = PCM
  w16(22, CHANNELS);
  w32(24, SAMPLE_RATE_HZ);
  uint32_t byteRate = SAMPLE_RATE_HZ * CHANNELS * (BITS_PER_SAMPLE / 8);
  w32(28, byteRate);
  uint16_t blockAlign = CHANNELS * (BITS_PER_SAMPLE / 8);
  w16(32, blockAlign);
  w16(34, BITS_PER_SAMPLE);

  // data chunk
  hdr[36] = 'd'; hdr[37] = 'a'; hdr[38] = 't'; hdr[39] = 'a';
  w32(40, 0); // placeholder data size

  f.write(hdr, sizeof(hdr));
}

static void patch_wav_sizes(File &f, uint32_t dataBytes) {
  // file size at offset 4 = 36 + dataBytes
  uint32_t riffSize = 36 + dataBytes;
  f.seek(4);
  f.write((uint8_t*)&riffSize, 4);

  // data size at offset 40
  f.seek(40);
  f.write((uint8_t*)&dataBytes, 4);

  f.flush();
}

// ---------------- File naming ----------------
static bool next_filename(char *out, size_t outLen) {
  // Find next free /rec_0001.wav ... /rec_9999.wav
  for (int i = 1; i <= 9999; i++) {
    snprintf(out, outLen, "/rec_%04d.wav", i);
    if (!SD.exists(out)) return true;
  }
  return false;
}

// ---------------- I2S ADC init ----------------
static bool i2s_adc_init() {
  // Configure ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(MIC_ADC_CH, ADC_ATTEN_DB_11); // larger range
  // Optional: also set Arduino attenuation (helps consistency)
  analogSetPinAttenuation(36, ADC_11db);

  // I2S config for ADC
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
  cfg.sample_rate = SAMPLE_RATE_HZ;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_I2S_MSB;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 6;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;

  // Route ADC1 channel into I2S
  if (i2s_set_adc_mode(ADC_UNIT_1, MIC_ADC_CH) != ESP_OK) return false;

  // ADC from I2S needs this
  i2s_adc_enable(I2S_PORT);

  // Prime
  i2s_zero_dma_buffer(I2S_PORT);
  return true;
}

// Convert I2S ADC samples to signed 16-bit PCM centered around 0
static inline int16_t adc_to_pcm(uint16_t s) {
  // ESP32 ADC via I2S comes in 16-bit words where useful 12-bit is in the high bits.
  uint16_t adc12 = (s >> 4) & 0x0FFF;      // 0..4095
  int32_t centered = (int32_t)adc12 - 2048; // roughly center
  // scale to 16-bit
  int32_t pcm = centered << 4; // 12->16 bit
  if (pcm > 32767) pcm = 32767;
  if (pcm < -32768) pcm = -32768;
  return (int16_t)pcm;
}

// ---------------- Recording control ----------------
static void start_recording() {
  if (recording) return;

  if (!next_filename(currentName, sizeof(currentName))) {
    Serial.println("ERR: No free filename");
    return;
  }

  wavFile = SD.open(currentName, FILE_WRITE);
  if (!wavFile) {
    Serial.println("ERR: open file failed");
    return;
  }

  dataBytesWritten = 0;
  write_wav_header_placeholder(wavFile);

  digitalWrite(PIN_LED, HIGH);
  recording = true;

  Serial.print("REC START: ");
  Serial.println(currentName);
}

static void stop_recording() {
  if (!recording) return;

  recording = false;
  digitalWrite(PIN_LED, LOW);

  // finalize header
  patch_wav_sizes(wavFile, dataBytesWritten);
  wavFile.close();

  Serial.println("REC STOP");
  Serial.print("Data bytes written: ");
  Serial.println(dataBytesWritten);
  Serial.print("File size: ");
  Serial.println((uint32_t)(dataBytesWritten + 44));
}

static void stream_audio_to_sd() {
  static uint16_t i2sBuf[I2S_SAMPLES_PER_READ];
  static int16_t  pcmBuf[I2S_SAMPLES_PER_READ];

  size_t bytesRead = 0;
  esp_err_t ok = i2s_read(I2S_PORT, (void*)i2sBuf, I2S_BYTES_PER_READ, &bytesRead, portMAX_DELAY);
  if (ok != ESP_OK || bytesRead == 0) return;

  const size_t samples = bytesRead / sizeof(uint16_t);

  for (size_t i = 0; i < samples; i++) {
    pcmBuf[i] = adc_to_pcm(i2sBuf[i]);
  }

  if (wavFile) {
    size_t written = wavFile.write((uint8_t*)pcmBuf, samples * sizeof(int16_t));
    dataBytesWritten += written;
  }

  // stop on press while recording
  if (button_pressed_edge()) {
    Serial.println("STOP PRESS DETECTED");
    stop_recording();
  }
}

// ---------------- Setup / loop ----------------
void setup() {
  Serial.begin(115200);
  delay(600);

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  pinMode(PIN_BUTTON, INPUT_PULLUP);

  Serial.println();
  Serial.println("ESP32 SD WAV Recorder, press button to start, press again to stop");

  // SD init
  SPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);

  // try slower first, then faster (some modules hate high speed)
  if (!SD.begin(PIN_SD_CS, SPI, 1000000)) {
    Serial.println("ERR: SD.begin failed at 1MHz");
    if (!SD.begin(PIN_SD_CS, SPI, 4000000)) {
      Serial.println("ERR: SD.begin failed at 4MHz");
      Serial.println("Fix SD wiring, 3V3 power, FAT32 format");
      while (true) delay(500);
    }
  }

  Serial.println("SD OK");

  // I2S ADC init
  if (!i2s_adc_init()) {
    Serial.println("ERR: I2S/ADC init failed");
    while (true) delay(500);
  }

  Serial.println("Ready");
}

void loop() {
  if (!recording) {
    if (button_pressed_edge()) {
      start_recording();
    }
    delay(5);
    return;
  }

  stream_audio_to_sd(); // stops itself on button press
}
