#include <Arduino.h>
#include <driver/i2s.h>
#include <SPI.h>
#include <SD.h>

//Wiring for MicroSD Card Adapter
#define PIN_SD_MISO 19
#define PIN_SD_MOSI 23
#define PIN_SD_SCK  18
#define PIN_SD_CS   5

//definition for the LED, PUSH BUTTON AND MIC OUTPUT
#define PIN_LED     26
#define PIN_BUTTON  27   
#define PIN_MIC_ADC 36//output from max4466 to pin 36   

//Audio
#define SAMPLE_RATE     44100//standard sampling rate   
#define BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_16BIT//each sample uses 2 bytes
#define CHANNELS        1//mono audio

//State variables
File wavFile;//the current file on the SD card
bool recording = false;//true when recording is active, false when stopped
String currentName;//stores the file name, like /rec_0001.wav
uint32_t totalDataBytes = 0;//counts how many bytes of audio were written
uint32_t actualSampleRate = SAMPLE_RATE;//stores the real I2S clock rate used by ESP32

//I2S Config
i2s_config_t i2s_config = {
  .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
  .sample_rate = SAMPLE_RATE,
  .bits_per_sample = BITS_PER_SAMPLE,
  .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
  .communication_format = I2S_COMM_FORMAT_I2S_MSB,
  .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
  .dma_buf_count = 8,
  .dma_buf_len = 1024,
  .use_apll = false,
  .tx_desc_auto_clear = false,
  .fixed_mclk = 0
};

//WAV Helpers
//Writes a 16-bit value in little endian format
void write_u16_le(File &f, uint16_t v) {
  uint8_t b[2] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF) };
  f.write(b, 2);
}

//Writes a 32-bit value in little endian format.
void write_u32_le(File &f, uint32_t v) {
  uint8_t b[4] = {
    (uint8_t)(v & 0xFF),
    (uint8_t)((v >> 8) & 0xFF),
    (uint8_t)((v >> 16) & 0xFF),
    (uint8_t)((v >> 24) & 0xFF)
  };
  f.write(b, 4);
}

//Writes 4-character labels
void write_tag(File &f, const char tag[4]) {
  f.write((const uint8_t*)tag, 4);
}

//Writing the WAV header
void write_wav_header(File &f) {
  write_tag(f, "RIFF");//marks the file as a riff file
  write_u32_le(f, 0);//writes 0 for now and patches it later.
  write_tag(f, "WAVE");//marks it as a WAV audio file

  write_tag(f, "fmt ");//starts the format section
  write_u32_le(f, 16);//PCM format block size
  write_u16_le(f, 1);//stardard PCM, uncompressed audio
  write_u16_le(f, CHANNELS);//writes 1 for mono
  write_u32_le(f, actualSampleRate);//writes the real sample rate into the WAV header

  uint32_t byteRate   = actualSampleRate * CHANNELS * (BITS_PER_SAMPLE / 8);//bytes per second
  uint16_t blockAlign = CHANNELS * (BITS_PER_SAMPLE / 8);//bytes per sample frame

  write_u32_le(f, byteRate);//writes 16
  write_u16_le(f, blockAlign);
  write_u16_le(f, BITS_PER_SAMPLE);

  write_tag(f, "data");//starts the audio data section
  write_u32_le(f, 0); // file is unknown at start so patched later
}

//Patching the WAV header after recording
void patch_wav_header(File &f) {
  uint32_t fileSize = f.size();//gets full size in bytes
  uint32_t subchunk2Size = fileSize - 44;//subtracts 44 byte WAV header,ramins is pure audio data
  uint32_t chunkSize     = fileSize - 8;//WAV format expects chunk size to be total file size minus 8
  f.seek(4);//patch bytes 4 to 7 
  write_u32_le(f, chunkSize);
  f.seek(40);//patch bytes 40 to 43
  write_u32_le(f, subchunk2Size);
}

//File Naming
String next_wav_name() {
  for (int i = 1; i <= 9999; i++) {
    char path[20];
    snprintf(path, sizeof(path), "/rec_%04d.wav", i);
    if (!SD.exists(path)) return String(path);
  }
  return String("/rec_last.wav");
}

//Recording Start
void start_recording() {
  currentName = next_wav_name();
  totalDataBytes = 0;

  wavFile = SD.open(currentName.c_str(), FILE_WRITE);
  if (!wavFile) {
    Serial.println("SD open failed");
    return;
  }

  write_wav_header(wavFile);
  wavFile.flush();

  recording = true;
  digitalWrite(PIN_LED, HIGH);

  Serial.print("REC START: ");
  Serial.println(currentName);
}

//Recording STOP
void stop_recording() {
  recording = false;
  digitalWrite(PIN_LED, LOW);

  if (!wavFile) return;

  wavFile.flush();
  patch_wav_header(wavFile);
  wavFile.close();

  Serial.println("REC STOP");
  Serial.print("Data bytes written: ");
  Serial.println(totalDataBytes);
}

//DC Removal Filter
int32_t x_prev = 0;
int32_t y_prev = 0;
const float HPF_ALPHA = 0.998f;

int16_t dc_filter(int16_t raw) {
  int32_t x_curr = raw;
  int32_t y_curr = (x_curr - x_prev) + (HPF_ALPHA * y_prev);
  x_prev = x_curr;
  y_prev = y_curr;
  return (int16_t)y_curr;
}

//Setup
void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  SPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
  if (!SD.begin(PIN_SD_CS, SPI, 4000000)) {
    Serial.println("SD init failed!");
    while (true) delay(500);
  }

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_0);
  i2s_adc_enable(I2S_NUM_0);

  //Query actual rate correctly
  float clk = i2s_get_clk(I2S_NUM_0);
  actualSampleRate = (uint32_t)clk;
  Serial.printf("I2S actual sample rate: %u\n", actualSampleRate);

  Serial.println("SD OK, I2S OK");
  Serial.println("Press button to start/stop recording");
}

//Loop
void loop() {
  static int lastButton = HIGH;
  int btn = digitalRead(PIN_BUTTON);

  if (btn == LOW && lastButton == HIGH) {
    if (!recording) start_recording();
    else stop_recording();
    delay(200); // debounce
  }
  lastButton = btn;

  if (recording) {
    int16_t buffer[1024];
    size_t bytesRead;
    i2s_read(I2S_NUM_0, (void*)buffer, sizeof(buffer), &bytesRead, portMAX_DELAY);

    // Apply DC removal filter + software gain
    for (int i = 0; i < bytesRead / 2; i++) {
      int16_t filtered = dc_filter(buffer[i]);
      int32_t amplified = (int32_t)filtered * 4; // 4x gain
      if (amplified > 32767) amplified = 32767;
      if (amplified < -32768) amplified = -32768;
      buffer[i] = (int16_t)amplified;
    }

    if (wavFile) {
      wavFile.write((uint8_t*)buffer, bytesRead);
      totalDataBytes += bytesRead;
    }
  }
}
