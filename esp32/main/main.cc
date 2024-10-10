/*
 * SPDX-FileCopyrightText: 2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include "sdkconfig.h"
#include "bsp/esp-bsp.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "esp_task_wdt.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "data.h"
namespace {
const tflite::Model* model = nullptr;


//finalmodel
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int scratchBufSize = 39 * 1024;
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 121 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external

}

// Defined in model_settings.h
constexpr int kNumCols = 32;
constexpr int kNumRows = 32;
constexpr int kNumChannels = 1;
constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;
constexpr int kCategoryCount = 10;
const char* kCategoryLabels[kCategoryCount] = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
};


unsigned long modelStartTime, modelEndTime;


static const char *TAG = "example";

const float MEAN = 0.1307f;
const float STD = 0.3081f;


void init_tflite_model(){

    //------------------------------- model init -----------------------------//

    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    if (tensor_arena == NULL) {
        size_t before_inference = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        // tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        size_t after_inference = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

        size_t inference_memory_usage = before_inference - after_inference;
        MicroPrintf("Dynamic ram usage of inference model: %d bytes\n", inference_memory_usage);

    }
    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }

    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    // micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddPad();
    // micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddMaxPool2D();



    // Build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Get information about the memory area to use for the model's input.
    input = interpreter->input(0);
    output = interpreter->output(0);
    
}

void run_inference(void){

  esp_task_wdt_reset();
  vTaskDelay(2);
  
  // normalize_image(image_data, f_mean, f_std_dev);
  
  // Run the model on this input and make sure it succeeds.
  modelStartTime = esp_timer_get_time();
  if (kTfLiteOk != interpreter->Invoke()) {
      MicroPrintf("Invoke failed.");
  }
  modelEndTime = esp_timer_get_time();
  MicroPrintf("Runtime of finalmodel: %d microseconds", modelEndTime - modelStartTime);
  TfLiteTensor* output = interpreter->output(0);
  // Output finalmodel results
  float max_value = output->data.f[0];
  int max_index = 0;
  for (size_t i = 1; i < kCategoryCount; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }
  MicroPrintf("The predicted digit is: %d, with confidence: %f", max_index, max_value); 

}

void setup(){
    esp_timer_init();
    init_tflite_model();
}

void get_test_data(float* image_data){
//todo: normalize the data
  for (int i = 0; i < kNumCols; i++) {
    for (int j = 0; j < kNumRows; j++) {
      image_data[i * kNumCols + j] = input_data[i * kNumCols + j];
    }
  }
} 

void loop(){
    get_test_data(input->data.f);
    run_inference(); 
    // vTaskDelay(5); // to avoid watchdog trigger
    vTaskDelay(1);
}


void tf_main(void) {
  setup();
  while (true) {
    loop();
    esp_task_wdt_reset();
    vTaskDelay(1);
  }
}

extern "C" void app_main() {
  xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}