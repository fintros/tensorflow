/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h"

#include <device.h>
#include <drivers/sensor.h>
#include <stdio.h>
#include <string.h>
#include <zephyr.h>

#define BUFLEN 300
int begin_index = 0;
struct device* sensor = NULL;
int current_index = 0;

float bufx[BUFLEN] = {0.0f};
float bufy[BUFLEN] = {0.0f};
float bufz[BUFLEN] = {0.0f};

bool initial = true;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  sensor = device_get_binding("ADXL362");
  if (sensor == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to get accelerometer, label: %s\n",
                         "ADXL362");
  } else {
    TF_LITE_REPORT_ERROR(error_reporter, "Got accelerometer, label: %s\n",
                         "ADXL362");
  }
  return kTfLiteOk;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length) {
  int rc;
  struct sensor_value accel;
  int samples_count;

  static int entrance_count = 0;
  printf("Read accelerometer data cycle %d\n", entrance_count++);

  samples_count = length;
  for (int i = 0; i < samples_count; i++) {
    rc = sensor_sample_fetch(sensor);
    if (rc < 0) {
      TF_LITE_REPORT_ERROR(error_reporter, "Fetch failed\n");
      return false;
    }
    rc = sensor_channel_get(sensor, SENSOR_CHAN_ACCEL_X, &accel);
    if (rc < 0) {
      TF_LITE_REPORT_ERROR(error_reporter, "ERROR: Update failed: %d\n", rc);
      return false;
    }
    input[i++] = (float)sensor_value_to_double(&accel) * 100;
    rc = sensor_channel_get(sensor, SENSOR_CHAN_ACCEL_Y, &accel);
    if (rc < 0) {
      TF_LITE_REPORT_ERROR(error_reporter, "ERROR: Update failed: %d\n", rc);
      return false;
    }
    input[i++] = (float)sensor_value_to_double(&accel) * 100;
    rc = sensor_channel_get(sensor, SENSOR_CHAN_ACCEL_Z, &accel);
    if (rc < 0) {
      TF_LITE_REPORT_ERROR(error_reporter, "ERROR: Update failed: %d\n", rc);
      return false;
    }
    input[i++] = (float)sensor_value_to_double(&accel) * 100;
    printf("X: %d, Y: %d, Z: %d\n", (int)(input[i-3]*100), (int)(input[i-2]*100), (int)(input[i-1]*100));
    
    k_usleep(40000); // 25 Hz
  }

  return true;
}
