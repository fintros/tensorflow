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

#include "tensorflow/lite/micro/examples/magic_wand/main_functions.h"

#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h"
#include "tensorflow/lite/micro/examples/magic_wand/constants.h"
#include "tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h"
#include "tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h"
#include "tensorflow/lite/micro/examples/magic_wand/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 128) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);

//  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
//  if (setup_status != kTfLiteOk) {
//    TF_LITE_REPORT_ERROR(error_reporter, "Set up failed\n");
//  }
}

const float g_slope_micro_f2e59fea_nohash_1_data[] = {
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    -766.0, 132.0,  709.0,  -751.0, 249.0,  659.0,
    -714.0, 314.0,  630.0,  -709.0, 244.0,  623.0,  -707.0, 230.0,  659.0,
    -704.0, 202.0,  748.0,  -714.0, 219.0,  728.0,  -722.0, 239.0,  710.0,
    -744.0, 116.0,  612.0,  -753.0, -49.0,  570.0,  -748.0, -279.0, 527.0,
    -668.0, -664.0, 592.0,  -601.0, -635.0, 609.0,  -509.0, -559.0, 606.0,
    -286.0, -162.0, 536.0,  -255.0, -144.0, 495.0,  -209.0, -85.0,  495.0,
    6.0,    416.0,  698.0,  -33.0,  304.0,  1117.0, -82.0,  405.0,  1480.0,
    -198.0, 1008.0, 1908.0, -229.0, 990.0,  1743.0, -234.0, 934.0,  1453.0,
    -126.0, 838.0,  896.0,  -78.0,  792.0,  911.0,  -27.0,  741.0,  918.0,
    114.0,  734.0,  960.0,  135.0,  613.0,  959.0,  152.0,  426.0,  1015.0,
    106.0,  -116.0, 1110.0, 63.0,   -314.0, 1129.0, -12.0,  -486.0, 1179.0,
    -118.0, -656.0, 1510.0, -116.0, -558.0, 1553.0, -126.0, -361.0, 1367.0,
    -222.0, -76.0,  922.0,  -210.0, -26.0,  971.0,  -194.0, 50.0,   1053.0,
    -178.0, 72.0,   1082.0, -169.0, 100.0,  1073.0, -162.0, 133.0,  1050.0,
    -156.0, 226.0,  976.0,  -154.0, 323.0,  886.0,  -130.0, 240.0,  1154.0,
    -116.0, 124.0,  916.0,  -132.0, 124.0,  937.0,  -153.0, 115.0,  981.0,
    -184.0, 94.0,   962.0,  -177.0, 85.0,   1017.0, -173.0, 92.0,   1027.0,
    -168.0, 158.0,  1110.0, -181.0, 101.0,  1030.0, -180.0, 139.0,  1054.0,
    -152.0, 10.0,   1044.0, -169.0, 74.0,   1007.0,
};


const float g_ring_micro_f9643d42_nohash_4_data[] = {
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    0.0,    0.0,    0.0,    -665.0, 228.0,  827.0,  -680.0, 339.0,  716.0,
    -680.0, 564.0,  812.0,  -679.0, 552.0,  818.0,  -665.0, 528.0,  751.0,
    -658.0, 432.0,  618.0,  -655.0, 445.0,  592.0,  -667.0, 484.0,  556.0,
    -684.0, 590.0,  510.0,  -674.0, 672.0,  475.0,  -660.0, 786.0,  390.0,
    -562.0, 1124.0, 128.0,  -526.0, 1140.0, 111.0,  -486.0, 1044.0, 33.0,
    -416.0, 652.0,  -134.0, -390.0, 534.0,  -143.0, -365.0, 381.0,  -117.0,
    -314.0, 60.0,   94.0,   -322.0, 7.0,    190.0,  -338.0, -95.0,  342.0,
    -360.0, -106.0, 842.0,  -351.0, -41.0,  965.0,  -352.0, 12.0,   960.0,
    -366.0, 42.0,   1124.0, -322.0, 56.0,   1178.0, -312.0, 15.0,   1338.0,
    -254.0, 10.0,   1532.0, -241.0, 5.0,    1590.0, -227.0, 60.0,   1565.0,
    -204.0, 282.0,  1560.0, -180.0, 262.0,  1524.0, -138.0, 385.0,  1522.0,
    -84.0,  596.0,  1626.0, -55.0,  639.0,  1604.0, -19.0,  771.0,  1511.0,
    16.0,   932.0,  1132.0, 15.0,   924.0,  1013.0, 1.0,    849.0,  812.0,
    -88.0,  628.0,  500.0,  -114.0, 609.0,  463.0,  -155.0, 559.0,  382.0,
    -234.0, 420.0,  278.0,  -254.0, 390.0,  272.0,  -327.0, 200.0,  336.0,
    -558.0, -556.0, 630.0,  -640.0, -607.0, 740.0,  -706.0, -430.0, 868.0,
    -778.0, 42.0,   1042.0, -763.0, 84.0,   973.0,  -735.0, 185.0,  931.0,
    -682.0, 252.0,  766.0,  -673.0, 230.0,  757.0,  -671.0, 218.0,  757.0,
    -656.0, 222.0,  714.0,  -659.0, 238.0,  746.0,  -640.0, 276.0,  731.0,
    -634.0, 214.0,  754.0,  -637.0, 207.0,  735.0,  -637.0, 194.0,  742.0,
    -634.0, 248.0,  716.0,  -631.0, 265.0,  697.0,  -628.0, 252.0,  797.0,
    -592.0, 204.0,  816.0,  -618.0, 218.0,  812.0,  -633.0, 231.0,  828.0,
    -640.0, 222.0,  736.0,  -634.0, 221.0,  787.0,
};

void loop() {
  // Attempt to read new data from the accelerometer.
//  bool got_data =
//      ReadAccelerometer(error_reporter, model_input->data.f, input_length);
  // If there was no new data, wait until next time.
//  if (!got_data) return;

  printf("Use test data\n");
  memcpy(model_input->data.f, &g_ring_micro_f9643d42_nohash_4_data[0], input_length*sizeof(float));
//  memcpy(model_input->data.f, &g_slope_micro_f2e59fea_nohash_1_data[0], input_length*sizeof(float));

  // Run inference, and report any error.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n",
                         begin_index);
    return;
  }
  // Analyze the results to obtain a prediction
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

  // Produce an output
  HandleOutput(error_reporter, gesture_index);
}
