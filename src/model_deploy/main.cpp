#include "mbed.h"
#include <cmath>
#include "DA7212.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "uLCD_4DGL.h"

#define bufferLength (32)
#define signalLength (1024)

DA7212 audio;
Serial pc(USBTX, USBRX);

int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];

int mode=0;              // control interrupt
int sub_mode=1;          // interfces totally 3
int main_page=0;
int trigger=1;           // control selection
int serialCount=0;
int note=0;
int gesture_index;
int first=0;             // initial
int last=0;              // previous mode 
int song=1;              // totally 3
int new_song=1;          // the next song 
int display=0;           // control display
int change=0;            // control change
float song_note[42];
float noteLength[42];
char type[4]={0x20, 0x31, 0x32, 0x33};

uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn button(SW2);
DigitalIn  button2(SW3);

EventQueue DNNqueue(32 * EVENTS_EVENT_SIZE);
EventQueue songqueue(32 * EVENTS_EVENT_SIZE);
Thread DNNthread(osPriorityNormal,80*1024);
Thread songthread(osPriorityNormal,80*1024);

void playNote(float freq[])              
{
  float frequency =  freq[note];
  for(int i = 0; (i < kAudioSampleFrequency / kAudioTxBufferSize)&& !mode; ++i)
  {
    for (int j = 0; j < kAudioTxBufferSize; j++)
    {
    waveform[j] = (int16_t) (sin((double)j * 2. * M_PI/(double) (kAudioSampleFrequency /( 500*frequency))) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
  if(note < 42){
  note += 1 ;
  }
  else{
  ;
  }
}

void loadSignal(void)
{
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  i = 0;
  serialCount =0;
  while(i < 42)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        song_note[i] = (float) atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  i =0;
  serialCount =0;
  while(i < 42)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        noteLength[i] = (float) atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
}

void ISR1()
{
  if(mode == 0)
    mode=1;       // song stop
  else 
    mode=0;       // song play
  first=1;
}

int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void DNN(){

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(),1);
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
  }
}

int main(int argc, char* argv[])
{
  songthread.start(callback(&songqueue, &EventQueue::dispatch_forever));
  DNNthread.start(DNN);
  button.rise(&ISR1); 
  audio.spk.pause();

  while(true){
    if(mode){
      audio.spk.pause();      // interrupt in
      if(trigger == 1)        // change mode with DNN
      {
        if(gesture_index == 0){
          last = 1;
          if(sub_mode < 2)
            sub_mode = 4;
          else
            sub_mode -= 1;
          change = 0;
        }
        if(gesture_index == 1){
          last = 1;
          if(sub_mode > 3)
            sub_mode = 1;
          else
            sub_mode += 1;
          change = 0;
        }
        if(first){             // interface at beginning
          if(sub_mode == 1){
            if(song > 2)
              new_song = song-1;
            else 
              new_song = 3;
            uLCD.cls();
            uLCD.printf("Backward To \n\n\n\n\n");
            uLCD.printf("%c",type[new_song]);
          }
          if(sub_mode == 2){
            if(song < 3)
              new_song = song+1;
            else 
              new_song = 1;
            uLCD.cls();
            uLCD.printf("Forward To \n\n\n\n\n");
            uLCD.printf("%c",type[new_song]);
          }
          if(sub_mode == 3){
            uLCD.cls();
            uLCD.printf("Change Songs\n\n\n\n\n");
            uLCD.printf("%c\n",type[1]);
            uLCD.printf("%c\n",type[2]);
            uLCD.printf("%c\n",type[3]);
          }  
        first = 0;  
        }
        if(sub_mode == 1){
          if(last){         // interface from previous to another
            if(sub_mode == 1){
              if(song > 1)
                new_song = song-1;
              else 
                new_song = 3;  
            }
            uLCD.cls();
            last = 0;
            uLCD.printf("Backward To\n\n\n\n\n");
            uLCD.printf("%c",type[new_song]); 
            main_page = 0;
          }
        }
        if(sub_mode == 2){
          if(last){
            if(sub_mode == 2){
              if(song < 3)
                new_song = song+1;
              else 
                new_song = 1;  
            }
            uLCD.cls();
            last = 0;
            uLCD.printf("Forward To\n\n\n\n\n");
            uLCD.printf("%c",type[new_song]);
            main_page = 0;
          }
        }
        if(sub_mode == 3){
          if(last){
            uLCD.cls();
            last = 0;
            uLCD.printf("Change Songs\n\n\n\n\n");
            main_page=0;
          }
        if(button2 == 0){
            trigger = 2;    // enter song selection 
        }
      }
    }
    if(trigger == 2){       // change with DNN
        if(gesture_index == 0){
          display = 0;
          if(song > 2)
            song -= 1;
          else
          {
            song = 3;
          }
        }
        if(gesture_index == 1){
          display = 0;
          if(song < 2)
            song += 1;
          else
          {
            song = 1;
          }
        }
        new_song = song;
        if(display == 0){    // display the songs
          uLCD.cls();
          display = 1;
          uLCD.printf("Change Songs\n\n\n\n\n");
          if(song == 1){
            uLCD.printf("Song1\n");
          }
          if(song == 2){
            uLCD.printf("Song2\n");
          }
          if(song == 3){          
            uLCD.printf("Song3\n");   
          }
        }
    }
  }
    else{
      if(main_page == 0){       // main page can display chosen songs
        song = new_song;
        uLCD.cls();
        if(sub_mode != 4){
          uLCD.printf("MP3 Player\n\n\n\n\nNow:%c\n",type[song]);
        }
        else{
        ;  
        }
        main_page = 1;
      }  
      if(change == 0){
        note =0;
        if(song == 1)
          pc.printf("%d\r\n",1);
        if(song == 2)
          pc.printf("%d\r\n",2);
        if(song == 3)
          pc.printf("%d\r\n",3);
        loadSignal();
        change = 1;
        pc.printf("%d\r\n",0);
      }
      trigger = 1;
      while(change&&!mode){
        songqueue.call(playNote,song_note);
        wait(4*noteLength[note]);
      } 
    }
  }
}