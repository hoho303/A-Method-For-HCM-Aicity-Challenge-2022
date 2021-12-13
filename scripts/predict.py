import crop as cr
import detect as det
import reg as rec
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Detect And Recognition Image')
    parser.add_argument('--test-input', help='the dir contains test images')
    parser.add_argument('--det-config', help='detect config file path')
    parser.add_argument(
        '--det-output', help='the dir saves output of detection')
    parser.add_argument(
        '--det-model', help='detect model file path')
    parser.add_argument(
        '--threshold-det',type = float, help='setting threshold for detection')
    parser.add_argument(
        '--crop-path', help='the dir saves crop images after detection')
    parser.add_argument('--rec-config', help='recognition config file path')
    parser.add_argument('--rec-model', help='recognition model file path')
    parser.add_argument('--threshold-rec',type = float, help='setting threshold for recognition')
    parser.add_argument('--output', help='the dir saves output after detection and recognition')

    args = parser.parse_args()
    return args

def predict(rootImage,config_det,checkpoint_det,threshold_det,output_det_path,crop_path,config_rec,checkpoint_reg,threshold_rec,predicted_clean):
  
  # Detect Scene Text Images
  print("Starting Detect Images....")
  det.detect(rootImage,output_det_path,config_det,checkpoint_det,threshold_det)
  print("Done")

  # Crop Images From Files To Recognition
  print("Starting Crop Images....")
  cr.crop(rootImage,output_det_path, crop_path)
  print("Done")

  # Recognition
  print("Starting Recogniton Images....")
  rec.recognition(crop_path,checkpoint_reg,config_rec,output_det_path,predicted_clean,threshold_rec)
  print("Done")

  # Clean And Write Output
  print("Starting Clean Predict....")
  rec.clean(output_det_path,predicted_clean)
  print("Finished")

def main():
    args = parse_args()
    
    predict(args.test_input,args.det_config,
            args.det_model,args.threshold_det,args.det_output,
            args.crop_path,args.rec_config,args.rec_model,args.threshold_rec,args.output)

if __name__ == '__main__':
    main()

