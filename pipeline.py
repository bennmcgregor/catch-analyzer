import cv2
import numpy as np

IDENTIFIER_THRESHOLD = 0.5
EVALUATOR_THRESHOLD = 0.5

def get_catch_evaluation(frame, image_preprocessor, evaluator_model):
  """ Runs the catch evaluator model on the image data
      Args:
        frame - a (180, 320, 3) numpy array representing the frame
        image_preprocessor - the image preprocessing closure that normalizes the image
        evaluator_model - the catch evaluator model instance
      Returns:
        the int label representation assigned to the frame
  """
  # convert image to grayscale (model only looking for shapes, not colours!)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame = np.expand_dims(frame, axis=(0,3))

  prediction = evaluator_model.predict(image_preprocessor(frame))

  return prediction[0][0]

def get_is_catch(processed_landmarks, identifier_model):
  """ Runs the catch identifier model on the pose data
      Args:
        processed_landmarks - (22, 2) numpy array
      Returns:
        is_catch - boolean value
  """
  prediction = identifier_model.predict(processed_landmarks)

  return prediction[0][0] > IDENTIFIER_THRESHOLD

def get_processed_landmarks(landmarks):
  """ Applies filter to landmarks and normalizes the data
      Args:
        landmarks - (33, 4) numpy array
      Returns:
        processed_landmarks - (None, 22, 2) normalized numpy array
  """
  processed_landmarks = filter_landmarks(landmarks)
  processed_landmarks = normalize_landmarks(processed_landmarks)
  processed_landmarks = np.expand_dims(processed_landmarks, axis=0)

  return processed_landmarks

def get_pose_data(frame, pose_model):
  """ Runs BlazePose model inference on the given frame
      Args:
        frame - a numpy array of image data of shape (180, 320, 3)
        pose_model - BlazePose model instance
      Returns:
        landmarks - a numpy array of shape (33, 4) of 
          landmark data for each point in the pose
  """
  # preprocess frame data
  frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
  # pass frame by reference
  frame.flags.writeable = False

  results = pose_model.process(frame)

  # write an array of shape (33, 4) where dim-2 is [x, y, z, visibility]
  landmarks = []
  if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
      landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
  else:
    landmarks = np.zeros((33,4))
  
  return np.array(landmarks)

def preprocess_frame(raw_frame):
  """ Preprocesses the frame data into the size and bitrate
      required for the model. It's best if raw_frame has a
      16:9 aspect ratio because this matches the data that 
      the model was trained on.
      Args:
        raw_frame - a numpy array representing the frame
      Returns:
        frame - a numpy array representing the frame resized
          to (h, w, channels) = (180, 320, 3)
  """
  interpolation = cv2.INTER_AREA
  if raw_frame.shape[1] < 320:
    interpolation = cv2.INTER_LINEAR
  frame = cv2.resize(raw_frame, dsize=(320, 180), interpolation=interpolation)

  return frame

def infer(video, pose_model, identifier_model, evaluator_model, show_frames=True, show_predictions=False):
  """ Runs inference on the video, frame-by-frame
      Args:
        video - path to the video file
        pose_model - BlazePose model instance
        identifier_model - identifier model instance
        evaluator_model - evaluator model instance
        show_frames - whether to display the video frame as it's analyzed
        show_predictions - whether to display the sigmoid output value on catch evaluations
      Returns:
        None. Prints out labels as the video file is read.
  """
  cap = cv2.VideoCapture(video)

  image_preprocessor = get_image_preprocessor()

  while cap.isOpened():
    success, raw_frame = cap.read()
    if not success:
      break
    
    frame = preprocess_frame(raw_frame)

    if show_frames:
      cv2_imshow(frame)

    landmarks = get_pose_data(frame, pose_model)
    processed_landmarks = get_processed_landmarks(landmarks)

    is_catch = get_is_catch(processed_landmarks, identifier_model)

    if is_catch: # conditionally continue with processing
      catch_evaluation = get_catch_evaluation(frame, image_preprocessor, evaluator_model)
      if show_predictions:
        print("Sigmoid output: " + str(catch_evaluation))
      if catch_evaluation > EVALUATOR_THRESHOLD:
        print("Good")
      else:
        print("Bad")
    else:
      print("Not Catch")

  cap.release()
