import mediapipe as mp
from google.colab.patches import cv2_imshow
import tensorflow as tf
import random
import math
import keras
from keras import backend as K

mp_pose = mp.solutions.pose

# all of the landmarks that will be input into the identifier model
pose_landmark = mp_pose.PoseLandmark
POSE_LANDMARK_FILTER = [
  pose_landmark.LEFT_SHOULDER,
  pose_landmark.RIGHT_SHOULDER,
  pose_landmark.LEFT_ELBOW,
  pose_landmark.RIGHT_ELBOW,
  pose_landmark.LEFT_WRIST,
  pose_landmark.RIGHT_WRIST,
  pose_landmark.LEFT_PINKY,
  pose_landmark.RIGHT_PINKY,
  pose_landmark.LEFT_INDEX,
  pose_landmark.RIGHT_INDEX,
  pose_landmark.LEFT_THUMB,
  pose_landmark.RIGHT_THUMB,
  pose_landmark.LEFT_HIP,
  pose_landmark.RIGHT_HIP,
  pose_landmark.LEFT_KNEE,
  pose_landmark.RIGHT_KNEE,
  pose_landmark.LEFT_ANKLE,
  pose_landmark.RIGHT_ANKLE,
  pose_landmark.LEFT_HEEL,
  pose_landmark.RIGHT_HEEL,
  pose_landmark.LEFT_FOOT_INDEX,
  pose_landmark.RIGHT_FOOT_INDEX,
]

def filter_landmarks(landmarks):
  """ Removes the unused landmarks and coordinate
      data for each landmark. z coordinate is not
      used since the BlazePose model is not yet
      trained to accuracy identify z coordinates.
      Args:
        landmarks - numpy array with shape (33, 4)
      Returns:
        filtered numpy array with shape (22, 2)
  """

  return landmarks[POSE_LANDMARK_FILTER, :-2]

def normalize_landmarks(landmarks):
  """ Translate landmarks to the origin
      then scale them to fit the unit circle
      instead of normalization to the image width/height
      Args:
        landmarks is a numpy array with shape (22, 2)
        where 22 is the number of landmarks in the sample
        and 2 is the rank of the [x, y] data for each landmark
      Returns:
        normalized (22, 2) tensor
  """
  centroid = np.mean(landmarks, axis=0)
  landmarks -= centroid
  # have a small value in the case of landmarks being a zero matrix
  furthest_distance = max(np.max(np.sqrt(np.sum(abs(landmarks)**2,axis=-1))), 1e-8)
  landmarks /= furthest_distance

  return landmarks

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))