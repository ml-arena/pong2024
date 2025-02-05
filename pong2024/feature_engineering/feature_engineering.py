import numpy as np

def extract_positions(observation):
    try:
      y_bottom = 34
      y_top = 194

      o = observation[y_bottom:y_top, :, 0]
      y_mask_left = (o[:, 19] == 213).astype(np.int32)
      y_mask_right = (o[:, 140] == 92).astype(np.int32)
      ball_mask = (o == 236).astype(np.int32)
      assert np.any(y_mask_left)
      assert np.any(y_mask_right)

      y_range = np.linspace(0, 1, y_top - y_bottom)
      x_range = np.linspace(0, 1, o.shape[1])

      y_pos_left = (y_range * y_mask_left).sum() / max(1, np.sum(y_mask_left))
      y_pos_right = (y_range * y_mask_right).sum() / max(1, np.sum(y_mask_right))

      x_ball_mask = ball_mask.max(axis=0)
      y_ball_mask = ball_mask.max(axis=1)
      x_ball = (x_range * x_ball_mask).sum() / max(1, x_ball_mask.sum())
      y_ball = (y_range * y_ball_mask).sum() / max(1, y_ball_mask.sum())
      is_ball_present = np.any(ball_mask)

      return y_pos_left, y_pos_right, x_ball, y_ball, is_ball_present
    except:
      print("error in position")
      return None, None, None, None, False
    

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def extract_velocities(current_obs, previous_obs):
    """
    Calculate velocities of the ball and paddles between two consecutive frames.
    
    Args:
        current_obs: Current frame observation
        previous_obs: Previous frame observation
    
    Returns:
        Tuple of (ball_vx, ball_vy, paddle_left_vy, paddle_right_vy)
    """
    if previous_obs is None:
        return 0, 0, 0, 0
    
    curr_pos = extract_positions(current_obs)
    prev_pos = extract_positions(previous_obs)
    
    if None in curr_pos or None in prev_pos:
        return 0, 0, 0, 0
        
    ball_vx = curr_pos[2] - prev_pos[2]  # x velocity of ball
    ball_vy = curr_pos[3] - prev_pos[3]  # y velocity of ball
    paddle_left_vy = curr_pos[0] - prev_pos[0]  # left paddle y velocity
    paddle_right_vy = curr_pos[1] - prev_pos[1]  # right paddle y velocity
    
    return ball_vx, ball_vy, paddle_left_vy, paddle_right_vy

def extract_distances(observation):
    """
    Calculate distances between ball and paddles.
    
    Args:
        observation: Current frame observation
    
    Returns:
        Tuple of (dist_to_left_paddle, dist_to_right_paddle, angle_to_left_paddle, angle_to_right_paddle)
    """
    y_left, y_right, x_ball, y_ball, is_ball = extract_positions(observation)
    
    if None in (y_left, y_right, x_ball, y_ball) or not is_ball:
        return 0, 0, 0, 0
        
    # Calculate Euclidean distances
    dist_to_left = np.sqrt((x_ball - 0.1)**2 + (y_ball - y_left)**2)  # 0.1 is approx x-position of left paddle
    dist_to_right = np.sqrt((x_ball - 0.9)**2 + (y_ball - y_right)**2)  # 0.9 is approx x-position of right paddle
    
    # Calculate angles (in radians)
    angle_to_left = np.arctan2(y_ball - y_left, x_ball - 0.1)
    angle_to_right = np.arctan2(y_ball - y_right, x_ball - 0.9)
    
    return dist_to_left, dist_to_right, angle_to_left, angle_to_right

def simplified_preprocess_image(observation, target_size=(28, 28), flip=False):
    """
    Simplified preprocessing to get binary values (0 and 1) from grayscale.
    
    Args:
        observation: Raw observation frame (210, 160, 3)
        target_size: Desired output size (height, width)
        flip: Whether to flip the image horizontally
    
    Returns:
        Binary observation as a 2D array with values strictly in {0, 1}
    """
    # Extract play area (remove score area)
    y_bottom, y_top = 34, 194
    play_area = observation[y_bottom:y_top, :, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(play_area, cv2.COLOR_RGB2GRAY)
    
    # Direct binary conversion: background is pure black (0), objects have intensity
    binary = (gray - 87 > 0).astype(np.uint8)
    
    def max_pool(img, kernel_size):
        """
        Perform max pooling with specified kernel size
        """
        h, w = img.shape
        pool_h, pool_w = kernel_size
        
        # Calculate output dimensions
        out_h = h // pool_h
        out_w = w // pool_w
        
        # Reshape and take max
        reshaped = img[:out_h*pool_h, :out_w*pool_w]  # trim if needed
        reshaped = reshaped.reshape(out_h, pool_h, out_w, pool_w)
        return reshaped.max(axis=(1,3))

    # In the preprocessing function:
    h_factor = binary.shape[0] // target_size[0]
    w_factor = binary.shape[1] // target_size[1]
    resized = max_pool(binary, (h_factor, w_factor))
    
    # Flip if needed (for player 2 perspective)
    if flip:
        resized = cv2.flip(resized, 1)
    
    return resized

def extract_game_state(observation):
    """
    Extract high-level game state features.
    
    Args:
        observation: Current frame observation
    
    Returns:
        Dict containing game state features
    """
    y_left, y_right, x_ball, y_ball, is_ball = extract_positions(observation)
    
    if None in (y_left, y_right, x_ball, y_ball):
        return {
            'ball_moving_right': False,
            'ball_moving_up': False,
            'left_paddle_center': 0.5,
            'right_paddle_center': 0.5,
            'ball_above_left_paddle': False,
            'ball_above_right_paddle': False
        }
    
    state = {
        'ball_moving_right': x_ball > 0.5,
        'ball_moving_up': y_ball < 0.5,
        'left_paddle_center': y_left,
        'right_paddle_center': y_right,
        'ball_above_left_paddle': y_ball < y_left,
        'ball_above_right_paddle': y_ball < y_right
    }
    
    return state

def combined_features(current_obs, previous_obs=None):
    """
    Combine all feature extractors into a single feature vector.
    
    Args:
        current_obs: Current frame observation
        previous_obs: Previous frame observation (optional)
    
    Returns:
        Dict containing all extracted features
    """
    # Basic positions
    y_left, y_right, x_ball, y_ball, is_ball = extract_positions(current_obs)
    
    # Velocities
    ball_vx, ball_vy, left_vy, right_vy = extract_velocities(current_obs, previous_obs)
    
    # Distances and angles
    dist_left, dist_right, angle_left, angle_right = extract_distances(current_obs)
    
    # Game state
    state = extract_game_state(current_obs)
    
    # Image features (downsampled)
    image_features = preprocess_image(current_obs, target_size=(42, 42))
    
    features = {
        'positions': {
            'y_left': y_left,
            'y_right': y_right,
            'x_ball': x_ball,
            'y_ball': y_ball,
            'is_ball': is_ball
        },
        'velocities': {
            'ball_vx': ball_vx,
            'ball_vy': ball_vy,
            'left_paddle_vy': left_vy,
            'right_paddle_vy': right_vy
        },
        'distances': {
            'dist_to_left': dist_left,
            'dist_to_right': dist_right,
            'angle_to_left': angle_left,
            'angle_to_right': angle_right
        },
        'game_state': state,
        'image_features': image_features
    }
    
    return features