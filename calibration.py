import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVR

# this is a example, the calibration points and model load and inference, and the fit code need to change in later
# Define the positions of the 9 calibration points on the screen
calibration_points = np.array([[0.1, 0.1, 0.0], [0.5, 0.1, 0.0], [0.9, 0.1, 0.0],
                               [0.1, 0.5, 0.0], [0.5, 0.5, 0.0], [0.9, 0.5, 0.0],
                               [0.1, 0.9, 0.0], [0.5, 0.9, 0.0], [0.9, 0.9, 0.0]])

# Define the parameters for the calibration process
num_calibration_points = 9
num_calibration_frames = 60 * 3 # 3 seconds with 60 frames per second

# Load the 3D gaze estimation model
model = torch.load("gaze_estimation_model.pt")

# Collect eye positions and corresponding calibration point positions during calibration
eye_positions = [] # List of eye positions
calibration_targets = [] # List of corresponding calibration point positions
for i in range(num_calibration_points):
    for j in range(num_calibration_frames):
        # Show the i-th calibration point on the screen
        # Wait for the user to fixate on the point for 1/60 seconds
        # Record the user's eye position
        # Add the eye position and calibration point position to the lists
        # Repeat for all 9 calibration points and all 180 frames per point
        pass

# Convert the data to PyTorch tensors and create a DataLoader for batch processing
eye_positions = torch.Tensor(eye_positions)
calibration_targets = torch.Tensor(calibration_targets)
dataset = TensorDataset(eye_positions, calibration_targets)
dataloader = DataLoader(dataset, batch_size=32)


# Use the hook of PyTorch to hook the last fully connection layer to have high dimension vector
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.fc[0].register_forward_hook(get_activation('out'))


# Use the 3D gaze estimation model to predict gaze directions for each calibration frame
predictions = []
predictions_hooked = []
for i, batch in enumerate(dataloader):
    inputs, _ = batch
    
    with torch.no_grad():
      outputs = model(inputs)
      pred_hooked = list(activation['out'].cpu().detach().numpy())
      
    predictions.append(outputs)
    predictions_hooked.append(pred_hooked)
predictions = torch.cat(predictions, dim=0)
predictions_hooked = torch.cat(predictions_hooked, dim=0)



# Fit an SVR model to the calibration data
svr = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)
svr.fit(predictions_hooked, calibration_targets)

# Save the calibrated SVR model
with open("calibrated_svr.pkl", "wb") as f:
    pickle.dump(svr, f)

# Apply the calibrated SVR model to estimate the user's gaze position in real time
while True:
    # Continuously track the position of the user's eyes
    # Use the 3D gaze estimation model to predict the gaze direction
    # Use the calibrated SVR model to map the gaze direction to the corresponding screen position
    # Use the predicted screen position to control the application or system
    pass
