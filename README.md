Hospital Gesture Recognition System 



I built this project to create a touchless gesture recognition system that can understand hand gestures using a camera and convert them into meaningful actions in a hospital environment.
The main idea was to design a system that allows patients or doctors to interact with a computer without touching any device. This is especially important in medical environments where hygiene, accessibility, and touchless interaction are critical.

This project is not just a classification model. It is a complete pipeline that starts from dataset preparation and training, and ends with real-time gesture recognition and action execution.

Project Overview
The system works as follows:
The camera captures a hand gesture
The image is processed and sent to the trained model
The model predicts the gesture class
The gesture is mapped to a specific medical action based on confidence and priority
Examples:

✋ Palm → Emergency Stop
✌️ Peace → Patient needs help
👍 Like → Confirm / Positive response
👎 Dislike → Reject / Negative response

This makes the system capable of real-world interaction, not just prediction.
Dataset & Preparation
I used a dataset containing 18 different hand gesture classes.
The dataset was manually split into:
70% Training
15% Validation
15% Testing
This split was done intentionally to avoid data leakage and ensure that validation and test results reflect the true performance of the model.
The dataset was organized into separate folders:
gesture_dataset_split/
    train/
    val/
    test/
    
This structure allows proper loading using ImageDataGenerator.
One important observation during training was that some classes achieved very high accuracy (up to 95%) because they had clearer and more sufficient data, while other classes had lower accuracy due to fewer samples. This highlighted the importance of balanced datasets.
The overall test accuracy achieved was:

75.33%

Model Architecture

The model is built using a CNN based on MobileNetV2 with Transfer Learning.
MobileNetV2 was used as the base model because it is pretrained on ImageNet and provides strong feature extraction capabilities.
On top of the base model, I added custom classification layers:
GlobalAveragePooling
Dense layer (512 neurons)
BatchNormalization
Dropout
Dense layer (256 neurons)
Output layer with Softmax activation
I also applied fine-tuning to the last layers of MobileNetV2 so the model could specialize in gesture recognition instead of general image features.
This improved:

Model accuracy
Feature adaptation
Generalization performance
Reduced overfitting
Training Process
For training, I used:

ImageDataGenerator for loading and preprocessing images
Data augmentation to improve generalization
Adam optimizer with a low learning rate for stable fine-tuning
I also used training callbacks:
EarlyStopping to prevent overfitting
ReduceLROnPlateau to improve training stability
ModelCheckpoint to save the best model
The final trained model was saved as:

Project_nti_final.py

Gesture Mapping System
After training, I implemented a gesture-to-action mapping system using a configuration file.
This allows each gesture prediction to trigger a meaningful system action.
Examples:

palm → emergency_stop
peace → need_help
like → confirm
dislike → reject
This makes the system function as a real interaction system, not just a classifier.
Gesture Action Handler
I implemented a custom class responsible for handling gesture predictions and converting them into actions.
Its responsibilities include:

Reading gesture predictions from the model
Checking if confidence exceeds a defined threshold
Executing the correct action
Preventing repeated triggers using debouncing
Logging action history
This improves the stability and reliability of the system.
Real-Time Testing

After training, the model was tested using a live camera feed.
The model was able to recognize gestures such as:
✋ Palm
👍 Like
👎 Dislike
✌️ Peace
with high confidence and convert them into the correct actions in real time.
This confirms that the system is capable of working in real-time environments.

Project Structure:

dataset/
gesture_dataset_split/
model/
training/
inference/
gesture_action_handler.py
medical_gesture_mapping.json
hospital_gesture_model_final.keras
class_names.json
Key Learnings

Through this project, I gained practical experience in:

Proper dataset preparation and splitting
Avoiding data leakage
Applying Transfer Learning using MobileNetV2
Fine-tuning deep learning models
Improving model generalization
Building a complete real-time AI pipeline
Converting model predictions into real system actions
This project helped me understand the full deep learning workflow from data preparation to real-world deployment.

If you want, I can also add:

Installation section
How to run the camera inference
In terminal:python test_camera.py
