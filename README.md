This is my cumulative project for APS360

The system recognises human actions from body pose alone. At inference, a frozen pretrained MoveNet extracts 17 skeletal keypoints per 
frame from single-camera video (e.g. a webcam); a custom-trained LSTM then reads the 30-frame sequence of keypoints and classifies the
movement into one of MPOSE2021's ~20 action categories (e.g. walking, running, jumping, boxing, hand-waving). A static MLP on the time-
averaged pose serves as a baseline to isolate the value of temporal modelling.    

Note: This repsoitory is for my reference only. DO NOT COPY WORK FROM THIS REPOSITORY. Plagiarism is a serious offence. Read UofT's plagiarism policies here: https://www.academicintegrity.utoronto.ca/process-and-procedures/

The owner of this repository is NOT RESPONSIBLE FOR ANY WORK COPIED AND PLAGIARIZED.
