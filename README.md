# Age-Race-Classification
Task - Age group prediction, given an image of one or multiple people predict their respective age groups

Raw Dataset - https://susanqq.github.io/UTKFace/

Age_folder_creation - Create folders for all images extracted and all cropped images of faces using dlib frontal face detector, INTER_AREA interpolation for shirinking dimensions, cubic for enlarging

Age_image_Data - Create image array, perform oversampling using SMOTE and random undersampling to balance out class size

Age_model - CNN model for prediction
