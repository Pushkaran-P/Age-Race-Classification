# Age-Race-Classification
Task - Predict the age group of individuals from an image, which may contain one or multiple people, and determine their respective age groups, binned in sizes of six.

Raw Dataset - https://susanqq.github.io/UTKFace/

Age_folder_creation - Create folders for all images extracted and all cropped images of faces using dlib frontal face detector, INTER_AREA interpolation for shirinking dimensions, cubic for enlarging

Age_image_Data - Create image array, perform oversampling using SMOTE and random undersampling to balance out class size

Age_model - CNN model for prediction, gives 64,65% train and test acc for now 

Current Task - Multi Scale Training or Pad black pixels (which place should i use gan ?)
