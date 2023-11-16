# Age-Race-Classification
Task - Predict the age group of individuals from an image, which may contain one or multiple people, and determine their respective age groups, binned in sizes of six.

Raw Dataset - https://susanqq.github.io/UTKFace/

Previously : Using SMOTE for resampling, bad as SMOTE creates noisy data, spatial inforamtion lost

Flow options : (Cannot send images of varying dimensions as batch, numpy is a bit.. so forget FCN and SPP )
  1. Use GAN (check personal log)
  2. Progressive resizing
  3. Adaptive pooling
Future : Hyperparameter Tuning
