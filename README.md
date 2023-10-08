# Age-Race-Classification
Task - Predict the age group of individuals from an image, which may contain one or multiple people, and determine their respective age groups, binned in sizes of six.

Raw Dataset - https://susanqq.github.io/UTKFace/

Previously : Using SMOTE for resampling, bad as SMOTE creates noisy data, spatial inforamtion lost

Flow options : (Cannot send images of varying dimensions as batch, numpy is a bit.. so forget FCN and SPP ) Use transfer learning or normal cnn ?
  1. Padding with black pixels to max height, max width. Transfer Learning
  2. Same as above sampling with GAN
  3. Multi Scale training by interpolating images to their nearest 100 then transfer learning
  4. Same as above sampling with GAN
  5. Progressive resizing then transfer learning
  6. Same as above sampling with GAN
