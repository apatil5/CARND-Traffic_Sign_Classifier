Step 2: Design and Test a Model Architecture
Pre-process the Data Set (normalization, grayscale, etc.)
Dataset preprocessing :
The submission describes the preprocessing techniques used and why these techniques were chosen.

Larger data set is important for a good Model to predict the class correctly. The provided training data set has fewer number of samples for a few classes. Therefore firstly I augmented the training data set by adding slightly modified images of the existing training images. Class IDs are sorted based on number of training samples corresponding to each class and then first 15 classes with least number of samples are chosen. The chosen images are then rotated slightly to generate modified image for training.

Further the images in augmented data set are then converted to grayscale. It reduces computation load without compromising on features. The grayscaled images are then normalized about zero. Normalised data set is uniformly in all the dimensions and it helps in optimising the parameters faster and effectively.

Last step of preprocessing is to shuffle the normalized data to train the model effectively for maximum number of classes in a single batch.


Model Architecture
Dataset preprocessing :
The submission describes the preprocessing techniques used and why these techniques were chosen.

Larger data set is important for a good Model to predict the class correctly. The provided training data set has fewer number of samples for a few classes. Therefore firstly I augmented the training data set by adding slightly modified images of the existing training images. Class IDs are sorted based on number of training samples corresponding to each class and then first 15 classes with least number of samples are chosen. The chosen images are then rotated slightly to generate modified image for training.

Further the images in augmented data set are then converted to grayscale. It reduces computation load without compromising on features. The grayscaled images are then normalized about zero. Normalised data set is uniformly in all the dimensions and it helps in optimising the parameters faster and effectively.

Last step of preprocessing is to shuffle the normalized data to train the model effectively for maximum number of classes in a single batch.

The LeNet model is used as the base architecture. The number of parameters in hidden layers are modified and tuned in iterative process. It was noted that the traffic signs have different shapes and marking and thus have several different features like edges, curves,contrast etc. Therefore number of parameters are increased in iterative fashion to achieve 97% validation accuracy.

First layer is responsible for edges. Therefore number of filters is chosen and tried to cater edges and curves of markings. Hidden Layer 1 : 32x32x1 -> 5x5 convolution with 20 filters -> 28x28x20  -> ReLU -> maxPooling 2x2 -> 14x14x20

Second layer is responsible for basic underlying shapes of signs. Therefore number of filters is chosen and tried to cater general shapes Hidden Layer 2 : 14x14x20 -> 5x5 convolution with 80 filters -> 10x10x80  -> ReLU -> maxPooling 2x2 -> 5x5x80

Size of Connected layers were chosen and tried to train the model differentiate between closely resembling signs. Fully connected layer 1 : flatten(5x5x80) -> 2000 ->Dropout layer while dropout probability to avoid overfitting 50%-> 240

Fully connected layer 2 : 240 -> Dropout layer probability 50% to avoid overfitting->150

Fully connected layer 3 : 150 -> 43

The number of parameters were initialised considering all the factors explained above. However final number of parameters is fixed after several iterations.


Train, Validate and Test the Model
Model Training and solution Approach
Adamoptmizer is used with training batch size 43.

Target is to achieve validation accuracy of 97% or above. The pipeline stops training the model if the target accuracy is achieved or epoch value hits 100, whichever happens first.

The learning rate is kept .00035. It slows down the learning process a bit but helps in achieving the target.

The graph presented below shows the learning progress.
