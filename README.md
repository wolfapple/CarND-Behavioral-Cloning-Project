# 1. Introduction
The purpose of this project is to build and learn a deep neural network that can mimic the behavior of humans driving a car. A simple simulator is used for this purpose. When we drive a car, the simulator stores images and steering angles for training. You can learn the neural network with the stored training data and check the results of training in the simulator's autonomous mode.

# 2. Overview of the simulator
## download link
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

## Running the Simulator
When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like.

### Training mode
  * You’ll enter the simulator and be able to drive the car with your arrow keys.

### Autonomous mode
  * Set up your development environment with the environment.yml
  * Run the server : `python drive.py mode.json`
  * You should see the car move around

# 3. Data Collection
Enter training mode in the simulator, start driving the car. When you are ready, hit the record button in the top right to start recording. Continue driving for a few minutes, hit the record button again to stop recording. IMG folder contains all the frames of your driving. Each row in driving_log.csv file correlates your image with the steering angle, throttle, brake, and speed of your car.

You can use sample data for track 1 [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

After learning using only the sample data for Track 1 above, successfully completing Track 2 was my challenge.

# 4. Network Architecture
My convolutional neural network architecture was inspired by NVIDIA's End to End Learning for Self-Driving Cars [paper](https://arxiv.org/pdf/1604.07316v1.pdf). Starting from this base model, I refer to various papers and made trial and error several times, finally making the following architecture.

| Lyaer (type)                    | Output Shape       | Param # |
|---------------------------------|--------------------|---------|
| lambda_1 (Lambda)               | (None, 64, 64, 3)  | 0       |
| convolution2d_1 (Convolution2D) | (None, 30, 30, 24) | 1824    |
| prelu_1 (PReLU)                 | (None, 30, 30, 24) | 21600   |

## Things to note are
  * I've added a lambda layer on the top similar to the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py) to normalize the data
    ```python
    Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3))
    ```

  * 트레이닝 시간 단축, weight 파일 용량 축소, 첫3개 valid, 마지막2개 maxpooling
  * PReLU 사용, he_normal, he_unicorm init
  * 첫 두개의 fc에다 dropout 0.5 적용

The main difference between our model and the NVIDIA mode is than we did use MaxPooling layers just after each Convolutional Layer in order to cut down training time. For more details about our network architecture please refer following figure.

# 5. Training
## Data Description
The dataset consists of 8036 rows. Since each row contains images corresponding to three cameras on the left, right, and center, a total of 24,108 images exist.

But, training data is very unbalanced. The 1st track contains a lot of shallow turns and straight road segments. So, the majority of the dataset's steering angles are zeros. Huge number of zeros is definitely going to bias our model towards predicting zeros.

[histogram 하나 첨부]

## Data preprocessing
I removed the useless part of the image(past the horizon, hood of the car) and kept track. And I resized the resulting image to a 64x64 in order to reduce training time. Resized images are fed into the neural network.

[preprocess된 이미지 하나 첨부]

## Data Augmentation
Augmentation refers to the process of generating new training data from a smaller data set. This helps us extract as much information from data as possible.

Since I wanted to proceed with only the given data set if possible, I used some data augmentation techniques to generate new learning data.

### Randomly choosing camera
During the training, the simulator captures data from left, center, and right cameras. Using images taken from left and right cameras to simulate and recover when the car drifts off the center of the road. My approach was to add/substract a static offset from the angle when choosing the left/right camera.

[이미지 세개랑 각도 테이블 첨부]

### Random shear
I applied random shear operation. The image is sheared horizontally to simulate a bending road. The pixels at the bottom of the image were held fixed while the top row was moved randomly to the left or right. The steering angle was changed proportionally to the shearing angle. However, I choose images with 0.9 probability for the random shearing process. I kept 10% of original images in order to help the car to navigate in the training track.

[sheared 이미지 첨부]

### Random flip
Each image was randomly horizontally flipped and negate the steering angle with equal probability. I think, this will have the effect of evenly turning left and right.

[이미지 첨부]

### Random gamma correction
Chaging brightness to simulate differnt lighting conditions. Random gamma correction is used as an alternative method changing the brightness of training images.

[이미지 첨부]

### Random shift vertically
The roads on the second track have hills and downhill, and the car often jumps while driving. To simulate such a road situation, I shifted the image vertically randomly. This work was applied after image preprocessing.

[이미지 첨부]

## Data Generators

# 6. Simulation

# 7. Evaluation

# 8. Conclusions

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit
for solving the given problem.

The README provides sufficient details of the characteristics and qualities of the architecture,
such as the type of model used, the number of layers, the size of each layer.
Visualizations emphasizing particular qualities of the architecture are encouraged.

The README describes how the model was trained and what the characteristics of the dataset are.
Information such as how the dataset was generated and examples of images from the dataset should be included.