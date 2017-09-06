# abcdl
A c++ light `Deep Learning` framework for ABC, Include `DNN`, `CNN` and `RNN`. <br>

## DNN example <br>
### 1. Configure layers <br>
  std::vector\<abcdl::dnn::Layer*> layers; <br>
  layers.push_back(new abcdl::dnn::InputLayer(784)); <br>
  layers.push_back(new abcdl::dnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc())); <br>
  layers.push_back(new abcdl::dnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost())); <br>

### 2.Initailize Network <br>
  abcdl::dnn::DNN dnn; <br>
  dnn.set_layers(layers); <br>

### 3.Load training data <br>
  abcdl::utils::MnistHelper<real> helper; <br>
  
  abcdl::algebra::Mat train_data; <br>
  helper.read_image("data/mnist/train-images-idx3-ubyte", &train_data, 60000); <br>
  
  abcdl::algebra::Mat train_label; <br>
  helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", &train_label, 10000); <br>
  
### 4. Train network <br>
  dnn.train(train_data, train_label); <br>

### 5.Predict <br>
  abcdl::algebra::Mat result; <br>
  abcdl::algebra::Mat predict_data; <br>
  helper.read_image("data/mnist/t10k-images-idx3-ubyte", &predict_data, 1); <br>
  dnn.predict(result, predict_data); <br>

### 6. Serialize model <br>
  const std::string path = "data/dnn.model"; <br>
  dnn.write_model(path); <br>

### 7. Deserialize model <br>
  dnn.load_model(path); <br>
