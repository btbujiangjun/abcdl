# abcdl
A c++ light Deep Learning framework for ABC </bt>

## DNN example <br>
### 1. Config layers </bt>
  std::vector<abcdl::dnn::Layer\*> layers; \<bt>
  layers.push_back(new abcdl::dnn::InputLayer(784)); \<bt>
  layers.push_back(new abcdl::dnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc())); \<bt>
  layers.push_back(new abcdl::dnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost())); \<bt>

### 2.Initailize Network \<bt>
  abcdl::dnn::DNN dnn; \<bt>
  dnn.set_layers(layers); \<bt>

### 3.Load training data \<bt>
  abcdl::utils::MnistHelper<real> helper; \<bt>
  
  abcdl::algebra::Mat train_data; \<bt>
  helper.read_image("data/mnist/train-images-idx3-ubyte", &train_data, 60000); \<bt>
  
  abcdl::algebra::Mat train_label; \<bt>
  helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", &train_label, 10000); \<bt>
  
### 4. Train network \<bt>
  dnn.train(train_data, train_label); \<bt>

### 5.Predict \<bt>
  abcdl::algebra::Mat result; \<bt>
  abcdl::algebra::Mat predict_data; \<bt>
  helper.read_image("data/mnist/t10k-images-idx3-ubyte", &predict_data, 1); \<bt>
  dnn.predict(result, predict_data); \<bt>

### 6. Serialize model \<bt>
  const std::string path = "data/dnn.model"; \<bt>
  dnn.write_model(path); \<bt>

### 7. Deserialize model \<bt>
  dnn.load_model(path); \<bt>
