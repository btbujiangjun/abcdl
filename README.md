# abcdl
A c++ light Deep Learning framework for ABC

DNN example:

1. Config layers
std::vector<abcdl::dnn::Layer*> layers;
layers.push_back(new abcdl::dnn::InputLayer(784));
layers.push_back(new abcdl::dnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc()));
layers.push_back(new abcdl::dnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));

2.Initailize Network
abcdl::dnn::DNN dnn;
dnn.set_layers(layers);

3.Load training data
abcdl::utils::MnistHelper<real> helper;
abcdl::algebra::Mat train_data;
helper.read_image("data/mnist/train-images-idx3-ubyte", &train_data, 60000);
abcdl::algebra::Mat train_label;
helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", &train_label, 10000);
  
4. Train network
dnn.train(train_data, train_label);

5.Predict
abcdl::algebra::Mat result;
abcdl::algebra::Mat predict_data;
helper.read_image("data/mnist/t10k-images-idx3-ubyte", &predict_data, 1);
dnn.predict(result, predict_data);

6. Serialize model
const std::string path = "data/dnn.model";
dnn.write_model(path);

7. Deserialize model
dnn.load_model(path);
