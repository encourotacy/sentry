#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <optional>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "tools/logger.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"buff", "r"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0F;
    std::vector<cv::Point2f> kpt;
  };

  YOLO11_BUFF(const std::string & config);
  YOLO11_BUFF(const std::string & config, const std::string & model_key);
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  struct OutputLayout
  {
    int channels = 0;
    int candidates = 0;
    bool channel_first = true;
    int keypoint_stride = 0;
    const char * format_name = "unknown";
  };

  struct Candidate
  {
    cv::Rect rect;
    float confidence = 0.0F;
    std::vector<cv::Point2f> keypoints;
  };

  ov::Core core;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  ov::Tensor input_tensor;
  std::string model_path_;
  bool output_layout_logged_ = false;
  const int NUM_POINTS = 6;

  void loadModel(const std::string & model_path);
  static std::string resolveModelPath(
    const YAML::Node & yaml, const std::string & model_key);
  std::optional<OutputLayout> detectOutputLayout(const ov::Shape & shape);
  float outputAt(
    const float * data, const OutputLayout & layout, int channel, int candidate) const;
  std::vector<Candidate> parseCandidates(const ov::Tensor & output, float factor);
  void drawObject(cv::Mat & image, const Object & obj, const cv::Scalar & point_color) const;
  void convert(
    const cv::Mat & input, cv::Mat & output, const bool normalize, const bool exchangeRB) const;
  float fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const;
  void printInputAndOutputsInfo(const ov::Model & network);
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif
