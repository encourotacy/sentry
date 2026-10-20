#include "yolo11_buff.hpp"

#include <sstream>
#include <stdexcept>

const double ConfidenceThreshold = 0.40f;
const double IouThreshold = 0.60f;

namespace
{
cv::Mat enhance_image(const cv::Mat & image)
{
  cv::Mat result;
  cv::convertScaleAbs(image, result, 1.2, 10);
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::filter2D(result, result, -1, kernel);
  return result;
}

std::string shape_to_string(const ov::Shape & shape)
{
  std::ostringstream stream;
  stream << shape;
  return stream.str();
}
}  // namespace

namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  loadModel(resolveModelPath(yaml, "model"));
}

YOLO11_BUFF::YOLO11_BUFF(const std::string & config, const std::string & model_key)
{
  auto yaml = YAML::LoadFile(config);
  loadModel(resolveModelPath(yaml, model_key));
}

void YOLO11_BUFF::loadModel(const std::string & model_path)
{
  model_path_ = model_path;
  model = core.read_model(model_path);
  compiled_model = core.compile_model(model, "CPU");
  infer_request = compiled_model.create_infer_request();
  input_tensor = infer_request.get_input_tensor();
  input_tensor.set_shape({1, 3, 640, 640});
  tools::logger()->info("[YOLO11_BUFF] loaded model: {}", model_path_);
}

std::string YOLO11_BUFF::resolveModelPath(
  const YAML::Node & yaml, const std::string & model_key)
{
  if (yaml[model_key]) {
    return yaml[model_key].as<std::string>();
  }
  if (yaml["model"]) {
    tools::logger()->warn(
      "[YOLO11_BUFF] {} 未配置，回退使用 model={}", model_key, yaml["model"].as<std::string>());
    return yaml["model"].as<std::string>();
  }
  throw std::runtime_error("[YOLO11_BUFF] yaml 缺少 model 配置");
}

std::optional<YOLO11_BUFF::OutputLayout> YOLO11_BUFF::detectOutputLayout(
  const ov::Shape & shape)
{
  if (shape.size() != 3 || shape[0] != 1) {
    tools::logger()->warn("[YOLO11_BUFF] unsupported output shape: {}", shape_to_string(shape));
    return std::nullopt;
  }

  OutputLayout layout;
  if (shape[1] == 17 || shape[1] == 23) {
    layout.channels = static_cast<int>(shape[1]);
    layout.candidates = static_cast<int>(shape[2]);
    layout.channel_first = true;
  } else if (shape[2] == 17 || shape[2] == 23) {
    layout.channels = static_cast<int>(shape[2]);
    layout.candidates = static_cast<int>(shape[1]);
    layout.channel_first = false;
  } else {
    tools::logger()->warn(
      "[YOLO11_BUFF] unsupported output shape: {}, expected channels 17 or 23",
      shape_to_string(shape));
    return std::nullopt;
  }

  if (layout.channels == 17) {
    layout.keypoint_stride = 2;
    layout.format_name = "xy";
  } else {
    layout.keypoint_stride = 3;
    layout.format_name = "xy_score";
  }

  if (!output_layout_logged_) {
    tools::logger()->info(
      "[YOLO11_BUFF] output layout: shape={}, channels={}, candidates={}, format={}, order={}",
      shape_to_string(shape), layout.channels, layout.candidates, layout.format_name,
      layout.channel_first ? "channel_first" : "candidate_first");
    output_layout_logged_ = true;
  }

  return layout;
}

float YOLO11_BUFF::outputAt(
  const float * data, const OutputLayout & layout, int channel, int candidate) const
{
  if (layout.channel_first) {
    return data[channel * layout.candidates + candidate];
  }
  return data[candidate * layout.channels + channel];
}

std::vector<YOLO11_BUFF::Candidate> YOLO11_BUFF::parseCandidates(
  const ov::Tensor & output, float factor)
{
  const auto layout = detectOutputLayout(output.get_shape());
  if (!layout.has_value()) {
    return {};
  }

  const float * output_buffer = output.data<const float>();
  std::vector<Candidate> candidates;
  for (int i = 0; i < layout->candidates; ++i) {
    const float score = outputAt(output_buffer, *layout, 4, i);
    if (score <= ConfidenceThreshold) continue;

    const float cx = outputAt(output_buffer, *layout, 0, i);
    const float cy = outputAt(output_buffer, *layout, 1, i);
    const float ow = outputAt(output_buffer, *layout, 2, i);
    const float oh = outputAt(output_buffer, *layout, 3, i);

    Candidate candidate;
    candidate.rect.x = static_cast<int>((cx - 0.5F * ow) * factor);
    candidate.rect.y = static_cast<int>((cy - 0.5F * oh) * factor);
    candidate.rect.width = static_cast<int>(ow * factor);
    candidate.rect.height = static_cast<int>(oh * factor);
    candidate.confidence = score;
    candidate.keypoints.reserve(NUM_POINTS);
    for (int j = 0; j < NUM_POINTS; ++j) {
      const int base = 5 + j * layout->keypoint_stride;
      const float x = outputAt(output_buffer, *layout, base, i) * factor;
      const float y = outputAt(output_buffer, *layout, base + 1, i) * factor;
      candidate.keypoints.emplace_back(x, y);
    }
    candidates.push_back(std::move(candidate));
  }

  return candidates;
}

void YOLO11_BUFF::drawObject(
  cv::Mat & image, const Object & obj, const cv::Scalar & point_color) const
{
  cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
  const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);
  const cv::Size textSize =
    cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
  const cv::Rect textBox(
    obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
  cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
  cv::putText(
    image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX,
    0.5, cv::Scalar(0, 0, 0));
  for (int j = 0; j < static_cast<int>(obj.kpt.size()); ++j) {
    cv::circle(image, obj.kpt[j], 2, point_color, -1, cv::LINE_AA);
  }
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<YOLO11_BUFF::Object>();
  }
  const float factor = fill_tensor_data_image(input_tensor, image);
  infer_request.infer();
  const ov::Tensor output = infer_request.get_output_tensor();
  const auto candidates = parseCandidates(output, factor);
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  boxes.reserve(candidates.size());
  confidences.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    boxes.push_back(candidate.rect);
    confidences.push_back(candidate.confidence);
  }
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);
  std::vector<Object> object_result;
  for (size_t i = 0; i < indexes.size(); ++i) {
    Object obj;
    const int index = indexes[i];
    obj.rect = candidates[index].rect;
    obj.prob = candidates[index].confidence;
    obj.kpt = candidates[index].keypoints;
    object_result.push_back(obj);
    drawObject(image, obj, cv::Scalar(255, 0, 0));
  }
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  const float factor = fill_tensor_data_image(input_tensor, image);
  infer_request.infer();
  const ov::Tensor output = infer_request.get_output_tensor();
  const auto candidates = parseCandidates(output, factor);
  int best_index = -1;
  float max_confidence = 0.0f;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].confidence > max_confidence) {
      max_confidence = candidates[i].confidence;
      best_index = static_cast<int>(i);
    }
  }
  std::vector<Object> object_result;
  if (best_index >= 0 && max_confidence > ConfidenceThreshold) {
    Object obj;
    obj.rect = candidates[best_index].rect;
    obj.prob = candidates[best_index].confidence;
    obj.kpt = candidates[best_index].keypoints;
    object_result.push_back(obj);
    if (max_confidence < 0.7) save(std::to_string(start), image);
    drawObject(image, obj, cv::Scalar(255, 255, 0));
    for (int i = 0; i < NUM_POINTS; ++i) {
      cv::putText(
        image, std::to_string(i + 1), obj.kpt[i] + cv::Point2f(5, -5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

float YOLO11_BUFF::fill_tensor_data_image(
  ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t num_channels = tensor_shape[1];
  const size_t height = tensor_shape[2];
  const size_t width = tensor_shape[3];
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{
    scale, 0.0, 0.0, 0.0, scale, 0.0,
  };
  cv::Mat blob_image;
  if (scale < 1.0f) {
    cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
    convert(blob_image, blob_image, true, true);
  } else {
    convert(input_image, blob_image, true, true);
    cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
  }
  float * const input_tensor_data = input_tensor.data<float>();
  for (size_t c = 0; c < num_channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        input_tensor_data[c * width * height + h * width + w] =
          blob_image.at<cv::Vec<float, 3>>(h, w)[c];
      }
    }
  }
  return 1 / scale;
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;
  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;
    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;
    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;
    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }
  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;
    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;
    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;
    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff
