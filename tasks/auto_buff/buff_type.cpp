#include "buff_type.hpp"

#include <algorithm>
#include <limits>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{

FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
  : center(keypoints_center), type(t)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
}

FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  if (t != _unlight) exit(-1);
}

PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
  : r_center(center),
    light_num(ts.size()),
    hit_ring(0),
    activated_arms(0),
    energy_type(EnergyType::SMALL),
    timestamp(0.0),
    distance_to_target(5.0),
    selected_board_index(0),
    candidate_board_index(-1),
    board_hit_rings(ts.size(), 0),
    board_confidences(ts.size(), 0.0),
    board_tracked_frames(ts.size(), 1),
    board_areas(ts.size(), 0.0)
{
  if (light_num == 0) {
    tools::logger()->debug("[PowerRune] 识别出错!");
    unsolvable_ = true;
    return;
  }

  for (auto & t : ts) {
    t.type = _light;
    t.angle = atan_angle(t.center);
  }

  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.angle < b.angle;
  });
  fanblades = ts;

  activated_arms = 0;
  for (const auto & blade : fanblades) {
    if (blade.type == _light) {
      activated_arms++;
    }
  }

  if (last_powerrune.has_value() && !last_powerrune->fanblades.empty()) {
    cv::Point2f last_selected_center = last_powerrune->target().center;
    float min_distance = std::numeric_limits<float>::max();
    int best_index = 0;
    for (size_t i = 0; i < fanblades.size(); ++i) {
      float distance = cv::norm(fanblades[i].center - last_selected_center);
      if (distance < min_distance) {
        min_distance = distance;
        best_index = static_cast<int>(i);
      }
    }
    selected_board_index = best_index;
  } else {
    selected_board_index = 0;
  }
}

int PowerRune::getLightArmCount() const
{
  int count = 0;
  for (const auto & blade : fanblades) {
    if (blade.type == _light) {
      count++;
    }
  }
  return count;
}

int PowerRune::getCurrentTargetArmIndex() const
{
  return resolved_selected_index();
}

double PowerRune::getFanBladeAngle(int index) const
{
  if (index < 0 || index >= static_cast<int>(fanblades.size())) {
    return 0.0;
  }
  return fanblades[index].angle;
}

std::vector<int> PowerRune::getLitBoardIndices() const
{
  std::vector<int> indices;
  for (size_t i = 0; i < fanblades.size(); i++) {
    if (isBoardLit(static_cast<int>(i))) indices.push_back(static_cast<int>(i));
  }
  return indices;
}

bool PowerRune::isBoardLit(int index) const
{
  if (index < 0 || index >= static_cast<int>(fanblades.size())) return false;
  return fanblades[index].type == _light || fanblades[index].type == _target;
}

void PowerRune::setBoardHitRing(int index, int ring)
{
  if (index < 0) return;
  if (index >= static_cast<int>(board_hit_rings.size())) board_hit_rings.resize(index + 1, 0);
  board_hit_rings[index] = ring;
}

int PowerRune::getHitRingForBoard(int index) const
{
  if (index < 0 || index >= static_cast<int>(board_hit_rings.size())) return 0;
  return board_hit_rings[index];
}

void PowerRune::setSelectedBoardIndex(int index)
{
  if (index < 0) {
    selected_board_index = -1;
    return;
  }
  if (isBoardLit(index)) selected_board_index = index;
}

void PowerRune::setCandidateBoardIndex(int index)
{
  if (index < 0) {
    candidate_board_index = -1;
    return;
  }
  if (isBoardLit(index)) candidate_board_index = index;
}

double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}

int PowerRune::resolved_selected_index() const
{
  if (isBoardLit(selected_board_index)) return selected_board_index;

  for (size_t i = 0; i < fanblades.size(); i++) {
    if (isBoardLit(static_cast<int>(i))) return static_cast<int>(i);
  }

  return 0;
}

void DoubleBoardController::reset_tracking()
{
  active_board_lost_frames_ = 0;
  last_active_center_ = {0.0F, 0.0F};
  last_candidate_center_ = {0.0F, 0.0F};
  state_.reset();
}

void DoubleBoardController::start_tracking(const PowerRune & rune)
{
  auto lit_indices = rune.getLitBoardIndices();
  state_.active_board_index = pick_best_board(rune, lit_indices);
  active_board_lost_frames_ = 0;
  if (state_.active_board_index >= 0 &&
      state_.active_board_index < static_cast<int>(rune.fanblades.size()))
  {
    last_active_center_ = rune.fanblades[state_.active_board_index].center;
  }
  update_candidate_tracking(rune, state_.active_board_index);
}

int DoubleBoardController::pick_best_board(
  const PowerRune & rune, const std::vector<int> & candidates) const
{
  if (candidates.empty()) return -1;

  cv::Point2f image_center(320.0F, 240.0F);
  double best_cost = std::numeric_limits<double>::max();
  int best_index = -1;

  for (int index : candidates) {
    if (!rune.isBoardLit(index)) continue;

    const auto & board = rune.fanblades[index];
    double yaw_offset_norm = std::clamp(
      static_cast<double>(std::abs(board.center.x - image_center.x) / image_center.x), 0.0, 1.0);
    double pitch_offset_norm = std::clamp(
      static_cast<double>(std::abs(board.center.y - image_center.y) / image_center.y), 0.0, 1.0);
    double confidence = index < static_cast<int>(rune.board_confidences.size()) ?
      rune.board_confidences[index] :
      0.0;
    int tracked_frames = index < static_cast<int>(rune.board_tracked_frames.size()) ?
      rune.board_tracked_frames[index] :
      1;
    double confidence_penalty = 1.0 - std::clamp(confidence, 0.0, 1.0);
    double stability_penalty =
      1.0 -
      std::clamp(tracked_frames / static_cast<double>(MAX_TRACK_FRAMES), 0.0, 1.0);
    double cost =
      0.60 * yaw_offset_norm + 0.20 * pitch_offset_norm + 0.10 * confidence_penalty +
      0.10 * stability_penalty;

    bool replace = false;
    if (best_index < 0 || cost < best_cost - 1e-6) {
      replace = true;
    } else if (std::abs(cost - best_cost) <= 1e-6) {
      double best_confidence = best_index < static_cast<int>(rune.board_confidences.size()) ?
        rune.board_confidences[best_index] :
        0.0;
      int best_tracked_frames = best_index < static_cast<int>(rune.board_tracked_frames.size()) ?
        rune.board_tracked_frames[best_index] :
        1;
      double best_yaw_offset_norm = std::clamp(
        static_cast<double>(
          std::abs(rune.fanblades[best_index].center.x - image_center.x) / image_center.x),
        0.0, 1.0);

      if (confidence > best_confidence + 1e-6) {
        replace = true;
      } else if (
        std::abs(confidence - best_confidence) <= 1e-6 &&
        yaw_offset_norm < best_yaw_offset_norm - 1e-6)
      {
        replace = true;
      } else if (
        std::abs(confidence - best_confidence) <= 1e-6 &&
        std::abs(yaw_offset_norm - best_yaw_offset_norm) <= 1e-6 &&
        tracked_frames > best_tracked_frames)
      {
        replace = true;
      } else if (
        std::abs(confidence - best_confidence) <= 1e-6 &&
        std::abs(yaw_offset_norm - best_yaw_offset_norm) <= 1e-6 &&
        tracked_frames == best_tracked_frames && index < best_index)
      {
        replace = true;
      }
    }

    if (replace) {
      best_cost = cost;
      best_index = index;
    }
  }

  return best_index;
}

int DoubleBoardController::rematch_board(
  const PowerRune & rune, const cv::Point2f & last_center, int exclude_index) const
{
  auto lit_indices = rune.getLitBoardIndices();
  if (lit_indices.empty()) return -1;

  float radius = 0.0F;
  for (int index : lit_indices) {
    radius += cv::norm(rune.fanblades[index].center - rune.r_center);
  }
  radius /= static_cast<float>(lit_indices.size());

  const float rematch_threshold = std::max(40.0F, radius * 0.45F);
  float min_distance = std::numeric_limits<float>::max();
  int rematched = -1;
  for (int index : lit_indices) {
    if (index == exclude_index) continue;
    float distance = cv::norm(rune.fanblades[index].center - last_center);
    if (distance < min_distance) {
      min_distance = distance;
      rematched = index;
    }
  }
  return (min_distance <= rematch_threshold) ? rematched : -1;
}

void DoubleBoardController::update_candidate_tracking(const PowerRune & rune, int exclude_index)
{
  int next_candidate = -1;
  if (state_.candidate_board_index >= 0) {
    next_candidate = rematch_board(rune, last_candidate_center_, exclude_index);
  }

  if (next_candidate < 0) {
    auto lit_indices = rune.getLitBoardIndices();
    lit_indices.erase(
      std::remove(lit_indices.begin(), lit_indices.end(), exclude_index), lit_indices.end());
    next_candidate = pick_best_board(rune, lit_indices);
  }

  state_.candidate_board_index = next_candidate;
  if (next_candidate >= 0 && next_candidate < static_cast<int>(rune.fanblades.size())) {
    last_candidate_center_ = rune.fanblades[next_candidate].center;
  } else {
    last_candidate_center_ = {0.0F, 0.0F};
  }
}

void DoubleBoardController::apply_selection(PowerRune & rune) const
{
  rune.setSelectedBoardIndex(-1);
  rune.setCandidateBoardIndex(-1);
  rune.setSelectedBoardIndex(state_.active_board_index);
  rune.setCandidateBoardIndex(state_.candidate_board_index);
  rune.hit_ring =
    rune.selectedBoardIndex() >= 0 ? rune.getHitRingForBoard(rune.selectedBoardIndex()) : 0;
}

void DoubleBoardController::updateSelection(
  PowerRune & rune, const std::chrono::steady_clock::time_point & timestamp)
{
  (void)timestamp;

  if (state_.active_board_index < 0) {
    if (rune.getLitBoardIndices().empty()) {
      reset_tracking();
    } else {
      start_tracking(rune);
    }
    apply_selection(rune);
    return;
  }

  int rematched_active = rematch_board(rune, last_active_center_, -1);
  if (rematched_active >= 0) {
    state_.active_board_index = rematched_active;
    last_active_center_ = rune.fanblades[rematched_active].center;
    active_board_lost_frames_ = 0;
    update_candidate_tracking(rune, state_.active_board_index);
    apply_selection(rune);
    return;
  }

  active_board_lost_frames_++;
  update_candidate_tracking(rune, -1);

  if (active_board_lost_frames_ >= REMATCH_LOST_FRAMES) {
    if (state_.candidate_board_index >= 0 &&
        state_.candidate_board_index < static_cast<int>(rune.fanblades.size()) &&
        rune.isBoardLit(state_.candidate_board_index))
    {
      state_.active_board_index = state_.candidate_board_index;
      last_active_center_ = rune.fanblades[state_.active_board_index].center;
      active_board_lost_frames_ = 0;
      update_candidate_tracking(rune, state_.active_board_index);
    } else {
      reset_tracking();
    }
  }

  apply_selection(rune);
}

}  // namespace auto_buff
