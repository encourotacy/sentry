// buff_type.hpp
#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <chrono>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"

namespace auto_buff
{
const int INF = 1000000;

struct SinusoidalParam {
  double a;
  double omega;
  double b;
  double phi;

  double getSpeed(double t) const { return a * sin(omega * t + phi) + b; }

  double getAngle(double start_angle, double t) const
  {
    return start_angle + (a / omega) * (cos(phi) - cos(omega * t + phi)) + b * t;
  }
};

enum class EnergyType {
  SMALL,
  BIG
};

struct DoubleBoardSelectionState {
  int active_board_index = -1;
  int candidate_board_index = -1;

  void reset()
  {
    active_board_index = -1;
    candidate_board_index = -1;
  }
};

enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE, PREDICT_TRACK };

class FanBlade
{
public:
  cv::Point2f center;
  std::vector<cv::Point2f> points;
  double angle, width, height;
  FanBlade_type type;

  explicit FanBlade() = default;
  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);
  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;
  int light_num;

  Eigen::Vector3d xyz_in_world;
  Eigen::Vector3d ypr_in_world;
  Eigen::Vector3d ypd_in_world;
  Eigen::Vector3d blade_xyz_in_world;
  Eigen::Vector3d blade_ypd_in_world;

  int hit_ring;
  int activated_arms;
  EnergyType energy_type;
  double timestamp;
  double distance_to_target;
  int selected_board_index;
  int candidate_board_index;
  std::vector<int> board_hit_rings;
  std::vector<double> board_confidences;
  std::vector<int> board_tracked_frames;
  std::vector<double> board_areas;

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[resolved_selected_index()]; };
  const FanBlade & target() const { return fanblades[resolved_selected_index()]; };

  bool is_unsolve() const { return unsolvable_; }

  int getLightArmCount() const;
  int getCurrentTargetArmIndex() const;
  double getFanBladeAngle(int index) const;
  std::vector<int> getLitBoardIndices() const;
  bool isBoardLit(int index) const;
  void setBoardHitRing(int index, int ring);
  int getHitRingForBoard(int index) const;
  void setSelectedBoardIndex(int index);
  void setCandidateBoardIndex(int index);
  int selectedBoardIndex() const { return selected_board_index; }
  int candidateBoardIndex() const { return candidate_board_index; }

private:
  bool unsolvable_ = false;
  double atan_angle(cv::Point2f v) const;
  int resolved_selected_index() const;
};

class DoubleBoardController
{
public:
  void updateSelection(PowerRune & rune, const std::chrono::steady_clock::time_point & timestamp);
  const DoubleBoardSelectionState & state() const { return state_; }

private:
  static constexpr int REMATCH_LOST_FRAMES = 2;
  static constexpr int MAX_TRACK_FRAMES = 100;

  DoubleBoardSelectionState state_;
  int active_board_lost_frames_ = 0;
  cv::Point2f last_active_center_ = {0.0F, 0.0F};
  cv::Point2f last_candidate_center_ = {0.0F, 0.0F};

  void reset_tracking();
  void start_tracking(const PowerRune & rune);
  void update_candidate_tracking(const PowerRune & rune, int exclude_index);
  int pick_best_board(const PowerRune & rune, const std::vector<int> & candidates) const;
  int rematch_board(
    const PowerRune & rune, const cv::Point2f & last_center, int exclude_index) const;
  void apply_selection(PowerRune & rune) const;
};

}  // namespace auto_buff
#endif  // BUFF__TYPE_HPP
