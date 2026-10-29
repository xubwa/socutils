// Stage timings of the variational loop, for finding where the time of a
// run on several nodes goes. Printed by the root, only when the environment
// variable ZSHCI_PROFILE is set.

#ifndef ZSHCI_SRC_PROFILE_H_
#define ZSHCI_SRC_PROFILE_H_

#include <omp.h>

#include <cstdio>
#include <cstdlib>

namespace zshci {

inline bool ProfileOn() {
  static const bool on = std::getenv("ZSHCI_PROFILE") != nullptr;
  return on;
}

// Mark(name) prints the seconds since the last mark (or construction).
class StageTimer {
 public:
  explicit StageTimer(bool print) : print_(print && ProfileOn()) {}
  void Mark(const char* name) {
    const double now = omp_get_wtime();
    if (print_) std::printf("  profile %-32s %8.2f s\n", name, now - last_);
    last_ = now;
  }

 private:
  bool print_;
  double last_ = omp_get_wtime();
};

}  // namespace zshci

#endif  // ZSHCI_SRC_PROFILE_H_
