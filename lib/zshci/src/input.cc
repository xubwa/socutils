#include "input.h"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace zshci {
namespace {

using Json = nlohmann::json;

// Reads typed values out of one JSON object and records which keys were
// asked for, so that leftover (misspelled or unsupported) keys are reported.
class ObjectReader {
 public:
  ObjectReader(const Json& object, std::string path)
      : object_(object), path_(std::move(path)) {
    if (!object_.is_object()) Fail("", "must be a JSON object");
  }

  bool Has(const std::string& key) {
    known_.insert(key);
    return object_.contains(key);
  }

  // Leaves *value untouched when the key is absent.
  template <class T>
  void Get(const std::string& key, T* value) {
    if (!Has(key)) return;
    try {
      *value = object_.at(key).get<T>();
    } catch (const Json::exception&) {
      Fail(key, "has the wrong type");
    }
  }

  const Json& At(const std::string& key) {
    known_.insert(key);
    return object_.at(key);
  }

  // Marks a key as accepted without interpreting it.
  void Ignore(const std::string& key) { known_.insert(key); }

  void CheckNoUnknownKeys() const {
    for (auto it = object_.begin(); it != object_.end(); ++it) {
      if (known_.count(it.key()) == 0) Fail(it.key(), "is not a known key");
    }
  }

  [[noreturn]] void Fail(const std::string& key,
                         const std::string& what) const {
    std::string name = path_;
    if (!key.empty()) name += (name.empty() ? "" : ".") + key;
    throw std::runtime_error("input: \"" + name + "\" " + what);
  }

 private:
  const Json& object_;
  std::string path_;
  std::set<std::string> known_;
};

Json ToJson(const Input& in) {
  Json j;
  j["mode"] = in.mode == Mode::kDiag ? "diag" : "select";
  j["integrals"] = in.integrals;
  j["symmetry_tolerance"] = in.symmetry_tolerance;
  j["orbitals"] = in.spatial_orbitals        ? "spatial"
                  : in.unrestricted_orbitals ? "unrestricted"
                                             : "spinor";
  j["scalar"] = in.scalar == ScalarChoice::kAuto   ? "auto"
                : in.scalar == ScalarChoice::kReal ? "real"
                                                   : "complex";
  j["prefix"] = in.prefix;
  j["nroots"] = in.nroots;
  if (!in.detfile.empty()) j["detfile"] = in.detfile;
  if (!in.dets.empty()) j["dets"] = in.dets;
  j["symmetry"] = Json::array();
  for (const LabelSet& set : in.symmetry) {
    Json item;
    if (set.labels.empty()) {
      item["labels"] = "fcidump";
    } else {
      item["labels"] = set.labels;
    }
    item["mod"] = set.mod;
    j["symmetry"].push_back(item);
  }
  j["restrictions"] = Json::array();
  for (const RestrictionInput& restriction : in.restrictions) {
    j["restrictions"].push_back({{"orbitals", restriction.orbitals},
                                 {"min", restriction.min_elec},
                                 {"max", restriction.max_elec}});
  }
  if (in.mode == Mode::kSelect) {
    j["select"] = {{"schedule", in.select.schedule},
                   {"max_iter", in.select.max_iter},
                   {"energy_tol", in.select.energy_tol},
                   {"davidson_tol_loose", in.select.davidson_tol_loose},
                   {"integral_threshold", in.select.integral_threshold}};
  }
  j["davidson"] = {{"tol", in.davidson.tol},
                   {"max_space_per_root", in.davidson.max_space_per_root},
                   {"max_iterations", in.davidson.max_iterations},
                   {"dense_dim", in.davidson.dense_dim}};
  static const char* const kPtNames[] = {"none", "deterministic",
                                         "stochastic", "semistochastic"};
  j["pt"] = {{"method", kPtNames[static_cast<int>(in.pt.method)]},
             {"epsilon2", in.pt.epsilon2},
             {"epsilon2_large", in.pt.epsilon2_large},
             {"num_samples", in.pt.num_samples},
             {"min_iterations", in.pt.min_iterations},
             {"max_iterations", in.pt.max_iterations},
             {"target_error", in.pt.target_error},
             {"seed", in.pt.seed},
             {"num_batches", in.pt.num_batches}};
  j["rdm"] = {{"one", in.rdm.one},
              {"two", in.rdm.two},
              {"transition", in.rdm.transition}};
  j["hij_threshold"] = in.hij_threshold;
  j["connect"] = {{"method", in.connect_by_groups     ? "groups"
                             : in.connect_by_segments ? "segments"
                                                      : "strings"},
                  {"segments", in.connect_segments},
                  {"batches", in.connect_batches}};
  if (!in.connect_group_a.empty()) j["connect"]["group_a"] = in.connect_group_a;
  j["hamiltonian"] = in.direct_hamiltonian ? "direct" : "stored";
  j["check"] = in.check;
  j["output_level"] = in.output_level;
  return j;
}

}  // namespace

Input ParseInput(const std::string& text) {
  Json root;
  try {
    root = Json::parse(text, /*cb=*/nullptr, /*allow_exceptions=*/true,
                       /*ignore_comments=*/true);
  } catch (const Json::exception& e) {
    throw std::runtime_error(std::string("input: not valid JSON: ") +
                             e.what());
  }

  Input in;
  ObjectReader top(root, "");

  std::string mode = "select";
  top.Get("mode", &mode);
  if (mode == "select") {
    in.mode = Mode::kSelect;
  } else if (mode == "diag") {
    in.mode = Mode::kDiag;
  } else {
    top.Fail("mode", "must be \"select\" or \"diag\"");
  }

  top.Get("integrals", &in.integrals);
  std::string scalar = "auto";
  top.Get("scalar", &scalar);
  if (scalar == "auto") {
    in.scalar = ScalarChoice::kAuto;
  } else if (scalar == "real") {
    in.scalar = ScalarChoice::kReal;
  } else if (scalar == "complex") {
    in.scalar = ScalarChoice::kComplex;
  } else {
    top.Fail("scalar", "must be \"auto\", \"real\" or \"complex\"");
  }
  top.Get("prefix", &in.prefix);
  top.Get("nroots", &in.nroots);
  if (in.nroots < 1) top.Fail("nroots", "must be at least 1");

  top.Get("dets", &in.dets);
  top.Get("detfile", &in.detfile);
  if (in.dets.empty() == in.detfile.empty()) {
    top.Fail("dets", "exactly one of \"dets\" and \"detfile\" must be given");
  }

  if (top.Has("symmetry")) {
    const Json& symmetry = top.At("symmetry");
    const bool is_list = symmetry.is_array();
    in.symmetry.clear();
    for (size_t k = 0; k < (is_list ? symmetry.size() : 1); ++k) {
      ObjectReader sym(
          is_list ? symmetry[k] : symmetry,
          is_list ? "symmetry[" + std::to_string(k) + "]" : "symmetry");
      LabelSet set;
      if (sym.Has("labels")) {
        const Json& labels = sym.At("labels");
        if (labels.is_string()) {
          if (labels.get<std::string>() != "fcidump") {
            sym.Fail("labels", "must be \"fcidump\" or an array of integers");
          }
        } else {
          sym.Get("labels", &set.labels);
        }
      }
      sym.Get("mod", &set.mod);
      if (set.mod < 0) sym.Fail("mod", "must not be negative");
      sym.CheckNoUnknownKeys();
      in.symmetry.push_back(set);
    }
  }

  std::string orbitals = "spinor";
  top.Get("orbitals", &orbitals);
  if (orbitals != "spinor" && orbitals != "spatial" &&
      orbitals != "unrestricted") {
    top.Fail("orbitals", "must be \"spinor\", \"spatial\" or \"unrestricted\"");
  }
  in.spatial_orbitals = orbitals == "spatial";
  in.unrestricted_orbitals = orbitals == "unrestricted";
  const bool spin_orbitals = in.spatial_orbitals || in.unrestricted_orbitals;
  if (spin_orbitals && in.scalar == ScalarChoice::kComplex) {
    top.Fail("scalar", "these integrals are real");
  }
  // ORBSYM is a matter of FCIDUMP files.
  if (in.unrestricted_orbitals && !top.Has("symmetry")) in.symmetry.clear();
  top.Get("symmetry_tolerance", &in.symmetry_tolerance);
  if (in.symmetry_tolerance < 0.0) {
    top.Fail("symmetry_tolerance", "must not be negative");
  }

  if (top.Has("restrictions")) {
    const Json& list = top.At("restrictions");
    if (!list.is_array()) top.Fail("restrictions", "must be an array");
    for (size_t r = 0; r < list.size(); ++r) {
      ObjectReader item(list[r], "restrictions[" + std::to_string(r) + "]");
      RestrictionInput restriction;
      item.Get("orbitals", &restriction.orbitals);
      item.Get("min", &restriction.min_elec);
      restriction.max_elec = static_cast<int>(restriction.orbitals.size());
      item.Get("max", &restriction.max_elec);
      item.CheckNoUnknownKeys();
      if (restriction.orbitals.empty()) {
        item.Fail("orbitals", "must list at least one spinor");
      }
      if (restriction.min_elec > restriction.max_elec) {
        item.Fail("min", "must not exceed \"max\"");
      }
      in.restrictions.push_back(restriction);
    }
  }

  if (in.mode == Mode::kSelect) {
    if (!top.Has("select")) top.Fail("select", "is required in select mode");
    ObjectReader sel(top.At("select"), "select");
    sel.Get("schedule", &in.select.schedule);
    sel.Get("max_iter", &in.select.max_iter);
    sel.Get("energy_tol", &in.select.energy_tol);
    sel.Get("davidson_tol_loose", &in.select.davidson_tol_loose);
    sel.Get("integral_threshold", &in.select.integral_threshold);
    sel.CheckNoUnknownKeys();

    if (in.select.schedule.empty() || in.select.schedule[0].first != 0) {
      sel.Fail("schedule", "must be [[0, epsilon1], [iteration, epsilon1], "
                           "...]");
    }
    int previous = -1;
    for (const std::pair<int, double>& entry : in.select.schedule) {
      if (entry.first <= previous || entry.second <= 0.0) {
        sel.Fail("schedule", "needs increasing iterations and positive "
                             "epsilon1");
      }
      previous = entry.first;
    }
    if (in.select.max_iter == 0) in.select.max_iter = previous + 6;
    if (in.select.max_iter <= previous) {
      sel.Fail("max_iter", "must exceed the last iteration of the schedule");
    }
  }

  if (top.Has("davidson")) {
    ObjectReader dav(top.At("davidson"), "davidson");
    dav.Get("tol", &in.davidson.tol);
    dav.Get("max_space_per_root", &in.davidson.max_space_per_root);
    dav.Get("max_iterations", &in.davidson.max_iterations);
    dav.Get("dense_dim", &in.davidson.dense_dim);
    if (in.davidson.tol <= 0.0) dav.Fail("tol", "must be positive");
    if (in.davidson.max_space_per_root < 2) {
      dav.Fail("max_space_per_root", "must be at least 2");
    }
    dav.CheckNoUnknownKeys();
  }

  if (top.Has("pt")) {
    ObjectReader pt(top.At("pt"), "pt");
    std::string method = "none";
    pt.Get("method", &method);
    if (method == "none") {
      in.pt.method = PtMethod::kNone;
    } else if (method == "deterministic") {
      in.pt.method = PtMethod::kDeterministic;
    } else if (method == "stochastic") {
      in.pt.method = PtMethod::kStochastic;
    } else if (method == "semistochastic") {
      in.pt.method = PtMethod::kSemistochastic;
    } else {
      pt.Fail("method", "must be \"none\", \"deterministic\", "
                        "\"stochastic\" or \"semistochastic\"");
    }
    pt.Get("epsilon2", &in.pt.epsilon2);
    pt.Get("epsilon2_large", &in.pt.epsilon2_large);
    pt.Get("num_samples", &in.pt.num_samples);
    pt.Get("min_iterations", &in.pt.min_iterations);
    pt.Get("max_iterations", &in.pt.max_iterations);
    pt.Get("target_error", &in.pt.target_error);
    pt.Get("seed", &in.pt.seed);
    pt.Get("num_batches", &in.pt.num_batches);
    pt.CheckNoUnknownKeys();
    if (in.pt.epsilon2 <= 0.0) pt.Fail("epsilon2", "must be positive");
    if (in.pt.method == PtMethod::kSemistochastic &&
        in.pt.epsilon2_large <= in.pt.epsilon2) {
      pt.Fail("epsilon2_large",
              "is required for the semistochastic method and must exceed "
              "epsilon2");
    }
    if (in.pt.num_samples < 2) pt.Fail("num_samples", "must be at least 2");
    if (in.pt.min_iterations < 2) {
      pt.Fail("min_iterations", "must be at least 2");
    }
    if (in.pt.max_iterations < in.pt.min_iterations) {
      pt.Fail("max_iterations", "must not be below min_iterations");
    }
    if (in.pt.num_batches < 1) pt.Fail("num_batches", "must be at least 1");
  }

  if (top.Has("rdm")) {
    ObjectReader rdm(top.At("rdm"), "rdm");
    rdm.Get("one", &in.rdm.one);
    rdm.Get("two", &in.rdm.two);
    rdm.Get("transition", &in.rdm.transition);
    rdm.CheckNoUnknownKeys();
  }

  const bool any_rdm = in.rdm.one || in.rdm.two || in.rdm.transition;
  in.hij_threshold = any_rdm ? -1.0 : 0.0;
  // An explicit non-negative threshold together with density matrices is
  // allowed, but the density matrices then miss the dropped pairs.
  top.Get("hij_threshold", &in.hij_threshold);
  in.connect_by_groups = spin_orbitals;
  if (top.Has("connect")) {
    ObjectReader connect(top.At("connect"), "connect");
    std::string method = spin_orbitals ? "groups" : "strings";
    connect.Get("method", &method);
    if (method != "strings" && method != "segments" && method != "groups") {
      connect.Fail("method", "must be \"strings\", \"segments\" or \"groups\"");
    }
    in.connect_by_segments = method == "segments";
    in.connect_by_groups = method == "groups";
    connect.Get("group_a", &in.connect_group_a);
    connect.Get("segments", &in.connect_segments);
    connect.Get("batches", &in.connect_batches);
    connect.CheckNoUnknownKeys();
    if (in.connect_segments < 1 || in.connect_segments > kMaxConnectSegments) {
      connect.Fail("segments", "must be between 1 and " +
                                   std::to_string(kMaxConnectSegments));
    }
    if (in.connect_batches < 1) connect.Fail("batches", "must be at least 1");
  }
  std::string hamiltonian = "stored";
  top.Get("hamiltonian", &hamiltonian);
  if (hamiltonian != "stored" && hamiltonian != "direct") {
    top.Fail("hamiltonian", "must be \"stored\" or \"direct\"");
  }
  in.direct_hamiltonian = hamiltonian == "direct";
  if (in.direct_hamiltonian) {
    if (in.connect_by_segments ||
        (top.Has("connect") && top.At("connect").contains("method") &&
         !in.connect_by_groups)) {
      top.Fail("hamiltonian",
               "\"direct\" works with the connection method "
               "\"groups\" only");
    }
    in.connect_by_groups = true;
  }
  top.Get("check", &in.check);
  top.Get("output_level", &in.output_level);

  // The select block is not used in diag mode but accepted, so that one file
  // can serve both modes.
  if (in.mode == Mode::kDiag) top.Ignore("select");
  top.CheckNoUnknownKeys();

  in.davidson.print = in.output_level > 1;
  in.resolved = ToJson(in).dump(2);
  return in;
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open input file " + path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace zshci
