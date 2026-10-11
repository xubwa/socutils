#include "integrals.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "highfive/H5File.hpp"

namespace zshci {
namespace {

std::string ToUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

// Returns the integers that follow "KEY=" in a namelist header, up to the next
// key or the end of the header. Empty if the key is absent.
std::vector<int> NamelistInts(const std::string& header,
                              const std::string& key) {
  std::vector<int> values;
  size_t pos = header.find(key + "=");
  if (pos == std::string::npos) return values;
  pos += key.size() + 1;
  while (pos < header.size()) {
    const char c = header[pos];
    if (std::isspace(static_cast<unsigned char>(c)) || c == ',') {
      ++pos;
      continue;
    }
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' ||
          c == '+')) {
      break;
    }
    char* end = nullptr;
    values.push_back(static_cast<int>(std::strtol(&header[pos], &end, 10)));
    pos = end - header.data();
  }
  return values;
}

}  // namespace

template <typename S>
Integrals<S>::Integrals(int num_orbs, std::vector<LabelSet> symmetry,
                        const Comm& comm)
    : num_orbs_(num_orbs), symmetry_(std::move(symmetry)) {
  const size_t n = num_orbs;
  if (n > UINT16_MAX) throw std::length_error("too many spinors");
  for (const LabelSet& set : symmetry_) {
    if (set.labels.size() != n) {
      throw std::invalid_argument(
          "symmetry: labels must have one entry per spinor");
    }
  }
  one_body_.assign(n * n, S(0.0));

  // The block of a pair is given by the sums of its labels, one per set.
  const size_t num_pairs = n * (n - 1) / 2;
  pair_slot_.resize(num_pairs);
  std::map<std::vector<int>, uint32_t> block_of_sums;
  std::vector<uint32_t> block_size;
  std::vector<int> sums(symmetry_.size());
  for (int q = 1; q < num_orbs; ++q) {
    for (int p = 0; p < q; ++p) {
      for (size_t k = 0; k < symmetry_.size(); ++k) {
        const LabelSet& set = symmetry_[k];
        int sum = set.labels[p] + set.labels[q];
        if (set.mod > 0) sum = ((sum % set.mod) + set.mod) % set.mod;
        sums[k] = sum;
      }
      const auto found =
          block_of_sums.emplace(sums, static_cast<uint32_t>(block_size.size()));
      if (found.second) block_size.push_back(0);
      const uint32_t block = found.first->second;
      pair_slot_[PairIndex(p, q)] = {block, block_size[block]++};
    }
  }
  const size_t num_blocks = block_size.size();
  block_offset_.assign(num_blocks + 1, 0);
  block_start_.assign(num_blocks + 1, 0);
  for (size_t b = 0; b < num_blocks; ++b) {
    const size_t size = block_size[b];
    block_offset_[b + 1] = block_offset_[b] + size * (size + 1) / 2;
    block_start_[b + 1] = block_start_[b] + size;
  }
  block_pairs_.resize(num_pairs);
  for (int q = 1; q < num_orbs; ++q) {
    for (int p = 0; p < q; ++p) {
      const PairSlot slot = pair_slot_[PairIndex(p, q)];
      block_pairs_[block_start_[slot.block] + slot.position] = {
          static_cast<uint16_t>(p), static_cast<uint16_t>(q)};
    }
  }

  two_body_ = SharedArray<S>(block_offset_[num_blocks], comm);
  if (comm.is_node_leader()) {
    std::fill(two_body_.data(), two_body_.data() + two_body_.size(), S(0.0));
  }
}

namespace {

// The spin projection (+1 for alpha, -1 for beta) followed by the given
// sets, each label doubled for the two spin orbitals of a spatial orbital.
std::vector<LabelSet> SpinOrbitalSymmetry(
    int num_spatial, const std::vector<LabelSet>& spatial_symmetry) {
  const size_t n = num_spatial;
  std::vector<LabelSet> sets(1);
  for (size_t p = 0; p < n; ++p) {
    sets[0].labels.push_back(1);
    sets[0].labels.push_back(-1);
  }
  for (const LabelSet& spatial : spatial_symmetry) {
    if (spatial.labels.size() != n) {
      throw std::invalid_argument(
          "symmetry: labels must have one entry per spatial orbital");
    }
    LabelSet set;
    set.mod = spatial.mod;
    for (int label : spatial.labels) {
      set.labels.push_back(label);
      set.labels.push_back(label);
    }
    sets.push_back(std::move(set));
  }
  return sets;
}

}  // namespace

template <typename S>
Integrals<S> Integrals<S>::SpinOrbitals(
    int num_spatial, const std::vector<LabelSet>& spatial_symmetry,
    const Comm& comm) {
  Integrals ints(2 * num_spatial,
                 SpinOrbitalSymmetry(num_spatial, spatial_symmetry), comm);
  ints.spin_orbitals_ = true;
  return ints;
}

template <typename S>
Integrals<S> Integrals<S>::Spatial(
    int num_spatial, const std::vector<LabelSet>& spatial_symmetry,
    const Comm& comm) {
  Integrals ints;
  const size_t n = num_spatial;
  if (2 * n > UINT16_MAX) throw std::length_error("too many orbitals");
  ints.num_spatial_ = num_spatial;
  ints.num_orbs_ = 2 * num_spatial;
  ints.spin_orbitals_ = true;
  ints.symmetry_ = SpinOrbitalSymmetry(num_spatial, spatial_symmetry);
  ints.one_body_.assign(4 * n * n, S(0.0));
  const size_t num_products = n * (n + 1) / 2;
  ints.two_body_ = SharedArray<S>(num_products * (num_products + 1) / 2, comm);
  if (comm.is_node_leader()) {
    std::fill(ints.two_body_.data(),
              ints.two_body_.data() + ints.two_body_.size(), S(0.0));
  }
  return ints;
}

template <typename S>
void Integrals<S>::SetSpatialOneBody(int p, int q, S value) {
  SetOneBody(2 * p, 2 * q, value);
  SetOneBody(2 * p + 1, 2 * q + 1, value);
}

template <typename S>
void Integrals<S>::SetSpatialTwoBody(int p, int q, int r, int s, S value) {
  // (pq|rs) moves electrons from q, s to p, r. The first set is the spin.
  for (size_t k = 1; k < symmetry_.size(); ++k) {
    const std::vector<int>& labels = symmetry_[k].labels;
    const int change =
        labels[2 * p] + labels[2 * r] - labels[2 * q] - labels[2 * s];
    if (!LabelConserved(change, symmetry_[k].mod)) {
      NoteForbidden(value);
      return;
    }
  }
  two_body_[SpatialIndex(p, q, r, s)] = value;
}

template <typename S>
void Integrals<S>::NoteForbidden(S value) {
  max_forbidden_ = std::max(max_forbidden_, Abs(value));
}

template <typename S>
S* Integrals<S>::Stored(size_t bra, size_t ket, bool* conjugated) {
  const PairSlot a = pair_slot_[bra];
  const PairSlot b = pair_slot_[ket];
  if (a.block != b.block) return nullptr;
  const size_t i = a.position;
  const size_t j = b.position;
  *conjugated = i < j;
  const size_t hi = i < j ? j : i;
  const size_t lo = i < j ? i : j;
  return &two_body_[block_offset_[a.block] + hi * (hi + 1) / 2 + lo];
}

template <typename S>
void Integrals<S>::SetOneBody(int p, int q, S value) {
  for (const LabelSet& set : symmetry_) {
    if (!LabelConserved(set.labels[p] - set.labels[q], set.mod)) {
      NoteForbidden(value);
      return;
    }
  }
  one_body_[p * num_orbs_ + q] = value;
}

template <typename S>
void Integrals<S>::SetAntisymOrdered(int p, int q, int r, int s, S value) {
  if (is_spatial()) {
    throw std::logic_error("spatial integrals are set by SetSpatialTwoBody");
  }
  bool conjugated = false;
  S* const stored = Stored(PairIndex(p, q), PairIndex(r, s), &conjugated);
  if (stored == nullptr) {
    NoteForbidden(value);
    return;
  }
  *stored = conjugated ? Conj(value) : value;
}

template <typename S>
void Integrals<S>::AddChemist(int p, int q, int r, int s, S value) {
  // (pq|rs) is the first term of <pr||qs>, and the second, with a minus
  // sign, of <pr||sq> = -<pr||qs>: either way it adds to <pr||qs>.
  if (p == r || q == s) return;
  bool negative = false;
  if (p > r) {
    std::swap(p, r);
    negative = !negative;
  }
  if (q > s) {
    std::swap(q, s);
    negative = !negative;
  }
  if (negative) value = -value;
  if (is_spatial()) {
    throw std::logic_error("spatial integrals are set by SetSpatialTwoBody");
  }
  // The Hermitian partner (qp|sr) adds conj(value) to <qs||pr>, which is the
  // same stored number seen from the other side; one of the two is added.
  bool conjugated = false;
  S* const stored = Stored(PairIndex(p, r), PairIndex(q, s), &conjugated);
  if (stored == nullptr) {
    NoteForbidden(value);
    return;
  }
  *stored += conjugated ? Conj(value) : value;
}

template <typename S>
void Integrals<S>::Finalize() {
  const size_t n = num_orbs_;
  pair_energy_.assign(n * n, 0.0);
  for (int p = 0; p < num_orbs_; ++p) {
    for (int q = 0; q < num_orbs_; ++q) {
      pair_energy_[p * n + q] = Real(Antisym(p, q, p, q));
    }
  }
}

template <typename S>
void Integrals<S>::Share(const Comm& comm) {
  comm.BcastValue(&num_elec_);
  comm.BcastValue(&core_energy_);
  comm.BcastValue(&max_forbidden_);
  comm.Bcast(&one_body_);
  two_body_.ShareFromRoot(comm);
  comm.Bcast(&pair_energy_);
}

namespace {

// The header of a FCIDUMP file, and the stream positioned after it.
struct FcidumpHeader {
  int num_orbs = 0;
  int num_elec = 0;
  std::vector<int> labels;
};

FcidumpHeader ReadHeader(const std::string& path, std::ifstream* file) {
  file->open(path);
  if (!*file) throw std::runtime_error("cannot open integral file " + path);

  std::string header;
  std::string line;
  bool header_closed = false;
  while (std::getline(*file, line)) {
    const std::string upper = ToUpper(line);
    const size_t first = upper.find_first_not_of(" \t\r");
    const bool is_slash = first != std::string::npos && upper[first] == '/';
    if (is_slash || upper.find("&END") != std::string::npos) {
      header_closed = true;
      break;
    }
    header += upper + " ";
  }
  if (!header_closed) {
    throw std::runtime_error(path + ": FCIDUMP header is not terminated");
  }

  const std::vector<int> norb = NamelistInts(header, "NORB");
  const std::vector<int> nelec = NamelistInts(header, "NELEC");
  if (norb.size() != 1 || nelec.size() != 1 || norb[0] <= 0) {
    throw std::runtime_error(path + ": cannot read NORB and NELEC");
  }
  FcidumpHeader result;
  result.num_orbs = norb[0];
  result.num_elec = nelec[0];
  result.labels = NamelistInts(header, "ORBSYM");
  if (!result.labels.empty() &&
      static_cast<int>(result.labels.size()) != norb[0]) {
    throw std::runtime_error(path + ": ORBSYM does not have NORB entries");
  }
  return result;
}

// One integral line: the value and the four indices. Returns false for a
// blank line; throws on anything else that does not parse.
struct FcidumpLine {
  double re = 0.0;
  double im = 0.0;
  bool is_complex = false;
  int index[4] = {0, 0, 0, 0};
};

bool ParseLine(std::string line, const std::string& path, size_t line_number,
               FcidumpLine* out) {
  for (char& c : line) {
    if (c == '(' || c == ')' || c == ',') c = ' ';
  }
  std::istringstream fields(line);
  std::vector<std::string> tokens;
  std::string token;
  while (fields >> token) tokens.push_back(token);
  if (tokens.empty()) return false;
  if (tokens.size() != 5 && tokens.size() != 6) {
    throw std::runtime_error(path + ": malformed integral line " +
                             std::to_string(line_number));
  }
  try {
    size_t next = 0;
    out->re = std::stod(tokens[next++]);
    out->is_complex = tokens.size() == 6;
    out->im = out->is_complex ? std::stod(tokens[next++]) : 0.0;
    for (int k = 0; k < 4; ++k) out->index[k] = std::stoi(tokens[next++]);
  } catch (const std::exception&) {
    throw std::runtime_error(path + ": malformed integral line " +
                             std::to_string(line_number));
  }
  return true;
}

}  // namespace

bool FcidumpIsComplex(const std::string& path, const Comm& comm) {
  int is_complex = 0;
  if (comm.is_root()) {
    std::ifstream file;
    ReadHeader(path, &file);
    std::string line;
    size_t line_number = 0;
    FcidumpLine parsed;
    while (std::getline(file, line)) {
      if (ParseLine(line, path, ++line_number, &parsed)) {
        is_complex = parsed.is_complex ? 1 : 0;
        break;
      }
    }
  }
  comm.BcastValue(&is_complex);
  return is_complex != 0;
}

template <typename S>
Integrals<S> ReadFcidump(const std::string& path,
                         const std::vector<LabelSet>& symmetry,
                         double symmetry_tolerance, const Comm& comm) {
  std::ifstream file;
  FcidumpHeader header;
  if (comm.is_root()) header = ReadHeader(path, &file);
  comm.BcastValue(&header.num_orbs);
  comm.Bcast(&header.labels);

  // A set without labels takes ORBSYM, or nothing if the file has none.
  std::vector<LabelSet> sets;
  for (const LabelSet& requested : symmetry) {
    LabelSet set = requested;
    if (set.labels.empty()) set.labels = header.labels;
    if (set.labels.empty()) continue;
    if (static_cast<int>(set.labels.size()) != header.num_orbs) {
      throw std::runtime_error(
          "symmetry: labels must have one entry per spinor");
    }
    sets.push_back(std::move(set));
  }

  Integrals<S> ints(header.num_orbs, std::move(sets), comm);
  if (!comm.is_root()) {
    ints.Share(comm);
    return ints;
  }
  ints.set_num_elec(header.num_elec);

  const int n = header.num_orbs;
  // One bit per integral (pq|rs), (pq|rs) and (rs|pq) sharing one: set when
  // the integral or its Hermitian partner has been used.
  const size_t num_products = static_cast<size_t>(n) * n;
  std::vector<bool> used(num_products * (num_products + 1) / 2, false);
  auto bit_of = [&](int p, int q, int r, int s) {
    const size_t pq = static_cast<size_t>(p) * n + q;
    const size_t rs = static_cast<size_t>(r) * n + s;
    const size_t hi = pq > rs ? pq : rs;
    const size_t lo = pq > rs ? rs : pq;
    return hi * (hi + 1) / 2 + lo;
  };
  std::string line;
  size_t line_number = 0;
  FcidumpLine parsed;
  while (std::getline(file, line)) {
    ++line_number;
    if (!ParseLine(line, path, line_number, &parsed)) continue;
    const int p = parsed.index[0];
    const int q = parsed.index[1];
    const int r = parsed.index[2];
    const int s = parsed.index[3];
    if (p < 0 || q < 0 || r < 0 || s < 0 || p > n || q > n || r > n || s > n) {
      throw std::runtime_error(path + ": orbital index out of range on line " +
                               std::to_string(line_number));
    }
    if (!ScalarTraits<S>::kIsComplex && parsed.im != 0.0) {
      throw std::runtime_error(
          path + ": line " + std::to_string(line_number) +
          " has an imaginary part, which a real calculation cannot hold");
    }
    const S value = ScalarTraits<S>::Make(parsed.re, parsed.im);
    if (p == 0 && q == 0 && r == 0 && s == 0) {
      ints.set_core_energy(parsed.re);
    } else if (q == 0 && r == 0 && s == 0) {
      continue;  // orbital energy
    } else if (r == 0 && s == 0) {
      if (p == 0 || q == 0) {
        throw std::runtime_error(path + ": bad one-body index on line " +
                                 std::to_string(line_number));
      }
      ints.SetOneBody(p - 1, q - 1, value);
      ints.SetOneBody(q - 1, p - 1, Conj(value));
    } else {
      if (p == 0 || q == 0 || r == 0 || s == 0) {
        throw std::runtime_error(path + ": bad two-body index on line " +
                                 std::to_string(line_number));
      }
      const size_t bit = bit_of(p - 1, q - 1, r - 1, s - 1);
      if (used[bit]) continue;
      used[bit] = true;
      used[bit_of(q - 1, p - 1, s - 1, r - 1)] = true;
      ints.AddChemist(p - 1, q - 1, r - 1, s - 1, value);
    }
  }
  if (ints.max_forbidden() > symmetry_tolerance) {
    char magnitude[32];
    std::snprintf(magnitude, sizeof(magnitude), "%.3e", ints.max_forbidden());
    throw std::runtime_error(
        path +
        ": the integrals do not have the symmetry of the labels: an "
        "integral of magnitude " +
        magnitude + " would change a label (see symmetry_tolerance)");
  }
  ints.Finalize();
  ints.Share(comm);
  return ints;
}

template <typename S>
Integrals<S> ReadSpatialFcidump(const std::string& path,
                                const std::vector<LabelSet>& symmetry,
                                double symmetry_tolerance, const Comm& comm) {
  std::ifstream file;
  FcidumpHeader header;
  if (comm.is_root()) header = ReadHeader(path, &file);
  comm.BcastValue(&header.num_orbs);
  comm.Bcast(&header.labels);
  const int n = header.num_orbs;

  std::vector<LabelSet> sets;
  for (const LabelSet& requested : symmetry) {
    if (!requested.labels.empty()) {
      sets.push_back(requested);
      continue;
    }
    // ORBSYM: the bits of (irrep - 1) multiply independently.
    for (int bit = 0; bit < 3; ++bit) {
      LabelSet set;
      set.mod = 2;
      bool any = false;
      for (int irrep : header.labels) {
        if (irrep < 1 || irrep > 8) {
          throw std::runtime_error(
              path +
              ": ORBSYM must hold numbers from 1 to 8 to be used as "
              "the symmetry of spatial orbitals");
        }
        set.labels.push_back(((irrep - 1) >> bit) & 1);
        any = any || set.labels.back() != 0;
      }
      if (any) sets.push_back(std::move(set));
    }
  }

  Integrals<S> ints = Integrals<S>::Spatial(n, sets, comm);
  if (!comm.is_root()) {
    ints.Share(comm);
    return ints;
  }
  ints.set_num_elec(header.num_elec);

  std::string line;
  size_t line_number = 0;
  FcidumpLine parsed;
  while (std::getline(file, line)) {
    ++line_number;
    if (!ParseLine(line, path, line_number, &parsed)) continue;
    const int p = parsed.index[0];
    const int q = parsed.index[1];
    const int r = parsed.index[2];
    const int s = parsed.index[3];
    if (p < 0 || q < 0 || r < 0 || s < 0 || p > n || q > n || r > n || s > n) {
      throw std::runtime_error(path + ": orbital index out of range on line " +
                               std::to_string(line_number));
    }
    if (parsed.im != 0.0) {
      throw std::runtime_error(
          path + ": line " + std::to_string(line_number) +
          " has an imaginary part; integrals over spatial orbitals are real");
    }
    const S value(parsed.re);
    if (p == 0 && q == 0 && r == 0 && s == 0) {
      ints.set_core_energy(parsed.re);
    } else if (q == 0 && r == 0 && s == 0) {
      continue;  // orbital energy
    } else if (r == 0 && s == 0) {
      if (p == 0 || q == 0) {
        throw std::runtime_error(path + ": bad one-body index on line " +
                                 std::to_string(line_number));
      }
      ints.SetSpatialOneBody(p - 1, q - 1, value);
      ints.SetSpatialOneBody(q - 1, p - 1, value);
    } else {
      if (p == 0 || q == 0 || r == 0 || s == 0) {
        throw std::runtime_error(path + ": bad two-body index on line " +
                                 std::to_string(line_number));
      }
      ints.SetSpatialTwoBody(p - 1, q - 1, r - 1, s - 1, value);
    }
  }
  if (ints.max_forbidden() > symmetry_tolerance) {
    char magnitude[32];
    std::snprintf(magnitude, sizeof(magnitude), "%.3e", ints.max_forbidden());
    throw std::runtime_error(
        path +
        ": the integrals do not have the symmetry of the labels: an "
        "integral of magnitude " +
        magnitude + " would change a label (see symmetry_tolerance)");
  }
  ints.Finalize();
  ints.Share(comm);
  return ints;
}

template <typename S>
Integrals<S> ReadUnrestrictedIntegrals(const std::string& path,
                                       const std::vector<LabelSet>& symmetry,
                                       double symmetry_tolerance,
                                       const Comm& comm) {
  int n = 0;
  std::vector<double> h1[2];
  std::vector<double> eri_aa, eri_bb, eri_ab;
  double ecore = 0.0;
  int num_elec = 0;
  if (comm.is_root()) {
    HighFive::File file(path, HighFive::File::ReadOnly);
    auto read = [&](const std::string& name, size_t rank,
                    std::vector<double>* data) {
      HighFive::DataSet dataset = file.getDataSet(name);
      const std::vector<size_t> dims = dataset.getDimensions();
      if (n == 0 && !dims.empty()) n = static_cast<int>(dims[0]);
      size_t size = 1;
      bool good = dims.size() == rank;
      for (size_t d : dims) {
        good = good && d == static_cast<size_t>(n);
        size *= d;
      }
      if (!good) {
        throw std::runtime_error(path + ": dataset " + name +
                                 " does not have the expected shape");
      }
      data->resize(size);
      dataset.read(data->data());
    };
    read("h1_a", 2, &h1[0]);
    read("h1_b", 2, &h1[1]);
    read("eri_aa", 4, &eri_aa);
    read("eri_bb", 4, &eri_bb);
    read("eri_ab", 4, &eri_ab);
    if (file.hasAttribute("ecore")) file.getAttribute("ecore").read(ecore);
    if (file.hasAttribute("nelec")) file.getAttribute("nelec").read(num_elec);
  }
  comm.BcastValue(&n);
  for (const LabelSet& set : symmetry) {
    if (set.labels.empty()) {
      throw std::runtime_error(
          "symmetry: an unrestricted calculation needs explicit labels");
    }
  }

  Integrals<S> ints = Integrals<S>::SpinOrbitals(n, symmetry, comm);
  if (!comm.is_root()) {
    ints.Share(comm);
    return ints;
  }
  ints.set_num_elec(num_elec);
  ints.set_core_energy(ecore);

  const size_t size = n;
  // (pq|rs) over spin orbitals.
  auto chemist = [&](int p, int q, int r, int s) {
    const int spin1 = p & 1;
    const int spin2 = r & 1;
    if ((q & 1) != spin1 || (s & 1) != spin2) return 0.0;
    const size_t a = p >> 1, b = q >> 1, c = r >> 1, d = s >> 1;
    if (spin1 != spin2) {
      return spin1 == 0 ? eri_ab[((a * size + b) * size + c) * size + d]
                        : eri_ab[((c * size + d) * size + a) * size + b];
    }
    return (spin1 == 0 ? eri_aa
                       : eri_bb)[((a * size + b) * size + c) * size + d];
  };
  const int m = 2 * n;
  for (int p = 0; p < m; ++p) {
    for (int q = 0; q < m; ++q) {
      if ((p & 1) != (q & 1)) continue;
      ints.SetOneBody(p, q, S(h1[p & 1][(p >> 1) * size + (q >> 1)]));
    }
  }
  // Every pair of pairs is offered, so that integrals the labels forbid are
  // noticed.
  for (int q = 1; q < m; ++q) {
    for (int p = 0; p < q; ++p) {
      for (int s = 1; s <= q; ++s) {
        for (int r = 0; r < s; ++r) {
          const double value = chemist(p, r, q, s) - chemist(p, s, q, r);
          if (value != 0.0) ints.SetAntisymOrdered(p, q, r, s, S(value));
        }
      }
    }
  }
  if (ints.max_forbidden() > symmetry_tolerance) {
    char magnitude[32];
    std::snprintf(magnitude, sizeof(magnitude), "%.3e", ints.max_forbidden());
    throw std::runtime_error(
        path +
        ": the integrals do not have the symmetry of the labels: an "
        "integral of magnitude " +
        magnitude + " would change a label (see symmetry_tolerance)");
  }
  ints.Finalize();
  ints.Share(comm);
  return ints;
}

template class Integrals<double>;
template class Integrals<Complex>;
template Integrals<double> ReadFcidump<double>(const std::string&,
                                               const std::vector<LabelSet>&,
                                               double, const Comm&);
template Integrals<Complex> ReadFcidump<Complex>(const std::string&,
                                                 const std::vector<LabelSet>&,
                                                 double, const Comm&);

template Integrals<double> ReadSpatialFcidump<double>(
    const std::string&, const std::vector<LabelSet>&, double, const Comm&);
template Integrals<Complex> ReadSpatialFcidump<Complex>(
    const std::string&, const std::vector<LabelSet>&, double, const Comm&);

template Integrals<double> ReadUnrestrictedIntegrals<double>(
    const std::string&, const std::vector<LabelSet>&, double, const Comm&);
template Integrals<Complex> ReadUnrestrictedIntegrals<Complex>(
    const std::string&, const std::vector<LabelSet>&, double, const Comm&);

}  // namespace zshci
