#include "integrals.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

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
Integrals<S>::Integrals(int num_orbs, const Comm& comm) : num_orbs_(num_orbs) {
  const size_t n = num_orbs;
  const size_t num_pairs = n * n;
  one_body_.assign(num_pairs, S(0.0));
  two_body_ = SharedArray<S>(num_pairs * (num_pairs + 1) / 2, comm);
  if (comm.is_node_leader()) {
    std::fill(two_body_.data(), two_body_.data() + two_body_.size(), S(0.0));
  }
  labels_.assign(n, 0);
}

template <typename S>
void Integrals<S>::Finalize() {
  const size_t n = num_orbs_;
  direct_.assign(n * n, 0.0);
  exchange_.assign(n * n, 0.0);
  for (int p = 0; p < num_orbs_; ++p) {
    for (int q = 0; q < num_orbs_; ++q) {
      direct_[p * n + q] = Real(TwoBody(p, p, q, q));
      exchange_[p * n + q] = Real(TwoBody(p, q, q, p));
    }
  }
}

template <typename S>
void Integrals<S>::Share(const Comm& comm) {
  comm.BcastValue(&num_elec_);
  comm.BcastValue(&core_energy_);
  comm.Bcast(&labels_);
  comm.Bcast(&one_body_);
  two_body_.ShareFromRoot(comm);
  comm.Bcast(&direct_);
  comm.Bcast(&exchange_);
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
Integrals<S> ReadFcidump(const std::string& path, const Comm& comm) {
  std::ifstream file;
  FcidumpHeader header;
  if (comm.is_root()) header = ReadHeader(path, &file);
  comm.BcastValue(&header.num_orbs);

  Integrals<S> ints(header.num_orbs, comm);
  if (!comm.is_root()) {
    ints.Share(comm);
    return ints;
  }
  ints.set_num_elec(header.num_elec);
  if (!header.labels.empty()) ints.set_labels(std::move(header.labels));

  const int n = header.num_orbs;
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
      ints.SetTwoBody(p - 1, q - 1, r - 1, s - 1, value);
    }
  }
  ints.Finalize();
  ints.Share(comm);
  return ints;
}

template class Integrals<double>;
template class Integrals<Complex>;
template Integrals<double> ReadFcidump<double>(const std::string&, const Comm&);
template Integrals<Complex> ReadFcidump<Complex>(const std::string&,
                                                 const Comm&);

}  // namespace zshci
