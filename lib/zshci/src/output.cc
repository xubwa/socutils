#include "output.h"

#include <cstdio>
#include <stdexcept>

#include "highfive/H5File.hpp"

namespace zshci {
namespace {

void WriteVector(HighFive::File* file, const std::string& name,
                 const std::vector<double>& data) {
  HighFive::DataSet dataset =
      file->createDataSet<double>(name, HighFive::DataSpace(data.size()));
  dataset.write_raw(data.data());
}

}  // namespace

void WriteEnergies(const std::string& prefix,
                   const std::vector<double>& energies) {
  const std::string path = prefix + "/shci.e";
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) throw std::runtime_error("cannot write " + path);
  const size_t written =
      std::fwrite(energies.data(), sizeof(double), energies.size(), file);
  const int closed = std::fclose(file);
  if (written != energies.size() || closed != 0) {
    throw std::runtime_error("error while writing " + path);
  }
}

template <typename S>
void WriteDensityMatrices(const std::string& prefix, int bra, int ket,
                          const DensityMatrices<S>& dm) {
  const size_t n = dm.num_orbs;
  HighFive::File file(prefix + "/rdm_" + std::to_string(bra) + "_" +
                          std::to_string(ket) + ".h5",
                      HighFive::File::Overwrite);
  // dm.one is already dm1[p,q] = <p^+ q> in row-major order.
  file.createDataSet<S>("dm1", HighFive::DataSpace({n, n}))
      .write_raw(dm.one.data());
  if (dm.two.empty()) return;

  // Stored: D2(p,q,r,s) = <p^+ q^+ s r>. Written: dm2[p,q,r,s] =
  // <p^+ r^+ s q> = D2(p,r,q,s).
  std::vector<S> dm2(dm.two.size());
  const long num = static_cast<long>(n);
#pragma omp parallel for schedule(static)
  for (long p = 0; p < num; ++p) {
    for (size_t q = 0; q < n; ++q) {
      for (size_t r = 0; r < n; ++r) {
        for (size_t s = 0; s < n; ++s) {
          dm2[((p * n + q) * n + r) * n + s] = dm.Two(p, r, q, s);
        }
      }
    }
  }
  file.createDataSet<S>("dm2", HighFive::DataSpace({n, n, n, n}))
      .write_raw(dm2.data());
}

template <typename S>
void WriteWaveFunction(const std::string& path, const WaveFunction<S>& wfn) {
  HighFive::File file(path, HighFive::File::Overwrite);
  const size_t width = wfn.det_width;
  const size_t num_dets = width == 0 ? 0 : wfn.det_words.size() / width;

  HighFive::DataSet dets = file.createDataSet<uint64_t>(
      "dets", HighFive::DataSpace({num_dets, width}));
  dets.write_raw(wfn.det_words.data());

  const size_t num_roots = wfn.coefficients.size();
  std::vector<double> re(num_roots * num_dets);
  std::vector<double> im(num_roots * num_dets);
  for (size_t k = 0; k < num_roots; ++k) {
    for (size_t i = 0; i < num_dets; ++i) {
      re[k * num_dets + i] = Real(wfn.coefficients[k][i]);
      im[k * num_dets + i] = Imag(wfn.coefficients[k][i]);
    }
  }
  file.createDataSet<double>("ci_real",
                             HighFive::DataSpace({num_roots, num_dets}))
      .write_raw(re.data());
  if (ScalarTraits<S>::kIsComplex) {
    file.createDataSet<double>("ci_imag",
                               HighFive::DataSpace({num_roots, num_dets}))
        .write_raw(im.data());
  }

  WriteVector(&file, "energies", wfn.energies);
  if (!wfn.pt_corrections.empty()) {
    WriteVector(&file, "pt_corrections", wfn.pt_corrections);
    WriteVector(&file, "pt_errors", wfn.pt_errors);
  }
  const size_t num_sets = wfn.label_mods.size();
  file.createDataSet<int>(
          "labels",
          HighFive::DataSpace({num_sets, static_cast<size_t>(wfn.num_orbs)}))
      .write_raw(wfn.labels.data());
  file.createDataSet<int>("label_mods", HighFive::DataSpace(num_sets))
      .write_raw(wfn.label_mods.data());
  file.createAttribute<int>("num_orbs", HighFive::DataSpace::From(wfn.num_orbs))
      .write(wfn.num_orbs);
  file.createAttribute<int>("num_elec", HighFive::DataSpace::From(wfn.num_elec))
      .write(wfn.num_elec);
  file.createAttribute<std::string>("input",
                                    HighFive::DataSpace::From(wfn.input))
      .write(wfn.input);
}

std::vector<uint64_t> ReadDetWords(const std::string& path, int* det_width) {
  HighFive::File file(path, HighFive::File::ReadOnly);
  HighFive::DataSet dataset = file.getDataSet("dets");
  const std::vector<size_t> dims = dataset.getDimensions();
  if (dims.size() != 2) {
    throw std::runtime_error(path + ": dataset \"dets\" must be N x W");
  }
  *det_width = static_cast<int>(dims[1]);
  std::vector<uint64_t> words(dims[0] * dims[1]);
  dataset.read(words.data());
  return words;
}

template void WriteDensityMatrices(const std::string&, int, int,
                                   const DensityMatrices<double>&);
template void WriteDensityMatrices(const std::string&, int, int,
                                   const DensityMatrices<Complex>&);
template void WriteWaveFunction(const std::string&,
                                const WaveFunction<double>&);
template void WriteWaveFunction(const std::string&,
                                const WaveFunction<Complex>&);

}  // namespace zshci
