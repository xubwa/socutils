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

std::string RdmPath(const std::string& prefix, int bra, int ket, int order) {
  return prefix + "/Dice_" + std::to_string(bra) + "_" + std::to_string(ket) +
         ".rdm" + std::to_string(order) + ".h5";
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

void WriteOneRdm(const std::string& prefix, int bra, int ket,
                 const DensityMatrices& dm) {
  const int n = dm.num_orbs;
  std::vector<double> re(static_cast<size_t>(n) * n);
  std::vector<double> im(re.size());
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      const Complex value = dm.One(p, q);
      re[q * n + p] = value.real();
      im[q * n + p] = value.imag();
    }
  }
  HighFive::File file(RdmPath(prefix, bra, ket, 1), HighFive::File::Overwrite);
  WriteVector(&file, "rdm1_real", re);
  WriteVector(&file, "rdm1_imag", im);
}

void WriteTwoRdm(const std::string& prefix, int root,
                 const DensityMatrices& dm) {
  std::vector<double> re(dm.two.size());
  std::vector<double> im(dm.two.size());
  const long size = static_cast<long>(dm.two.size());
#pragma omp parallel for schedule(static)
  for (long k = 0; k < size; ++k) {
    re[k] = dm.two[k].real();
    im[k] = -dm.two[k].imag();
  }
  HighFive::File file(RdmPath(prefix, root, root, 2),
                      HighFive::File::Overwrite);
  WriteVector(&file, "rdm2_real", re);
  WriteVector(&file, "rdm2_imag", im);
}

void WriteWaveFunction(const std::string& path, const WaveFunction& wfn) {
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
      re[k * num_dets + i] = wfn.coefficients[k][i].real();
      im[k * num_dets + i] = wfn.coefficients[k][i].imag();
    }
  }
  file.createDataSet<double>("ci_real",
                             HighFive::DataSpace({num_roots, num_dets}))
      .write_raw(re.data());
  file.createDataSet<double>("ci_imag",
                             HighFive::DataSpace({num_roots, num_dets}))
      .write_raw(im.data());

  WriteVector(&file, "energies", wfn.energies);
  if (!wfn.pt_corrections.empty()) {
    WriteVector(&file, "pt_corrections", wfn.pt_corrections);
    WriteVector(&file, "pt_errors", wfn.pt_errors);
  }
  file.createDataSet<int>("labels", HighFive::DataSpace(wfn.labels.size()))
      .write_raw(wfn.labels.data());
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

}  // namespace zshci
