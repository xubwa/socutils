import numpy
import re
from pyscf import gto
from scipy.linalg import block_diag
from scipy.io import FortranFile


class Labels:

    @staticmethod
    def pyscf_sph_labels(mol: gto.Mole):
        """
        Return list of AO labels in Pyscf order.
        """
        labels = mol.ao_labels()
        labels = [label.split() for label in labels]
        labels = [label[:-1] + Labels._pyscf_sph_split(label[-1]) for label in labels]
        labels = ["".join(label) for label in labels]
        return labels

    _sph_re = re.compile(r"^(\d+)([a-z])(.*)$")

    _sph_map = {
        "": "0",
        "x": "1",
        "y": "-1",
        "z": "0",
        "xy": "-2",
        "yz": "-1",
        "z^2": "0",
        "xz": "1",
        "x2-y2": "2",
    }

    @staticmethod
    def _pyscf_sph_split(label: str):
        match = Labels._sph_re.match(label)
        if not match:
            raise ValueError(f"Label {label} does not match the expected format.")
        group = list(match.groups())

        group[2] = Labels._sph_map.get(group[2], group[2])
        return group

    @staticmethod
    def cfour_sph_labels(mol: gto.Mole):
        """
        Return list of AO labels in Cfour order.
        """
        labels = mol.ao_labels()
        labels = [label.split() for label in labels]
        labels = [label[:-1] + Labels._pyscf_sph_split(label[-1]) for label in labels]
        labels = Labels._cfour_sph_rules(labels)
        labels = ["".join(label) for label in labels]
        return labels

    _l_order = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
        "h": 5,
        "i": 6,
        "k": 7,
        "l": 8,
    }

    @staticmethod
    def _cfour_sph_rules(label: list):

        def m_sort_key(m_str: str):
            m = int(m_str)
            if m == 0:
                return (float("inf"), 0)
            return (-abs(m), 0 if m > 0 else 1)

        def sort_key(row: list):
            atm = int(row[0])
            l = row[3]
            m = row[4]
            n = int(row[2])
            return (atm, Labels._l_order.get(l, 99), m_sort_key(m), n)

        return sorted(label, key=sort_key)

    @staticmethod
    def sph_cfour2pyscf(mol: gto.Mole):
        cfour_labels = Labels.cfour_sph_labels(mol)
        pyscf_labels = Labels.pyscf_sph_labels(mol)
        return Labels.gen_mat(cfour_labels, pyscf_labels)

    @staticmethod
    def sph_pyscf2cfour(mol: gto.Mole):
        pyscf_labels = Labels.pyscf_sph_labels(mol)
        cfour_labels = Labels.cfour_sph_labels(mol)
        return Labels.gen_mat(pyscf_labels, cfour_labels)

    @staticmethod
    def sph2c_pyscf2cfour(mol: gto.Mole):
        mat = Labels.sph_pyscf2cfour(mol)
        mat = block_diag(mat, mat)
        return mat

    @staticmethod
    def sph2c_cfour2pyscf(mol: gto.Mole):
        mat = Labels.sph_cfour2pyscf(mol)
        mat = block_diag(mat, mat)
        return mat

    @staticmethod
    def spinor_cfour2pyscf(mol: gto.Mole):
        mat = Labels.sph_cfour2pyscf(mol)
        mat = block_diag(Labels.sph_cfour2pyscf(mol), Labels.sph_cfour2pyscf(mol))
        mat = mat @ numpy.vstack(mol.sph2spinor_coeff())
        return mat

    @staticmethod
    def spinor_pyscf2cfour(mol: gto.Mole):
        mat = mol.sph2spinor_coeff()
        mat = numpy.vstack(mat)
        mat = mat.T.conj() @ block_diag(
            Labels.sph_pyscf2cfour(mol), Labels.sph_pyscf2cfour(mol)
        )
        return mat

    @staticmethod
    def gen_mat(source_order, target_order):
        """
        Construct a permutation matrix P such that:
            A_target = P.T @ A_source @ P
        """
        assert len(source_order) == len(
            target_order
        ), "Source and target orders must have the same length."
        n = len(source_order)
        P = numpy.zeros((n, n))
        for i, ao_label in enumerate(target_order):
            try:
                j = source_order.index(ao_label)
            except ValueError:
                raise ValueError(f"Label {ao_label} not found in source_order.")
            P[j, i] = 1
        return P


def carr_py2f(mol: gto.Mole, mat: numpy.ndarray, filename: str = "fortranFile"):
    mat = mat.T
    mat = mat.reshape(-1)
    mat = numpy.vstack((mat.real, mat.imag)).T.flatten()
    with FortranFile(filename, "w") as f:
        f.write_record(mat)


def carr_f2py(mol: gto.Mole, filename: str):
    fortran = FortranFile(filename, "r")
    arr = fortran.read_reals(dtype=numpy.float64)
    arr = arr[0::2] + 1j * arr[1::2]
    #arr = arr.reshape(mol.nao_2c(), mol.nao_2c()).T
    return arr
