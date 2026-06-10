import os
import pathlib
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = ["--config", cfg, "--", "-j"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=self.build_temp)

        # Place the shared library inside the x2camf package so that the
        # ctypes loader finds it next to libx2camf.py.
        patterns = ["libx2camf_c*.so", "libx2camf_c*.dylib", "x2camf_c*.dll"]
        shared_libs = []
        for pattern in patterns:
            shared_libs.extend(pathlib.Path(extdir).glob(pattern))
        if not shared_libs:
            raise RuntimeError(f"No x2camf_c shared library found in {extdir}")

        target_package_dir = extdir / "x2camf"
        target_package_dir.mkdir(parents=True, exist_ok=True)
        for lib in shared_libs:
            target = target_package_dir / lib.name
            lib.replace(target)
            print(f"Moved {lib} -> {target}")
            # Also copy into the source tree for editable/in-place use.
            inplace = pathlib.Path("x2camf") / lib.name
            if inplace.resolve() != target.resolve():
                import shutil
                shutil.copy2(target, inplace)
                print(f"Copied {target} -> {inplace}")


setup(
    name='x2camf-c',
    version='0.1',
    description='pure-C reimplementation of the X2CAMF code with a ctypes '
                'interface (no C++, no pybind11)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=[CMakeExtension("x2camf_c")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        "x2camf": ["libx2camf_c*.so", "libx2camf_c*.dylib", "x2camf_c*.dll"],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=["numpy"],
    python_requires=">=3.7",
)
