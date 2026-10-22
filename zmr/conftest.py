'''Test-session environment check.

The reference implementation and several tests use socutils.fci.zfci, which
lives only in the socutils-git checkout.  Make it importable first, e.g.

    source ~/apps/use-socutils-git.sh
'''
import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: long-running test (brute-force references on larger systems)')
    try:
        import socutils.fci.zfci  # noqa: F401
    except ImportError as err:
        raise pytest.UsageError(
            'socutils.fci.zfci is not importable (%s).  Put the socutils-git '
            'checkout on PYTHONPATH, e.g. `source ~/apps/use-socutils-git.sh`.'
            % err)
