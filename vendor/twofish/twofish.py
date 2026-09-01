"""Import shim for the ``twofish`` C extension.

The real package (https://pypi.org/project/twofish/) ships only an sdist and
needs a C compiler to build. ``pyjks`` - pulled in transitively by ``hopsworks``
- imports ``twofish`` at module load but only instantiates :class:`Twofish` when
decrypting a Twofish-encrypted JKS keystore entry, a code path the API-key REST
login to Hopsworks never touches.

On Linux / macOS the real sdist builds fine and this shim is not used (see the
environment marker in ``requirements.txt``). On Windows without MSVC build tools
it stands in so the install succeeds. If something ever does invoke the cipher,
this raises loudly rather than corrupting data.
"""

__version__ = "0.3.0"


class Twofish:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "this 'twofish' is a pure-Python import shim with no cipher "
            "implementation (the real package needs a C compiler, absent on this "
            "machine). Something asked to actually run Twofish - install the real "
            "'twofish' wheel: pip install --force-reinstall --no-binary twofish twofish"
        )
