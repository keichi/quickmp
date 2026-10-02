Checking Vectorization
======================

The STOMP implementation shared by the CPU and VE backends
(``src/stomp.hpp``) relies on compiler auto-vectorization. After changing it,
check the optimization reports of the compilers to make sure that the loops
are still vectorized.

Loops to Check
--------------

The innermost loops over ``j`` in the following functions account for almost
all of the execution time and must be vectorized:

- ``selfjoin_rows``: the loop that updates ``QT2[j]`` from ``QT[j - 1]``
- ``abjoin_rows``: the loop that updates ``QT2[j]`` from ``QT[j - 1]``

The other loops (statistics, the first row of each chunk, merging per-thread
profiles and conversion to distances) are O(n) and matter much less, but they
are vectorized as well unless noted in `Known Caveats`_. Outer loops are not
vectorized by design.

Each loop is instantiated for both the normalized and non-normalized metrics,
so it appears twice in the reports.

CPU (GCC, Clang, Intel oneAPI)
------------------------------

Compile ``src/cpu/backend.cpp``, which includes ``src/stomp.hpp``, with the
release flags and an optimization report:

.. code-block:: bash

   FLAGS="-std=c++17 -O3 -march=native -ffast-math -Isrc -c src/cpu/backend.cpp -o /dev/null"

   # GCC: look for "loop vectorized" and "couldn't vectorize loop"
   g++ $FLAGS -fopenmp -fopt-info-vec-all 2> gcc.txt
   grep stomp.hpp gcc.txt

   # Clang: look for "vectorized loop" and "loop not vectorized"
   clang++ $FLAGS -fopenmp -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
       -Rpass-analysis=loop-vectorize 2> clang.txt
   grep stomp.hpp clang.txt

   # Intel oneAPI: look for "LOOP WAS VECTORIZED" in the report file
   icpx $FLAGS -qopenmp -qopt-report=3 -qopt-report-phase=vec -qopt-report-file=icpx.txt

GCC and Clang report the line and column of each loop. icpx reports nested
loops in a ``LOOP BEGIN`` / ``LOOP END`` hierarchy; inner loops sometimes have
no location, in which case identify them by the source locations of the loads
and stores listed under them.

VE (NEC Compiler)
-----------------

Compile ``src/ve/stomp.vcpp`` with the same options as the device library and
a vectorization diagnostic:

.. code-block:: bash

   nc++ -x c++ -fopenmp -fpic -O4 -finline-functions -fdiag-vector=2 \
       -I/opt/nec/ve/share/veoffload-veda/include -Isrc \
       -c src/ve/stomp.vcpp -o /dev/null 2>&1 | grep stomp.hpp

Look for ``Vectorized loop.`` and ``Unvectorized loop.`` messages. Use
``-report-all`` to additionally get a listing (``stomp.L``) that shows how each
loop was optimized.

Known Caveats
-------------

- **icpx ignores** ``__restrict`` **on function arguments** and assumes
  dependencies between ``QT2`` and the other arrays, which prevents
  vectorization of the main loops. The row functions therefore copy their
  arguments to local ``__restrict`` pointers. Keep this pattern when adding
  new arguments.
- **GCC versions the main loops** ("loop versioned for vectorization because of
  possible aliasing"), i.e., it adds a runtime alias check before the vectorized
  loop. The arrays never overlap, so the vectorized version is always taken.
- **Clang may not vectorize** ``std::sqrt`` with some standard libraries (e.g.,
  Clang 21 with the system libstdc++ on RHEL 8 reports "library call cannot be
  vectorized" even with ``-ffast-math``). Only the O(n) conversion to distances
  is affected.
- The loop that assigns infinity to subsequences without any pair in the
  non-normalized ``selfjoin`` uses integer operations on purpose (``-ffast-math``
  assumes that infinities do not occur) and is not vectorized. It is O(n).
- **NCC** does not vectorize loops whose bound contains a ``std::min`` call
  ("Vectorization obstructive function reference"). This only affects the
  short loop over the exclusion zone of the first subsequence.
