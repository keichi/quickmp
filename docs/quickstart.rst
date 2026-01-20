Quick Start
===========

Basic Usage
-----------

The typical workflow involves initializing the backend, performing computations,
and finalizing when done:

.. code-block:: python

   import numpy as np
   import quickmp

   # Initialize the backend
   quickmp.initialize()

   # Create a time series
   T = np.random.rand(1000)

   # Compute the matrix profile with window size 100
   mp = quickmp.selfjoin(T, m=100)

   # Finalize when done
   quickmp.finalize()

AB-Join
-------

To compute the matrix profile between two different time series:

.. code-block:: python

   import numpy as np
   import quickmp

   quickmp.initialize()

   T1 = np.random.rand(1000)
   T2 = np.random.rand(800)

   # Compute matrix profile between T1 and T2
   mp = quickmp.abjoin(T1, T2, m=100)

   quickmp.finalize()

Normalized vs Unnormalized Distance
-----------------------------------

By default, quickmp uses Z-normalized Euclidean distance.
You can use raw Euclidean distance by setting ``normalize=False``:

.. code-block:: python

   # Z-normalized Euclidean distance (default)
   mp_normalized = quickmp.selfjoin(T, m=100, normalize=True)

   # Raw Euclidean distance
   mp_unnormalized = quickmp.selfjoin(T, m=100, normalize=False)

Multi-Device Usage
------------------

On systems with multiple devices (e.g., multiple Vector Engines), you can
select which device to use:

.. code-block:: python

   quickmp.initialize()

   # Get the number of available devices
   num_devices = quickmp.get_device_count()
   print(f"Available devices: {num_devices}")

   # Switch to a specific device
   quickmp.use_device(0)

   # Check current device
   current = quickmp.get_current_device()
   print(f"Current device: {current}")

   quickmp.finalize()

Parallel Execution with Streams
-------------------------------

quickmp supports stream-based parallelism for concurrent computations:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   import numpy as np
   import quickmp

   quickmp.initialize()

   # Get available streams
   num_streams = quickmp.get_stream_count()

   def compute_on_stream(stream_id, data):
       return quickmp.selfjoin(data, m=100, stream=stream_id)

   # Run computations in parallel using different streams
   with ThreadPoolExecutor(max_workers=num_streams) as executor:
       datasets = [np.random.rand(1000) for _ in range(num_streams)]
       futures = [
           executor.submit(compute_on_stream, i, data)
           for i, data in enumerate(datasets)
       ]
       results = [f.result() for f in futures]

   quickmp.finalize()
