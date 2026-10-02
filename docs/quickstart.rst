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

   # Switch to a specific device (for the calling thread)
   quickmp.use_device(0)

   # Check current device
   current = quickmp.get_current_device()
   print(f"Current device: {current}")

   quickmp.finalize()

The selected device is per thread. Threads that have not called
``use_device`` use device 0.

Batch Computation
-----------------

``selfjoin`` and ``abjoin`` use all cores of the device for a single time
series. To compute the matrix profiles of many time series of the same length,
``selfjoin_batch`` is faster since it processes the time series in parallel:

.. code-block:: python

   import numpy as np
   import quickmp

   quickmp.initialize()

   # Each row is a time series
   T = np.random.rand(1000, 7200)
   P = quickmp.selfjoin_batch(T, m=10)  # shape: (1000, 7191)

   quickmp.finalize()

To use multiple devices, split the time series among one thread per device:

.. code-block:: python

   import threading

   # T: 2D array of time series, after quickmp.initialize()
   def worker(device, T):
       quickmp.use_device(device)
       results[device] = quickmp.selfjoin_batch(T, m=10)

   num_devices = quickmp.get_device_count()
   chunks = np.array_split(T, num_devices)
   results = [None] * num_devices
   threads = [threading.Thread(target=worker, args=(d, chunks[d]))
              for d in range(num_devices)]
   for t in threads:
       t.start()
   for t in threads:
       t.join()
   P = np.concatenate(results)

Since each call already uses all cores of the device, calling functions
concurrently from multiple threads on the same device does not make them
faster.

Number of Threads
-----------------

quickmp uses OpenMP to parallelize computations over the cores of a device.
The number of threads defaults to the number of cores and can be changed with
``OMP_NUM_THREADS`` on CPU and ``VE_OMP_NUM_THREADS`` on Vector Engine.
