Design Decisions
================

This page documents key architectural decisions in roiextractors, including the rationale
behind them. It is intended for future developers so they know why things are the way they are.

Timestamps
----------

Roiextractors is tightly integrated with
`NeuroConv <https://neuroconv.readthedocs.io/en/main/index.html>`_, which writes data to NWB.
When NeuroConv decides whether to store timestamps or just a sampling rate in NWB, the
timestamp API must satisfy three constraints:

1. **Respect user intentions.** If the user (or upstream code) has explicitly set timestamps
   via ``set_times()``, those must take priority. The user may have computed corrected timing
   from TTL pulses or applied temporal alignment shifts, and that decision should be preserved.

2. **Be faithful to the source format.** If the data format contains hardware timestamps
   recorded by the acquisition system and the user has not used ``set_times()``, those
   timestamps should be written to NWB.

3. **Avoid unnecessary allocation.** If neither user-set nor native timestamps exist, timing
   can only be reconstructed from the sampling rate. The API should let NeuroConv detect this
   case without allocating a full ``np.arange(n) / sampling_frequency`` array.

These three constraints create a tension. A single method like ``has_time_vector()`` that only
checks whether ``_times`` has been cached cannot distinguish between "no timestamps were set"
and "native timestamps exist but have not been loaded yet." Eagerly loading native timestamps at
initialization would resolve this ambiguity, but it would increase the memory footprint of every
extractor regardless of whether those timestamps are ever needed. To address this, the current
solution separates the question of *provenance* (does the source format have timestamps?) from the act
of *materializing* them.

Separation of ``get_native_timestamps`` and ``get_timestamps``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To satisfy these constraints, the timestamp API separates concerns into distinct methods:

- ``get_timestamps()`` provides a consistent API that always returns timestamps regardless of
  the state of the extractor. Its contract is simple: call it, get a ``numpy.ndarray`` of times
  in seconds. Users never need to worry about whether the format has native timestamps, whether
  someone called ``set_times()``, or whether timing must be reconstructed from the sampling rate.

- ``get_native_timestamps()`` is the method that format readers override. It answers a
  **provenance** question: *"Does this format have hardware timestamps?"* If yes, it returns
  them; if not, the default implementation returns ``None``. Returning ``None`` rather than
  raising an exception lets NeuroConv distinguish "no native timestamps" from "timestamps
  exist but failed to load". This addresses constraint 2.

With this separation, downstream code can resolve all three constraints through a simple
priority chain:

.. code-block:: text

  What timing information is available?
      |
      |-- has_time_vector()?
      |       |
      |       yes --> user-set timestamps (constraint 1: respect user intentions)
      |
      |-- get_native_timestamps() is not None?
      |       |
      |       yes --> hardware timestamps from the source format (constraint 2: faithful to source)
      |
      '-- sampling rate only (constraint 3: no unnecessary allocation)

``has_time_vector()`` is a cheap check (it tests whether ``_times`` has been cached).
``get_native_timestamps()`` is only called when the first check fails, and it returns ``None``
immediately for formats without hardware timestamps, so no array is allocated in that case.
When native timestamps do exist, loading them is unavoidable: any consumer that needs to decide
between writing a rate or a full timestamp array must inspect the actual values.

.. note::

   The name ``get_native_timestamps`` was chosen over ``get_original_timestamps`` because the
   latter already exists in NeuroConv with different semantics: there it returns timestamps as
   they were at initialization, before any user modifications or temporal alignment shifts,
   rather than indicating provenance. Using a distinct name avoids this conflict.

The original discussion is in `issue #448 <https://github.com/catalystneuro/roiextractors/issues/448>`_
and `PR #465 <https://github.com/catalystneuro/roiextractors/pull/465>`_.


Why ``get_native_timestamps`` is abstract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``get_native_timestamps()`` is marked ``@abstractmethod`` on both ``ImagingExtractor`` and
``SegmentationExtractor``, even though the method body provides a default ``return None``. This
is intentional: it forces every concrete extractor to explicitly implement the method and decide
whether its format has hardware timestamps.

Without the ``@abstractmethod`` decorator, a developer writing a new extractor could easily
overlook ``get_native_timestamps`` entirely. Their extractor would silently inherit the default
``return None``, and if the format *did* contain hardware timestamps, no one would notice until
a user found incorrect timing in their NWB file. Since timestamp provenance directly determines
what NeuroConv writes to NWB (a sampling rate vs. a full timestamps array), a silent omission
here is a data-correctness bug.

The cost of the decorator is small: extractors without native timestamps write a two-line
override that returns ``None``. The benefit is that every extractor author is required to
consciously opt out rather than accidentally inheriting a default they may not know exists.


Copy vs. view semantics for sliced timestamps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond deciding *which* timestamps to store, slicing introduces a question about *how*
timestamps are shared in memory between parent and child extractors.

When an extractor is temporally sliced (e.g., via ``slice_samples()``), there are two constraints
in tension:

- We do not want to **copy** the full timestamps array into the child extractor. That would
  duplicate memory unnecessarily.
- But the child must be **independent** from the parent. Calling ``set_times()`` on the child
  should not affect the parent, and vice versa.

This is an instance of the general `aliasing problem
<https://numpy.org/doc/stable/user/basics.copies.html>`_: shared mutable state risks unintended
side effects. In NumPy, this manifests as the view vs. copy distinction. A **view**
(``parent._times[start:end]``) is memory-efficient but shares the underlying data: in-place
mutations on one side propagate to the other. A **copy** is safe but duplicates memory.

The current implementation uses a view, so it is memory-efficient but also has the benefits of a
copy:

.. code-block:: python

   self._times = self._parent_imaging._times[start_sample:end_sample]

This works because of an internal contract that roiextractors must uphold: ``_times`` is only
ever replaced wholesale (via ``set_times()``), never modified element-by-element. Consider both
directions:

- **Parent calls ``set_times()``**: the parent rebinds its ``_times`` to a new array. The
  child's view still references the original array, so it is unaffected.
- **Child calls ``set_times()``**: the child rebinds its ``_times`` to a new array, abandoning
  the view. The parent's array is unaffected.

In both cases, the replacement (rather than mutation) of ``_times`` means the view is either
preserved or abandoned, never partially modified. The result is the memory efficiency of views
with the isolation of copies.

See
`PR #498 <https://github.com/catalystneuro/roiextractors/pull/498>`_ for the original discussion.
