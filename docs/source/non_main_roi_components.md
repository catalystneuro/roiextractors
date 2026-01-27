# Non-Main ROI Components: Background and Neuropil Handling

## Overview

This document describes how different segmentation methods handle ROI components beyond the main cellular ROIs. These include background components, neuropil signals, and other non-cellular signals that are important for contamination correction and accurate fluorescence estimation.

## Terminology and Concepts

### Scientific Literature Usage

The calcium imaging field uses several overlapping but distinct terms for non-cellular signals. Based on the major papers and methods:

**Common Terms in the Literature:**

1. **Neuropil** - Most commonly used term
   - Refers to the dense network of neuronal processes (axons, dendrites) and their synapses
   - In imaging context: fluorescence from out-of-focus neurites and processes
   - Used by: Suite2p, CaImAn, FISSA, most neuroscience papers

2. **Neuropil contamination** / **Neuropil decontamination**
   - The problem of out-of-focus fluorescence contaminating somatic signals
   - FISSA paper: "signals from surrounding neurites (axons and dendrites)"
   - Can contribute 10-30% of measured fluorescence depending on numerical aperture

3. **Background fluorescence**
   - Broader term including all non-signal fluorescence
   - Includes: autofluorescence, scattered light, non-specific dye binding
   - Often used interchangeably with "neuropil" but technically more general

4. **Out-of-focus contamination**
   - Specific to the optical phenomenon in 2-photon microscopy
   - Signals from above/below the focal plane due to elongated PSF along z-axis

5. **Global neuropil signal** vs **Local neuropil signal**
   - Global: spatially uniform background across entire field of view
   - Local: region-specific background around each ROI (Suite2p approach)

**In ROIExtractors context:**
- **"Background components"** = Global/regional components with spatial masks (CaImAn/Minian style)
- **"Neuropil"** = Local per-cell signals without spatial masks (Suite2p style)

Both terms are valid, but "neuropil" is more common in neuroscience literature, while "background" is more generic and used in signal processing.

### Background Components

**Background components** are global or regional spatial-temporal components that capture non-cellular fluorescence sources in the field of view. They have both:
- **Spatial masks**: Image or pixel masks showing where the background component is located
- **Temporal traces**: Fluorescence dynamics of the background over time
- **Separate ROI IDs**: Identified with IDs starting with `"background"`

**Purpose**: Model spatially distributed background fluorescence from sources like:
- Out-of-focus neuropil
- Autofluorescence
- Scattered light
- Non-specific dye binding

**Key Characteristics**:
- Independent spatial components (not tied to individual cells)
- Can have overlapping spatial footprints with cells
- Number of background components is determined by the algorithm
- Retrieved via `get_background_ids()`, `get_background_image_masks()`, `get_background_pixel_masks()`

### Neuropil Components

**Neuropil components** are local background signals measured around individual ROIs. In Suite2p's implementation:
- **No separate spatial masks**: Neuropil is defined algorithmically as an annulus around each cell
- **Temporal traces**: One neuropil trace per cell ROI
- **Same ROI IDs**: Uses the same IDs as the corresponding cell ROIs
- **ROI ID Mapping**: The neuropil trace at index `i` corresponds to the cell ROI at index `i` with ID `i`
- **Response type**: `"neuropil"` (not `"background"`)

**Purpose**: Correct for local contamination from:
- Nearby out-of-focus neurites
- Nearby cell bodies
- Local neuropil activity

**Key Characteristics**:
- Paired with individual cells (one-to-one relationship)
- Each neuropil trace is identified by the **cell's ROI ID**, not the annulus itself
- The annulus spatial location is defined algorithmically by Suite2p (not stored as a mask)
- Number of neuropil traces equals number of cells
- Retrieved via `get_traces(name="neuropil")`

**Important**: When you call `get_traces(roi_ids=[0, 5, 10], name="neuropil")`, you get the neuropil signals from the annuli around cells 0, 5, and 10, NOT from separate neuropil ROIs.

### Summary: Background vs Neuropil

| Feature | Background | Neuropil |
|---------|-----------|----------|
| **Spatial representation** | Has explicit spatial masks | No explicit masks stored |
| **Relationship to cells** | Independent components | One per cell (paired) |
| **Number of components** | Algorithm-determined (typically 1-5) | Equals number of cells |
| **ROI IDs** | Separate (e.g., `"background0"`) | Same as cell IDs |
| **Response type** | `"background"` | `"neuropil"` |
| **Detection method** | ID starts with `"background"` | Response type = `"neuropil"` |
| **Retrieval methods** | `get_background_*()` methods | `get_traces(name="neuropil")` |
| **Use case** | Global background subtraction | Local contamination correction |

## Segmentation Methods with Non-Main Components

### Methods Overview

| Method | Background Components | Neuropil Components | Other Non-Main Components |
|--------|---------------------|-------------------|--------------------------|
| **CaImAn** | Yes | No | No |
| **Minian** | Yes | No | No |
| **Suite2p** | No | Yes | No |
| **NWB** | Yes (if written) | Yes (if written) | Yes (any labeled trace type) |
| **SIMA** | No | No | No |
| **Inscopix** | No | No | No |
| **EXTRACT** | No | No | No |
| **CNMF-E** | No | No | No |

---

## CaImAn Segmentation Extractor

**File**: [src/roiextractors/extractors/caiman/caimansegmentationextractor.py](../../src/roiextractors/extractors/caiman/caimansegmentationextractor.py)

### Background Component Support

**Status**: Full support

**Data Source**:
- **Spatial**: `b` field in HDF5 file
  - Shape: `(n_pixels, n_background_components)`
  - Format: Fortran-order flattened spatial footprints
  - Located at: `estimates['b']` in the HDF5 file
- **Temporal**: `f` field in HDF5 file
  - Shape: `(n_background_components, n_timesteps)`
  - Located at: `estimates['f']` in the HDF5 file

**ROI ID Format**:
```
"background0", "background1", "background2", ...
```
Pattern: `background{index}` where index starts at 0

**Implementation Details**:
- Background spatial masks are converted from Fortran-order flattened format to pixel masks during initialization
- Pixel masks stored as `(n_pixels, 3)` arrays with columns `[y, x, weight]`
- Background and cell ROIs stored together in `_ROIMasks` container
- Background traces stored as `_RoiResponse` with type `"background"`
- Code reference: Lines [170-173](../../src/roiextractors/extractors/caiman/caimansegmentationextractor.py#L170-L173), [244-266](../../src/roiextractors/extractors/caiman/caimansegmentationextractor.py#L244-L266)

**Usage Example**:
```python
from roiextractors import CaimanSegmentationExtractor

extractor = CaimanSegmentationExtractor(file_path="caiman_output.hdf5")

# Get background component IDs
bg_ids = extractor.get_background_ids()
# Returns: ["background0", "background1", ...]

# Get background spatial masks (image format)
bg_image_masks = extractor.get_background_image_masks()
# Returns: np.ndarray with shape (height, width, n_backgrounds)

# Get background spatial masks (pixel format)
bg_pixel_masks = extractor.get_background_pixel_masks()
# Returns: list of (n_pixels, 3) arrays with [y, x, weight]

# Get background temporal traces
bg_traces = extractor.get_traces(name="background")
# Returns: np.ndarray with shape (n_timesteps, n_backgrounds)
```

**Why CaImAn Uses Background Components**:

CaImAn (Constrained Nonnegative Matrix Factorization) models the imaging data as:
```
Y = A*C + b*f + noise
```
where:
- `A` = spatial footprints of cells
- `C` = temporal activity of cells
- `b` = spatial background components
- `f` = temporal background dynamics

This allows modeling spatially varying background that cannot be captured by simple per-pixel baselines.

---

## Minian Segmentation Extractor

**File**: [src/roiextractors/extractors/minian/miniansegmentationextractor.py](../../src/roiextractors/extractors/minian/miniansegmentationextractor.py)

### Background Component Support

**Status**: Full support

**Data Source**:
- **Spatial**: `b.zarr` file
  - Shape: `(height, width)` for single background OR `(height, width, n_backgrounds)` for multiple
  - Format: Direct image masks (not flattened)
  - Zarr structure: `b.zarr/b` dataset
- **Temporal**: `f.zarr` file
  - Shape: `(frame, n_background_components)`
  - Zarr structure: `f.zarr/f` dataset

**ROI ID Format**:
```
"background-0", "background-1", "background-2", ...
```
Pattern: `background-{index}` where index starts at 0 (note the hyphen `-`)

**Implementation Details**:
- Background masks stored directly as image masks (not flattened)
- Handles both 2D (single background) and 3D (multiple backgrounds) arrays
- Automatically expands 2D background to 3D format: `(H, W)` → `(H, W, 1)`
- Background and cell masks concatenated along last axis
- Validates spatial-temporal consistency (raises error if spatial masks exist without temporal traces)
- Code reference: Lines [94-100](../../src/roiextractors/extractors/minian/miniansegmentationextractor.py#L94-L100), [176-191](../../src/roiextractors/extractors/minian/miniansegmentationextractor.py#L176-L191), [242-264](../../src/roiextractors/extractors/minian/miniansegmentationextractor.py#L242-L264)

**Usage Example**:
```python
from roiextractors import MinianSegmentationExtractor

extractor = MinianSegmentationExtractor(folder_path="minian_output/")

# Get background component IDs
bg_ids = extractor.get_background_ids()
# Returns: ["background-0", "background-1", ...]

# Get background spatial masks
bg_image_masks = extractor.get_background_image_masks()
# Returns: np.ndarray with shape (height, width, n_backgrounds)

# Get background pixel masks
bg_pixel_masks = extractor.get_background_pixel_masks()
# Returns: list of (n_pixels, 3) arrays with [y, x, weight]

# Get background temporal traces
bg_traces = extractor.get_traces(name="background")
# Returns: np.ndarray with shape (n_timesteps, n_backgrounds)

# Number of background components
n_bg = extractor.get_num_background_components()
```

**Why Minian Uses Background Components**:

Minian uses a similar matrix factorization approach to CaImAn, modeling spatially distributed background fluorescence. The background components help separate cellular signals from:
- Diffuse neuropil fluorescence
- Slow-varying baseline fluctuations
- Spatially structured background patterns

**Validation**:
Minian enforces strict spatial-temporal consistency (lines [123-127](../../src/roiextractors/extractors/minian/miniansegmentationextractor.py#L123-L127)):
- If `b.zarr` exists but `f.zarr` is missing → raises `ValueError`
- Ensures background masks always have corresponding temporal dynamics

---

## Suite2p Segmentation Extractor

**File**: [src/roiextractors/extractors/suite2p/suite2psegmentationextractor.py](../../src/roiextractors/extractors/suite2p/suite2psegmentationextractor.py)

### Neuropil Component Support

**Status**: Neuropil traces only (no background components)

**Data Source**:
- **Spatial**: None stored explicitly (defined algorithmically)
- **Temporal**: `Fneu.npy` (or `Fneu_chan2.npy` for channel 2)
  - Shape: `(n_frames, n_cells)`
  - One neuropil trace per cell

**ROI IDs**: Uses the same ROI IDs as the cells (not separate IDs)

**Implementation Details**:
- Neuropil stored as `_RoiResponse` with type `"neuropil"` (lines [154-157](../../src/roiextractors/extractors/suite2p/suite2psegmentationextractor.py#L154-L157))
- ROI IDs for neuropil traces match cell ROI IDs
- No explicit spatial masks for neuropil (Suite2p defines neuropil as an annulus around each ROI)
- Neuropil is NOT counted as background components (`get_num_background_components()` returns 0)

**Usage Example**:
```python
from roiextractors import Suite2pSegmentationExtractor

extractor = Suite2pSegmentationExtractor(folder_path="suite2p/")

# Get neuropil traces (one per cell)
neuropil_traces = extractor.get_traces(name="neuropil")
# Returns: np.ndarray with shape (n_frames, n_cells)

# Neuropil uses same ROI IDs as cells
cell_ids = extractor.get_roi_ids()
# Returns: [0, 1, 2, ..., n_cells-1]

# Get neuropil for specific cells
neuropil_subset = extractor.get_traces(roi_ids=[0, 5, 10], name="neuropil")
# Returns: np.ndarray with shape (n_frames, 3)
# Column 0 = neuropil around cell 0
# Column 1 = neuropil around cell 5
# Column 2 = neuropil around cell 10

# The neuropil trace is paired with the cell - same ROI ID
cell_trace_0 = extractor.get_traces(roi_ids=[0], name="raw")
neuropil_trace_0 = extractor.get_traces(roi_ids=[0], name="neuropil")
# Both use ROI ID 0, but cell_trace_0 is FROM the cell,
# neuropil_trace_0 is FROM the annulus AROUND the cell

# Note: get_background_ids() returns empty list
bg_ids = extractor.get_background_ids()
# Returns: []

# Note: get_num_background_components() returns 0
n_bg = extractor.get_num_background_components()
# Returns: 0
```

**Why Suite2p Uses Neuropil Instead of Background**:

Suite2p takes a different approach to background correction:
- Each cell has a local neuropil signal measured from an annulus around the ROI
- The corrected fluorescence is: `F_corrected = F_cell - neuropil_coefficient * F_neuropil`
- This local approach is simpler and assumes background is primarily from nearby neuropil
- Does not model global background patterns like CaImAn/Minian

**Key Difference**: Suite2p's neuropil is cell-specific and doesn't have separate spatial masks, making it fundamentally different from CaImAn/Minian's background components.

---

## NWB Segmentation Extractor

**File**: [src/roiextractors/extractors/nwbextractors/nwbextractors.py](../../src/roiextractors/extractors/nwbextractors/nwbextractors.py)

### Multi-Component Support

**Status**: Full support for any trace type (background, neuropil, denoised, deconvolved, baseline, etc.)

**Data Source**:
NWB files can store multiple `RoiResponseSeries` with different trace types in:
- `processing/<module>/Fluorescence/<RoiResponseSeries>`
- `processing/<module>/DfOverF/<RoiResponseSeries>`

**Supported Trace Types** (line [317](../../src/roiextractors/extractors/nwbextractors/nwbextractors.py#L317)):
- `raw`: Raw fluorescence
- `dff`: ΔF/F
- `neuropil`: Neuropil/local background traces
- `deconvolved`: Deconvolved/spike inference
- `denoised`: Denoised traces
- `baseline`: Baseline fluorescence
- `background`: Background component traces

**ROI ID Format**:
- Background: IDs must start with `"background"` to be detected as background components
- Neuropil: Uses same IDs as cells
- Other trace types: Uses IDs from `PlaneSegmentation.id` column

**Implementation Details**:
- All trace types stored as `_RoiResponse` objects with appropriate response types
- Background detection relies on ROI ID string matching: `str(roi_id).startswith("background")`
- Spatial masks read from `PlaneSegmentation` columns: `image_mask`, `pixel_mask`, or `voxel_mask`
- Code reference: Lines [312-374](../../src/roiextractors/extractors/nwbextractors/nwbextractors.py#L312-L374)

**Usage Example**:
```python
from roiextractors import NwbSegmentationExtractor

extractor = NwbSegmentationExtractor(
    file_path="ophys.nwb",
    processing_module_name="ophys"
)

# Get available trace types
traces_dict = extractor.get_traces_dict()
# Returns: {"raw": array, "dff": array, "neuropil": array, "background": array, ...}

# Get background components (if ROI IDs start with "background")
bg_ids = extractor.get_background_ids()
# Returns: ["background0", "background1", ...] or []

# Get neuropil traces (if available)
neuropil_traces = extractor.get_traces(name="neuropil")

# Get deconvolved traces (if available)
deconv_traces = extractor.get_traces(name="deconvolved")
```

**Why NWB is Flexible**:

NWB is a data standard, not a specific algorithm, so it can store outputs from any segmentation method:
- CaImAn outputs → stores background components
- Suite2p outputs → stores neuropil traces
- Any custom pipeline → stores any trace type with appropriate naming

**Important**: The NWB extractor will only recognize components as "background" if their ROI IDs start with `"background"`. Otherwise, they are treated as regular ROIs.

---

## Methods Without Non-Main Components

The following extractors only support main cell ROIs:

### SIMA Segmentation Extractor
**File**: [src/roiextractors/extractors/simaextractor/simasegmentationextractor.py](../../src/roiextractors/extractors/simaextractor/simasegmentationextractor.py)
- Only loads cell ROI traces and masks
- No background or neuropil support

### Inscopix Segmentation Extractor
**File**: [src/roiextractors/extractors/inscopixextractors/inscopixsegmentationextractor.py](../../src/roiextractors/extractors/inscopixextractors/inscopixsegmentationextractor.py)
- Only loads cell traces from `.isxd` files
- No background or neuropil in Inscopix data format

### EXTRACT Segmentation Extractor
**File**: [src/roiextractors/extractors/schnitzerextractor/extractsegmentationextractor.py](../../src/roiextractors/extractors/schnitzerextractor/extractsegmentationextractor.py)
- Only loads cell ROI traces and masks
- No background or neuropil support

### CNMF-E Segmentation Extractor
**File**: [src/roiextractors/extractors/schnitzerextractor/cnmfesegmentationextractor.py](../../src/roiextractors/extractors/schnitzerextractor/cnmfesegmentationextractor.py)
- Only loads cell ROI traces and masks
- No background or neuropil support
- Note: CNMF-E algorithm does model background, but it's not currently extracted

---

## API Methods for Non-Main Components

### SegmentationExtractor Base Class Methods

The base class provides standardized methods for accessing non-main components:

#### Background Components

```python
def get_background_ids() -> list:
    """Get the list of background component IDs.

    Returns
    -------
    background_ids : list
        List of background component IDs (detected by ID starting with "background")
    """
```

```python
def get_background_image_masks(background_ids=None) -> np.ndarray:
    """Get background image masks.

    Parameters
    ----------
    background_ids : array_like, optional
        List of background IDs to retrieve. If None, gets all backgrounds.

    Returns
    -------
    background_image_masks : np.ndarray
        3D array with shape (height, width, n_backgrounds)
        Values are 0 or weights for weighted masks
    """
```

```python
def get_background_pixel_masks(background_ids=None) -> list[np.ndarray]:
    """Get background pixel masks (sparse representation).

    Parameters
    ----------
    background_ids : array_like, optional
        List of background IDs to retrieve. If None, gets all backgrounds.

    Returns
    -------
    pixel_masks : list of np.ndarray
        List of (n_pixels, 3) arrays with columns [y, x, weight]
    """
```

```python
def get_num_background_components() -> int:
    """Get the number of background components.

    Returns
    -------
    n_backgrounds : int
        Number of background components
    """
```

#### All Trace Types (Including Neuropil)

```python
def get_traces(roi_ids=None, start_frame=None, end_frame=None, name="raw") -> np.ndarray:
    """Get fluorescence traces of specified type.

    Parameters
    ----------
    roi_ids : array_like, optional
        List of ROI IDs to retrieve
    start_frame : int, optional
        Start frame index
    end_frame : int, optional
        End frame index
    name : str
        Trace type: "raw", "dff", "neuropil", "background", "deconvolved", "denoised", "baseline"

    Returns
    -------
    traces : np.ndarray
        2D array with shape (n_frames, n_rois)
    """
```

```python
def get_traces_dict() -> dict:
    """Get all available trace types as a dictionary.

    Returns
    -------
    traces_dict : dict
        Dictionary with keys: "raw", "dff", "neuropil", "background", "deconvolved", "denoised", "baseline"
        Values are np.ndarray or None if not available
    """
```

---

## Implementation Notes

### Background Detection Logic

The base class identifies background components by string matching (lines [436](../../src/roiextractors/segmentationextractor.py#L436), [692](../../src/roiextractors/segmentationextractor.py#L692)):

```python
background_ids = [rid for rid in all_roi_ids if str(rid).startswith("background")]
```

**Implication**: Any ROI ID starting with `"background"` will be treated as a background component, regardless of the actual data content.

### Counting Fallback

When ROI masks are not available, `get_num_background_components()` falls back to counting from response traces (lines [696-706](../../src/roiextractors/segmentationextractor.py#L696-L706)):

```python
for response in self._roi_responses:
    if response.response_type in {"neuropil", "background"}:
        # Count from trace dimensions
        return int(data.shape[1])
```

**Note**: This treats both `"neuropil"` and `"background"` response types as background-like for counting purposes, even though they are conceptually different.

### ROI Masks Container

Background and cell ROIs are stored together in `_ROIMasks` container with a unified ROI ID mapping:

```python
_roi_masks = _ROIMasks(
    data=pixel_masks,  # or image_masks
    mask_tpe="nwb-pixel_mask",  # or "nwb-image_mask"
    field_of_view_shape=(height, width),
    roi_id_map={
        0: 0,              # cell 0
        1: 1,              # cell 1
        "background0": 2,  # background 0
        "background1": 3,  # background 1
    }
)
```

This unified storage allows efficient access to all spatial components.

---

## Conversion and Compatibility

### Converting Between Formats

When converting between formats, be aware of the differences:

**CaImAn → Suite2p**:
- Background components will be lost (Suite2p doesn't support them)
- Would need to convert to per-cell neuropil (not straightforward)

**Suite2p → CaImAn**:
- Neuropil traces will be lost (CaImAn expects global background)
- Could store neuropil as separate trace type in NWB

**Any Format → NWB**:
- Background components: Store with IDs starting with `"background"`
- Neuropil traces: Store as `RoiResponseSeries` with name `"Neuropil"`
- Both can coexist in same NWB file

### FrameSlice Support

`FrameSliceSegmentationExtractor` properly propagates background components (added in [PR #378](https://github.com/catalystneuro/roiextractors/pull/378)):

```python
sliced = FrameSliceSegmentationExtractor(
    parent_segmentation=extractor,
    start_sample=100,
    end_sample=1000
)

# Background masks are preserved
bg_ids = sliced.get_background_ids()  # Same as parent
bg_masks = sliced.get_background_image_masks()  # Same as parent

# Background traces are sliced
bg_traces = sliced.get_traces(name="background")  # [100:1000, :]
```

---

## Best Practices

### For Users

1. **Check what's available**: Always use `get_traces_dict()` to see what trace types are available before accessing them.

2. **Background vs Neuropil**: Understand which your data has:
   - `get_num_background_components() > 0` → Has background components
   - `get_traces(name="neuropil") is not None` → Has neuropil traces

3. **ID conventions**: When writing data to NWB, use consistent ID naming:
   - Background: `"background0"`, `"background1"`, etc.
   - Cells: Integers or descriptive strings without "background" prefix

4. **Validation**: Check spatial-temporal consistency:
   ```python
   n_bg = len(extractor.get_background_ids())
   bg_traces = extractor.get_traces(name="background")
   assert bg_traces.shape[1] == n_bg
   ```

### For Developers

1. **Adding new extractors**: If your format has background-like components:
   - Use ID naming: `"background{index}"` for global background
   - Use response type `"neuropil"` for per-cell local background
   - Store spatial masks if available

2. **Testing**: Ensure you test:
   - `get_background_ids()` returns correct IDs
   - `get_background_image_masks()` returns correct shapes
   - Temporal traces match spatial components

3. **Documentation**: Clearly document whether your method provides:
   - Global background components
   - Per-cell neuropil
   - Both
   - Neither

---

## Changelog References

Background component support was added in several stages:

- **v0.4.18** ([PR #291](https://github.com/catalystneuro/roiextractors/pull/291)):
  - Added `get_background_ids()`, `get_background_image_masks()`, `get_background_pixel_masks()`
  - Fixed CaImAn background extraction bugs
  - Added distinction for raw vs denoised traces

- **v0.5.9** ([PR #378](https://github.com/catalystneuro/roiextractors/pull/378)):
  - Added background support to `FrameSliceSegmentationExtractor`

---

## Summary Table: ROI ID Conventions

| Method | Cell IDs | Background IDs | Neuropil IDs |
|--------|----------|----------------|--------------|
| CaImAn | 0, 1, 2, ... | `"background0"`, `"background1"`, ... | N/A |
| Minian | 0, 1, 2, ... | `"background-0"`, `"background-1"`, ... | N/A |
| Suite2p | 0, 1, 2, ... | N/A | Same as cell IDs |
| NWB | From PlaneSegmentation.id | Must start with `"background"` | Same as cell IDs |

**Note the subtle difference**: CaImAn uses `background0` while Minian uses `background-0` (with hyphen).

---

## FAQ: Understanding Neuropil ROI IDs

**Q: In Suite2p, what ROI ID does the neuropil trace have?**

A: The neuropil trace uses the **same ROI ID as its corresponding cell**. This is because the neuropil is not a separate ROI, but rather a measurement taken from the area around the cell.

**Q: When I call `get_traces(roi_ids=[5], name="neuropil")`, what am I getting?**

A: You are getting the neuropil signal from the annulus (ring-shaped region) **around cell 5**. The trace is identified by cell 5's ID, but the signal comes from the surrounding neuropil, not from cell 5 itself.

**Q: Does the neuropil have its own spatial mask?**

A: No. Suite2p does not store explicit spatial masks for neuropil regions. The neuropil region is defined algorithmically as an annulus around each cell ROI. This information is computed by Suite2p during segmentation but not stored in the output files that ROIExtractors reads.

**Q: If I have 100 cells, how many neuropil traces are there?**

A: Exactly 100 neuropil traces - one for each cell. The neuropil traces have a one-to-one pairing with cell ROIs.

**Q: How is this different from CaImAn/Minian background?**

A: The key differences:

| Aspect | Suite2p Neuropil | CaImAn/Minian Background |
|--------|------------------|--------------------------|
| **Spatial representation** | No masks stored | Has explicit spatial masks |
| **Number of components** | Same as number of cells | Independent (e.g., 1-5 total) |
| **ROI IDs** | Use cell IDs | Separate background IDs |
| **What it represents** | Annulus around EACH cell | Global/regional background components |
| **Retrieval** | `get_traces(roi_ids=[i], name="neuropil")` | `get_background_image_masks(background_ids=["background0"])` |

**Q: Can I get the spatial mask of where the neuropil signal came from?**

A: Not directly from ROIExtractors, because Suite2p doesn't store these masks in its output files (`F.npy`, `Fneu.npy`, `stat.npy`). Suite2p computes neuropil regions on-the-fly during processing but only saves the temporal traces. If you need the spatial definition, you would need to:
1. Use Suite2p's code to recompute the neuropil masks from the cell masks
2. Or extract this information during Suite2p's processing before it's discarded

**Q: Concrete example - what exactly is stored?**

A: Here's what the data structures look like:

```python
# Suite2p stores:
F.npy        # Cell fluorescence: shape (n_cells, n_frames)
Fneu.npy     # Neuropil fluorescence: shape (n_cells, n_frames)
stat.npy     # Cell ROI masks only (no neuropil masks)

# When loaded into ROIExtractors:
extractor.get_roi_ids()
# Returns: [0, 1, 2, 3, 4]  (5 cells)

extractor.get_traces(name="raw")
# Returns: (n_frames, 5)  - fluorescence FROM the 5 cells

extractor.get_traces(name="neuropil")
# Returns: (n_frames, 5)  - fluorescence FROM annuli AROUND the 5 cells

# Both traces use the same ROI IDs [0, 1, 2, 3, 4]
# but they represent signals from different spatial regions:
# - "raw" = signal from inside the cell mask
# - "neuropil" = signal from outside the cell mask (annulus)
```

The neuropil traces are **indexed by the cell they surround**, not by a separate neuropil ROI ID.

---

## Terminology Recommendations

Based on the scientific literature and common usage in the calcium imaging field:

### Preferred Terminology

**For scientific/neuroscience contexts:**
- Use **"neuropil"** rather than "background" when referring to contamination from neuronal processes
- Use **"neuropil contamination"** or **"neuropil signal"** for out-of-focus fluorescence from neurites
- Use **"background fluorescence"** for non-biological sources (autofluorescence, scattered light, etc.)

**For technical/implementation contexts:**
- **"Background components"** is appropriate when referring to the mathematical/computational components in matrix factorization (CaImAn/Minian)
- **"Neuropil traces"** is appropriate for per-cell local background measurements (Suite2p)

### Recommended Alternative Terms

If you want a more general term that encompasses both background components AND neuropil:

**Option 1: "Non-somatic components"**
- Pro: Clearly distinguishes from cell bodies (soma)
- Pro: Neutral, doesn't imply spatial structure
- Con: Doesn't indicate these are contamination sources

**Option 2: "Contamination components"**
- Pro: Clearly indicates purpose (signals to be removed/corrected)
- Pro: Encompasses all sources (neuropil, background, autofluorescence)
- Con: Somewhat negative connotation

**Option 3: "Auxiliary components"**
- Pro: Neutral term indicating secondary/supporting components
- Pro: Doesn't pre-judge whether they're good or bad
- Con: Less specific about what they represent

**Option 4: "Non-neuronal ROIs"**
- Pro: Clear distinction from neuronal ROIs
- Con: Technically incorrect - neuropil IS neuronal (from neurites)
- Con: Misleading for glial cell imaging

### Current ROIExtractors Convention

ROIExtractors currently uses:
- **"background"** in method names (`get_background_ids()`, `get_background_image_masks()`)
- **"neuropil"** as a trace response type (`get_traces(name="neuropil")`)

This dual terminology reflects the field's actual usage:
- CaImAn/Minian → background components (spatial + temporal)
- Suite2p → neuropil traces (temporal only, local per-cell)

### Recommendation for Future Development

For maximum clarity and alignment with neuroscience literature, consider:

1. **Keep current API** for backward compatibility
2. **Add aliases**:
   - `get_neuropil_components()` as alias for `get_background_ids()` (when they have spatial masks)
   - Document that "background" and "neuropil" are used interchangeably
3. **In documentation**: Prefer "neuropil" when discussing the biological/optical phenomenon
4. **In code**: Continue using "background" for the computational components in matrix factorization

**Most accurate umbrella term**: **"Non-somatic components"** or **"Contamination sources"**

However, the current mixed usage of "background" (for components with spatial masks) and "neuropil" (for local per-cell traces) accurately reflects how the field actually uses these terms and should be maintained.

---

## See Also

- [SegmentationExtractor API Reference](api/segmentationextractor.rst)
- [NWB Conversion Guide](usage.rst)
- [Compatible Formats](compatible.rst)

## References

Key papers on this topic:
- **FISSA**: Keemink et al. (2018) "FISSA: A neuropil decontamination toolbox for calcium imaging signals" Scientific Reports
- **CaImAn**: Giovannucci et al. (2019) "CaImAn an open source tool for scalable calcium imaging data analysis" eLife
- **Suite2p**: Pachitariu et al. (2017) "Suite2p: beyond 10,000 neurons with standard two-photon microscopy" bioRxiv
