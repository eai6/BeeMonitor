# Videos

## Getting video in

### From a BeeMonitor unit

An enrolled unit uploads its clips and photos by itself (see [Build a unit](../hardware/index.md)). They
appear under **Processing**, already linked to the device, its location and the layout it was recorded with.

### Upload from any camera

**Processing → Upload videos.** Drop in any number of files of any size (MP4, MOV, MKV, AVI, H.264). They go
straight from your browser to storage in parts: if the connection drops or you close the tab, add the same
files again and they pick up where they stopped.

- **When it was recorded.** Each file's time is taken from, in order: the file itself (MP4/MOV cameras write
  it), a date-time in the file name (`2026-10-04_09_12_40`, `20261004_091240`, `IMG_20261004_091240`…), the
  start time you type for the batch, and finally the upload time. The page shows which one each file got;
  clips left on the upload time can be found later with the **Recording time unknown** filter.
- **Where (optional).** Pick a saved site, add a new one (a name, and a pin on the map if you want), or
  leave it empty. A site with a location gives BioCLIP its list of local species. If the clips came off a
  unit's card, choose that device instead.
- **Label.** A batch name to find these clips again (**Batch** filter in Processing).
- **Afterwards.** Choose a pipeline to run on the new clips — straight away when the last file lands, or
  with one click.
- Files you already uploaded (same name and size) are skipped unless you say otherwise. AVI files are
  converted to MP4 after upload so they play in the browser; analysis can use them right away.

### From code

The [API](api.md) uploads clips and runs pipelines from a script or a notebook.

## Browsing videos and photos

**Processing** shows your clips by day, filtered by hotel, date, hour of day and search.
Switch to **Photos** for the periodic full-resolution photos.

<!-- SCREENSHOT: the Processing hub -->

- Hover a clip to preview it; click to open it.
- Select clips, choose a pipeline, **Run pipeline** ([Pipelines](pipelines.md)).

### A clip's page

The player, its resolution, frame rate and length, the **photos taken at the trigger** (when motion
photos are on) and the clip's analysis runs. **Delete from device** frees its space on the card;
**Delete video** removes it from the cloud too.

### A photo's page

**Fit** shows the whole photo; **100%** loads the original — click where to zoom and drag to pan.
**Download original** saves the full-resolution file.
