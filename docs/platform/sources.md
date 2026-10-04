# Getting video in

You don't need a BeeMonitor unit to use the platform. Video from any camera can be analysed.

## From a BeeMonitor unit

An enrolled unit uploads its clips and photos by itself (see [Set up a unit](../software/setup.md)). They
appear under **Processing**, already linked to the device, its location and the layout it was recorded with.

## Upload clips

**Processing → Upload a video** (or **Upload Videos** for several files at once).

- Formats: `.mp4`, `.mov`, `.mkv`, `.h264`, up to 5 GiB each. Files go straight to storage from your browser.
- **Device (optional)**: attribute the clip to one of your devices, for example a copy from its SD card.
  The clip then shows on the device page and uses the device's location.
- **Site (optional)**: for clips not from a device. Groups the clip under a site in the Processing filters.

If a file name contains a timestamp, it is used as the recording time.

## Connect cloud storage

For footage that already lives in a bucket or a shared folder, go to **Sources → Add Data Source**:

| Source | You provide |
|---|---|
| AWS S3 | Bucket, optional key prefix, and an access key for a **read-only** IAM user. The form generates the policy. |
| Google Cloud Storage | Bucket and a service-account JSON key with read access |
| Google Drive | The folder ID (the last part of the folder's URL) and an OAuth token |

Then **Browse** the source, select files (or all of them) and import them. Files already imported are skipped,
so you can import again as new footage arrives. Credentials are stored encrypted.

!!! warning
    Give BeeMonitor read-only access to just the bucket it needs. Never use an admin key or
    `AmazonS3FullAccess`.

## From code

The [API](api.md) uploads clips and runs pipelines from a script or a Colab notebook.
