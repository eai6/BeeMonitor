# Set up & deploy

You need an account on the [BeeMonitor platform](https://beemonitor.edwardamoah.com), the assembled
unit, its microSD card, and a computer with [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
No terminal needed.

## 1. Download the image

On the platform, go to **Devices → Add a device** and download the BeeMonitor image (`.img.xz`).

## 2. Flash the card

In Raspberry Pi Imager: **Choose OS → Use custom image** → the downloaded file. In the settings (gear),
set a hostname, your **WiFi** network and your locale. Write the card and leave it in the computer.

## 3. Enrol the card

Back on **Devices → Add a device**: **Generate token → Choose SD card & write**, and pick the card's
`bootfs` drive. This links the card to your account.

!!! note
    Writing the token needs Chrome or Edge. On Safari or Firefox, use the command shown on that page.

## 4. Power on

Put the card in the unit and power it on. On first boot it registers itself and appears on your
**Devices** page within a few minutes.

## 5. Check it

On the device page:

1. The unit shows **Online**, with storage, temperature and connection.
2. **Take photo** shows the hotel.
3. **Edit ROI & reference objects** — draw the hotel and its nest tubes.
4. **Autofocus**, then **Take photo** again to check it's sharp.
5. Wave at the camera: a clip appears under the device's videos once uploaded.

## Updates

When a new version is out, the device page offers **Update**. The unit installs it on its next check-in.

## 4G (optional)

A unit with a Sixfab 4G HAT needs its modem configured once — see
[Cellular connectivity](https://github.com/eai6/BeeMonitor/blob/main/hardware/README.md#step-10-cellular-connectivity-sixfab-4g-lte).
Over 4G the unit sends health and takes commands; clips and photos wait for WiFi.

## Field deployment

1. **Set up the unit first** on WiFi at home or in the lab (the steps above),
   so you know it records before it goes out.
2. **Mount it** on the tripod facing what you study (see the table below), with the sun behind the camera if
   you can.
3. **Check the picture** on the device page: **Take photo**, then **Edit ROI & reference objects** to draw
   the ROI (and, at a hotel, its nest tubes), then **Autofocus**, then **Take photo** again.
4. **Set the recording window** to the hours insects are active (for example 8:00–18:00) on the device page.
5. **Power**: on solar, put the battery box in shade and the panel facing south (in the northern hemisphere).

### Placement

| Setting | Distance | ROI | Notes |
|---|---|---|---|
| Nest hotel | 50–100 cm, square to the hotel face | The hotel; each tube as a reference object | The validated setup |
| Flower / patch | 30–60 cm, or closer for small flowers | The flowers only, not the surrounding foliage | Wind on leaves causes false triggers, so keep the ROI tight |
| Indoor assay | Fixed above or beside the arena | The arena | Use mains power, even lighting, and *Continuous* mode for timed trials |

<!-- PHOTO: a unit deployed at a hotel, with the solar panel -->

!!! tip
    A unit with 4G sends health and takes commands from anywhere; clips and photos wait on the card
    until it next has WiFi. Without 4G, visit the site with a phone hotspot or collect the card.
