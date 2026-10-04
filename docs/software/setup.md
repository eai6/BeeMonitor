# Set up a unit

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
