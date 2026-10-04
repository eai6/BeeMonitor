# Troubleshooting

??? question "The unit doesn't appear on my Devices page"
    Give it a few minutes on first boot. Check that the card was enrolled (step 3 of
    [Set up a unit](setup.md)) and that the WiFi name and password in Raspberry Pi Imager were right.
    Re-flashing and re-enrolling the same Pi keeps it as the same device.

??? question "The device shows Offline"
    It has stopped checking in: no power, no WiFi/4G, or the Witty Pi schedule has it switched off.
    Health tiles stay blank while it's offline; the history chart shows when it was last on.

??? question "Videos are blurry"
    Draw the ROI first, then press **Autofocus** on the device page and **Take photo** to check. Autofocus
    looks at the ROI (or the centre if none is drawn) and keeps that focus across restarts.
    **Reset focus** starts over. Focus in daylight.

??? question "No clips are recorded"
    Check the **recording window** (hours) and that recording isn't **Off**. Motion only counts inside
    the ROI: if the ROI is drawn too small, bees can't trigger a clip.

??? question "Clips stay on the device"
    They upload over WiFi only. A unit on 4G keeps them on the card until it next has WiFi.

??? question "The SD card is filling up"
    Uploaded clips and photos stay on the card until you clear them. Use **Free space on the device** on
    the device page; the cloud copies are kept.

??? question "A pipeline run finds no nest visits"
    Draw the ROI and the nest tubes (reference objects) for the device. Without them the run still tracks
    bees, but it can't tell which nest they enter.

??? question "Photos say \"16 MP fallback\""
    The Pi couldn't spare the memory for a 64 MP frame. It happens on 1 GB Pis; use a 2 GB or larger Pi.
