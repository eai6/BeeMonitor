# Assembly

![Recording module](../assets/recording_module.png)

## Recording module

1. Screw the **Raspberry Pi** onto the standoffs in the enclosure body.
2. Press the **Witty Pi 4** onto the Pi's GPIO header. Line up all 40 pins.
3. Connect the **camera**:

    === "HQ camera / 64 MP OwlSight"
        Lift the latch of the Pi's camera port, insert the ribbon with the contacts facing the HDMI
        ports, and press the latch down. Mount the camera in the lid's window.

    === "Luxonis OAK-1-AF"
        Plug the OAK into a **blue (USB 3)** port. Mount it in the lid's window the right way up —
        the OAK can't flip its picture.

4. Mount the **DC-DC converter** in the body and wire its 5 V output to the Witty Pi's power input.
5. Run the power cable through the cable gland and tighten it.
6. Fit the lid, but don't seal it until the unit records well ([Set up a unit](../software/setup.md)).

<!-- PHOTO: one per step -->

## Energy module

![Energy module](../assets/energy_module.png)

!!! warning "Connect in this order"
    1. Battery to the charge controller (**BAT**) — always first.
    2. Solar panel to the controller (**PV**).
    3. Controller load output to the recording module's DC-DC input.

    Connecting the panel before the battery can damage the controller.

## Weatherproofing

- Seal cable entries with silicone.
- Make sure the cable glands are tight.
- Conformal coating on exposed boards is optional.

**Next:** [Field deployment](deployment.md)
