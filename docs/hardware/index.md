# Hardware overview

A BeeMonitor unit is a camera that records only when insects are active and uploads the clips. It works at
a nest hotel, by a flower patch or on a lab bench. You don't need one to use the platform, because
[uploaded video](../platform/sources.md) works too, but a unit gives you motion-triggered recording,
remote control and saved layouts.

A BeeMonitor unit has two parts:

- **Recording module** (~$350) — a Raspberry Pi 4, a camera and a Witty Pi 4 in a 3D-printed enclosure.
  It records only when something moves and uploads the clips.
- **Energy module** (~$245, optional) — a solar panel, charge controller and battery for sites without mains power.

![System architecture](../assets/hardware_architecture.png)

## Choose a configuration

| | Options |
|---|---|
| **Computer** | Raspberry Pi 4, **2 GB or more** (4 GB recommended). A 1 GB Pi records fine but can't take 64 MP photo bursts. |
| **Camera** | Raspberry Pi HQ camera (12 MP, 1080p video) · Arducam 64 MP OwlSight (autofocus, 64 MP photos) · Luxonis OAK-1-AF (USB, autofocus, 12 MP video and photos) |
| **Power** | Mains (USB-C) · Solar panel + battery (energy module) |
| **Connection** | WiFi · WiFi + 4G (Sixfab LTE HAT) for remote sites |

!!! tip
    Start with mains power and WiFi on your bench. Add solar and 4G once the unit records well.

**Next:** [Components](components.md)
