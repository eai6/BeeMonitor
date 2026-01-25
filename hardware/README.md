# BeeHotelMonitoring Project

Object: The object of this project is to monitor solitary bee activity in and out of bee hotels

# Manual

## 1. Download Source Code into your desktop
```
cd ~/Desktop
git clone https://github.com/eai6/BeeHotelMonitoring.git
cd BeeHotelMonitoring/beeHotelCode
```

## 2. Install dependancies using terminal

### 2.1 Update and Upgrade 
```
sudo apt update
sudo apt upgrade
```
### 2.2 Install picamera
```
sudo apt install -y python3-picamera2
```
### 2.3 Install opencv
```
sudo apt install python3-opencv
```

## 3. Create program directories
```
python3 makeDirectories.py
```

## 4. RunFocus to check and focus camera

```
python3 runFocus.py
```
Note: Make sure the camera was connected when you turn on the Raspberry PI. If it was not connected on boot, then you should reboot the system.

```
sudo reboot
```

## 5. Set-up a service to run driver for monitoring
### Create Service file
```
sudo nano /lib/systemd/system/beeHotelRecord.service 
```

### Paste the code below

```
[Unit]
Description=beeHotel
After=multi-user.target

[Service]
Type=idle
ExecStart=/usr/bin/python /home/apis/Desktop/BeeHotelMonitoring/beeHotelCode/driver.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Update Deamon
```
sudo systemctl daemon-reload
```

## 6. Test BeeHotelMonitoring Service

### 6.1 Start the beeHotelMonitoring Service
```
sudo systemctl start beeHotelRecord.service
```
### 6.2 Check status of service
```
sudo systemctl status beeHotelRecord.service
```
### 6.3 Command to stop service
```
sudo systemctl stop beeHotelRecord.service
```


## 7. Enable BeeHotelMonitoring Service on boot
```
sudo systemctl enable beeHotelRecord.service
```

## 8. Set-up WittyPi

### 8.1 Get WittyPi
```
cd ~/Desktop
wget http://uugear.com/repo/WittyPi4/install.sh
sudo sh install.sh
```

### 8.2 Set the recorder to launch at startup
```
nano /home/apis/Desktop/wittypi/afterStartup.sh 
```
Add the following:
```
sudo python /home/apis/Desktop/BeeHotelMonitoring/beeHotelCode/driver.py
```

### 8.3 Allow IC2
```
sudo raspi-config
```
Interface Options >> I5 I2C >> YES >> OK

### 8.3 Test the WittyPi through 1 cycle
Open terminal 
```
cd ~/Desktop/wittypi/
sudo ./wittyPi.sh
```

First Select (1) to write system time to RTC on the wittiPI (i.e., assuming the time on the Raspberry PI is accurate and the time on the RTC is not. You can set this up when configuring the OS). If it is the other way around then you should select (2)

Schedule next startup (4): suggest something relatively soon like ?? ??:01:00 (one minute past the next hour. Change the minute to something reasonable)

Schedule next shutdown (5): suggest something 60 seconds before the startup like ?? ??:00 (at the next hour)

Wait for Pi to shutdown and restart 60 seconds later

### 8.4 Set up actual WP script
Create Scheduler
```
sudo nano /home/apis/Desktop/wittypi/schedules/beeHotelScheduler_2024.wpi
```
Paste the code below

```
BEGIN 2024-03-00 07:50:00
END   2024-09-01 00:00:00
ON    H10 M15 # will start recording from 7:50am to 6:05pm
OFF   H13 M45 # will be will be off until the next day
```

### Select custom WittyPi Schedule
```
cd ~/Desktop/wittypi/
sudo ./wittyPi.sh
```
Choose schedule script (6)

Pick the beeHotelSchedulor script

Verify that the next power on/off times make sense

## 9. Set-Up VCN

### 9.1 Allow VNC
```
sudo raspi-config
```
Navigate to Interfacing Options > VNC and enable VNC.

### 9.2 Sign In VNC
1. Enter email and password
2. Allow cloud and direct connection
3. Authenticate with Unix password
4. Encryption with at leat 128-bit
5. Allow all users 


## Set-up Wifi Access Point

### Get dnsmasq and hostapd
```
sudo apt install dnsmasq hostapd
```

### Stop dnsmasq and hostapd until configuration is set
```
sudo systemctl stop dnsmasq
sudo systemctl stop hostapd
```

### Configure static IP
```
sudo nano /etc/dhcpcd.conf
```

Paste the code below at the bottom

```
# Wifi Access Point Config
interface wlan0
static ip_address=192.168.0.10/24
nohook wpa_supplicant
```

### Restart 
```
sudo service dhcpcd restart
```

### Configure DCHPC
```
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.orig
sudo nano /etc/dnsmasq.conf
```

Paste the code below
```
interface=wlan0
dhcp-range=192.168.0.11,192.168.0.30,255.255.255.0,24h
```


### Configure the access point host software
Now it is time to configure the access point software:
```
sudo nano /etc/hostapd/hostapd.conf
```
Add the below information to the configuration file:
```
country_code=US
interface=wlan0
ssid=YOURSSID
channel=9
auth_algs=1
wpa=2
wpa_passphrase=YOURPWD
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP CCMP
rsn_pairwise=CCMP
```
Make sure to change the ssid and wpa_passphrase. We now need to tell the system where to find this configuration file. Open the hostapd file:
```
sudo nano /etc/default/hostapd
```
Find the line with #DAEMON_CONF, and replace it with this:
DAEMON_CONF="/etc/hostapd/hostapd.conf"

### Start up the wireless access point
Run the following commands to enable and start hostapd:

```
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl start hostapd
```

## Managing Access Point
### Disable access point 
```
sudo systemctl disable hostapd dnsmasq
```
comment the static ip config in 
```
sudo nano /etc/dhcpcd.conf
```
Reboot
```
sudo reboot
```
### Enable access point
```
sudo systemctl enable hostapd dnsmasq
```
uncomment the static IP config in 
```
sudo nano /etc/dhcpcd.conf
```
Reboot
```
sudo reboot
```
