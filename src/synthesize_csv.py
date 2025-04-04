from datetime import time, datetime, timedelta

def getStartTime(filename):
    site, date, hour, minutes, seconds = filename.split('.')[0].split('_')

    # Original time object
    original_time = time(int(hour), int(minutes), int(seconds))  # 2:30:45 PM

    #from datetime import datetime

    # Convert to a datetime object
    date = datetime.strptime(date, "%Y-%m-%d")
    base_datetime = datetime.combine(date, original_time)

    return base_datetime

def getTimestamp(frame_number, base_datetime, fps=30):
    return base_datetime + timedelta(seconds= int(frame_number/fps))


def processCSV(events, filename):
    base_datetime = getStartTime(filename)
    events['timestamp'] = events['frame_number'].apply(lambda x: getTimestamp(x, base_datetime))
    events['filename'] = filename
    return events[['timestamp', 'nest', 'action']]