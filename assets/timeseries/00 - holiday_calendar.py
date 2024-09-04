import pandas.tseries.holiday as holidays
from datetime import date, timedelta, datetime
import pandas as pd

def holiday_calendar(df):
    """Create holiday calendar"""
    pd.options.mode.chained_assignment = None
    us_holidays = holidays()
    holiday_cal = us_holidays.holidays(start=(df.Date.min()), end=(df.Date.max()), return_name=True)
    holiday_cal = holiday_cal.reset_index(name='Holiday').rename(columns={'index': 'Date'})
    black_friday = holiday_cal[(holiday_cal['Holiday'] == 'Thanksgiving')]
    black_friday['Date'] = black_friday['Date'] + timedelta(days=1)
    black_friday['Holiday'] = 'Black Friday'
    cyber_monday = holiday_cal[(holiday_cal['Holiday'] == 'Thanksgiving')]
    cyber_monday['Date'] = black_friday['Date'] + timedelta(days=3)
    cyber_monday['Holiday'] = 'Cyber Monday'
    holiday_cal = holiday_cal.append(black_friday, ignore_index=True)
    holiday_cal = holiday_cal.append(cyber_monday, ignore_index=True)
    return holiday_cal