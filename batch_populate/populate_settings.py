import datetime
settings = {"display_progress": True, 
            "reserve_jobs": True,
            "suppress_errors": True,
            "order": "random",
            "processes" : 10}

def now():
    return datetime.datetime.today().strftime("%Y-%m-%d | %H:%M:%S")