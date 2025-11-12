import time
import datetime


def general_engine(worker, configDict, startTime, initResult, dataGrid):
    start = time.time()
    workerResult = worker(config=configDict, initResult=initResult, timeStamp=startTime)
    end = time.time()
    print(f"Took {end - start}s ({datetime.timedelta(seconds=end - start)})")
    return workerResult