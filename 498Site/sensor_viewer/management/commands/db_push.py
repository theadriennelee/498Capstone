from django.core.management.base import BaseCommand, CommandError
import os
import pandas as pd
from datetime import datetime
from dateutil import parser
from sensor_viewer import models
#from polls.models import Question as Poll

class Command(BaseCommand):
    help = "Pushes data from a report to the DB"

    def add_arguments(self, parser):
        parser.add_argument("timesteps", type=int)

    def handle(self, *args, **options):
        # Retrieve the data
        path = os.path.join(os.getcwd(), "data", "demo_report.csv")
        sensor_data = pd.read_csv(path)
        records_remaining = options["timesteps"]

        # Parse through it and push records to the database
        for index, row in sensor_data.iterrows():
            timestamp = datetime.strptime("{} +0000".format(row["date"]), "%Y-%m-%d %H:%M:%S %z")
            latest_timestamp = models.get_latest_timestamp()
            # Extract the rows to push
            if latest_timestamp:
                if latest_timestamp >= timestamp:
                    continue
            values = row.tolist()[1:]
            for i in range(0, int(len(values) / 3)):
                sensor_name = sensor_data.columns[i*3+1]
                # If the sensor doesn't exist in the table (check columns), add it to the table
                sensor = models.sensor_exists(sensor_name)
                if not sensor:
                    sensor = models.register_sensor(sensor_name, sensor_name, 2.0)

                # Push the data
                models.push_sensor_data(sensor, timestamp, values[i*3], values[i*3+1], values[i*3+2])
            records_remaining -= 1
            if records_remaining < 1:
                print("%d record(s) pushed", options["timesteps"])
                break


