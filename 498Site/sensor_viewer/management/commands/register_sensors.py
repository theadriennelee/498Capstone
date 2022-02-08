from django.core.management.base import BaseCommand, CommandError
import os
import pandas as pd
from datetime import datetime
from sensor_viewer import models
#from polls.models import Question as Poll

class Command(BaseCommand):
    help = "Pushes data from a report to the DB"

    def handle(self, *args, **options):
        # Retrieve the data
        sensors = ["T1", "T2", "T3", "T4", "T5", "T6"]
        for sensor in sensors:
            models.register_sensor(sensor, sensor, 2.0)