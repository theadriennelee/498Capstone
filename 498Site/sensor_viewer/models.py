from django.db import models

# Create your models here.
class Sensor(models.Model):
    name = models.TextField()
    display_name = models.TextField()
    sensor_threshold = models.FloatField()
    status = models.IntegerField()


class SensorData(models.Model):
    sensor = models.ForeignKey(Sensor, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    value = models.FloatField()
    error = models.FloatField()
    anomaly = models.BooleanField()

    class Meta:
        unique_together = (("sensor_id", "timestamp"))


def get_latest_timestamp():
    """
    Get the most recent timestamp in the DB
    :return:
    """
    try:
        recent = SensorData.objects.latest("timestamp")
    except:
        return None
    return recent.timestamp


def register_sensor(name, display, threshold):
    sensor = Sensor.objects.create(name=name, display_name=display, sensor_threshold=threshold, status=0)
    return sensor


def push_sensor_data(sensor, timestamp, value, error, anomaly):
    data = SensorData.objects.create(sensor=sensor, timestamp=timestamp, value=value, error=error, anomaly=anomaly)


def sensor_exists(name):
    try:
        sensor = Sensor.objects.get(name=name)
        return sensor
    except:
        return False