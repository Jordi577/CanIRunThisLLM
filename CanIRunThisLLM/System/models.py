from django.db import models

class LLMMapping(models.Model):
    name = models.CharField(max_length=255, primary_key=True)
    model_config = models.JSONField()                    
    parameters = models.BigIntegerField()
    quant_level = models.CharField(max_length=10)
    context_window = models.IntegerField()
    cache_bit = models.IntegerField()
    cuda_overhead = models.IntegerField()

    def __str__(self):
        return self.name
    
class AppleMSeriesProcessor(models.Model):
    name = models.CharField(max_length=255, primary_key=True)
    bandwidth = models.FloatField()
