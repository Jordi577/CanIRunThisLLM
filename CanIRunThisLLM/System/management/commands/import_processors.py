import json
import os
from django.core.management.base import BaseCommand
from System.models import AppleMSeriesProcessor

class Command(BaseCommand):
    help = "Import Apple M-Series processors from JSON"

    def handle(self, *args, **options):
        # Build the path relative to this file's location.
        current_dir = os.path.dirname(__file__)
        json_file_path = os.path.join(current_dir, 'json', 'm_series_database.json')
        
        self.stdout.write(f"Loading JSON from: {json_file_path}")
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            self.stderr.write(f"File not found: {json_file_path}")
            return

        self.stdout.write("JSON loaded successfully.")

        for item in data:
            for processor_name, details in item.items():
                if 'Bandwidth' in details:
                    bandwidth = int(float(details['Bandwidth']))
                    self._create_processor(processor_name, bandwidth)

                else:
                    for variant, variant_details in details.items():
                        if 'Bandwidth' in variant_details:
                            composite_name = f"{processor_name} {variant}"
                            bandwidth = int(float(variant_details['Bandwidth']))
                            self._create_processor(composite_name, bandwidth)

        self.stdout.write(self.style.SUCCESS('Processors imported successfully.'))

    def _create_processor(self, name, bandwidth):
        processor, created = AppleMSeriesProcessor.objects.get_or_create(
            name=name,
            defaults={'bandwidth': bandwidth}
        )

        if not created:
            processor.bandwidth = bandwidth
            processor.save()
