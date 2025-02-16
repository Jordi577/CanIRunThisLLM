import json
import os
from django.core.management.base import BaseCommand
from System.models import LLMMapping

class Command(BaseCommand):
    help = "Import LLM mappings from a JSON file containing a list of mappings."

    def handle(self, *args, **options):
        # Build the absolute path to the JSON file.
        current_dir = os.path.dirname(__file__)
        json_file_path = os.path.join(current_dir, 'json', 'llm_database.json')
        self.stdout.write(f"Loading JSON from: {json_file_path}")

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            self.stderr.write(f"File not found: {json_file_path}")
            return
        except json.JSONDecodeError as e:
            self.stderr.write(f"Error decoding JSON: {e}")
            return

        self.stdout.write("JSON loaded successfully.")

        # Since the JSON is a list, iterate over each mapping.
        for item in data:
            name = item.get('name')
            if not name:
                self.stderr.write(f"Skipping item with no 'name': {item}")
                continue

            self._create_or_update_mapping(name, item)

        self.stdout.write(self.style.SUCCESS("LLM mappings imported successfully."))

    def _create_or_update_mapping(self, name, details):
        # Convert 'parameters' to an integer if possible.
        parameters = details.get('parameters', 0)
        try:
            parameters = int(parameters)
        except (ValueError, TypeError):
            parameters = 0

        mapping, created = LLMMapping.objects.get_or_create(
            name=name,
            defaults={
                'model_config': details.get('model_config', {}),
                'parameters': parameters,
                'quant_level': details.get('quant_level', ''),
                'context_window': details.get('context_window', 0),
                'cache_bit': details.get('cache_bit', 0),
                'cuda_overhead': details.get('cuda_overhead', 0)
            }
        )

        if not created:
            # Update existing mapping.
            mapping.model_config = details.get('model_config', {})
            mapping.parameters = parameters
            mapping.quant_level = details.get('quant_level', '')
            mapping.context_window = details.get('context_window', 0)
            mapping.cache_bit = details.get('cache_bit', 0)
            mapping.cuda_overhead = details.get('cuda_overhead', 0)
            mapping.save()
