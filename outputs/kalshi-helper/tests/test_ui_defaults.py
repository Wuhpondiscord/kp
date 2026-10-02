"""Catch browser number constraints that silently prevent form submission."""
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
import unittest


class Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'input' and values.get('type') == 'number':
            self.inputs.append(values)


class DefaultConstraintTests(unittest.TestCase):
    def test_all_static_numeric_defaults_are_browser_valid(self):
        root = Path(__file__).resolve().parents[1] / 'helper' / 'static'
        count = 0
        for name in ['home.html', 'training.js']:
            parser = Inputs()
            parser.feed((root / name).read_text(encoding='utf-8'))
            for field in parser.inputs:
                if 'value' not in field:
                    continue
                count += 1
                with self.subTest(file=name, field=field.get('id')):
                    value = Decimal(field['value'])
                    if 'min' in field:
                        self.assertGreaterEqual(value, Decimal(field['min']))
                    if 'max' in field:
                        self.assertLessEqual(value, Decimal(field['max']))
                    if field.get('step') != 'any':
                        step = Decimal(field.get('step', '1'))
                        base = Decimal(field.get('min', field['value']))
                        self.assertEqual((value - base) % step, 0)
        self.assertGreaterEqual(count, 12)
