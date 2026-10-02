import unittest
from helper.coverage import coverage_issue,training_coverage
from helper.live import practice_candidates


class CoverageTests(unittest.TestCase):
    def test_specific_reasons_and_legacy_scope(self):
        self.assertIn('Too early',coverage_issue({},'KXHIGHNY','2026-09-20T00:00Z','2026-09-22T00:00Z'))
        self.assertIn('Too late',coverage_issue({},'KXHIGHNY','2026-09-20T00:00Z','2026-09-20T02:00Z'))
        self.assertIn('Unsupported market',coverage_issue({},'MENTIONS','2026-09-20T00:00Z','2026-09-20T12:00Z'))
        self.assertIsNone(coverage_issue({},'KXHIGHNY','2026-09-20T00:00Z','2026-09-20T12:00Z'))

    def test_new_scope_requires_data_in_every_partition(self):
        parts=[[dict(series='KXHIGHNY',hours_left=h) for h in bounds] for bounds in ([3,27],[4,26],[3,25])]
        scope=training_coverage(parts)
        self.assertEqual(scope['min_hours'],4);self.assertEqual(scope['max_hours'],25)
        self.assertIsNotNone(coverage_issue({'coverage':scope},'KXHIGHNY','2026-09-20T00:00Z','2026-09-20T03:00Z'))

    def test_practice_ranking_uses_selected_models_scope(self):
        markets=[dict(ticker='outside',series='KXHIGHNY',close_time='2026-09-20T12:00Z',bid=.3,ask=.31,spread=.01),dict(ticker='inside',series='KXHIGHNY',close_time='2026-09-20T04:00Z',bid=.3,ask=.4,spread=.1)]
        active={'coverage':dict(series=['KXHIGHNY'],min_hours=3,max_hours=5)}
        self.assertEqual(practice_candidates(markets,'2026-09-20T00:00Z',active)[0]['ticker'],'inside')

    def test_disjoint_partition_windows_are_rejected(self):
        parts=[[dict(series='KXHIGHNY',hours_left=h) for h in bounds] for bounds in ([3,5],[12,18],[24,26])]
        with self.assertRaisesRegex(ValueError,'No time window'):
            training_coverage(parts)

    def test_scope_boundaries_are_inclusive_and_not_rounded_outward(self):
        parts=[[dict(series='KXHIGHNY',hours_left=h) for h in (4.0000004,26.9999996)] for _ in range(3)]
        scope=training_coverage(parts)
        self.assertEqual(scope['min_hours'],4.0000004)
        self.assertEqual(scope['max_hours'],26.9999996)
        self.assertIsNotNone(coverage_issue({'coverage':scope},'KXHIGHNY','2026-09-20T00:00Z','2026-09-20T04:00Z'))

    def test_supported_contracts_take_priority_over_liquid_unsupported_contracts(self):
        markets=[dict(ticker='outside',series='KXHIGHNY',close_time='2026-09-22T00:00Z',bid=.3,ask=.31,spread=.01),dict(ticker='inside',series='KXHIGHNY',close_time='2026-09-20T12:00Z',bid=0,ask=.01,spread=.01)]
        self.assertEqual(practice_candidates(markets,'2026-09-20T00:00Z')[0]['ticker'],'inside')
